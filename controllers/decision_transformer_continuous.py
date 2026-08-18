"""Continuous-action Decision Transformer.

Differences from the discrete DT used in the previous DPT paper:
  - the action head is a regression head (MSE loss), not cross-entropy;
  - predicted actions are tanh-squashed and affine-mapped into the env action box.

Token order per timestep: (return-to-go, state, action); action is predicted from
the state token with full causal context.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from controllers.behavior_cloning import squash_to_action_range


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, max_tokens: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        mask = torch.triu(torch.ones(max_tokens, max_tokens, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        out, _ = self.attn(x, x, x, attn_mask=self.causal_mask[:T, :T], need_weights=False)
        return out


class Block(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, max_tokens: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, max_tokens, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ContinuousDecisionTransformer(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 3, context_len: int = 24,
                 embed_dim: int = 128, n_layers: int = 3, n_heads: int = 4,
                 dropout: float = 0.1, max_ep_len: int = 200):
        super().__init__()
        self.obs_dim, self.act_dim, self.context_len = obs_dim, act_dim, context_len
        max_tokens = 3 * context_len

        self.embed_rtg = nn.Linear(1, embed_dim)
        self.embed_state = nn.Linear(obs_dim, embed_dim)
        self.embed_action = nn.Linear(act_dim, embed_dim)
        self.embed_time = nn.Embedding(max_ep_len, embed_dim)
        self.ln_in = nn.LayerNorm(embed_dim)

        self.blocks = nn.ModuleList(
            [Block(embed_dim, n_heads, max_tokens, dropout) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(embed_dim)
        self.action_head = nn.Linear(embed_dim, act_dim)   # regression head

    def forward(self, states, actions, rtgs, timesteps):
        """states (B,T,obs), actions (B,T,act), rtgs (B,T,1), timesteps (B,T) -> (B,T,act)."""
        B, T, _ = states.shape
        te = self.embed_time(timesteps)
        r = self.embed_rtg(rtgs) + te
        s = self.embed_state(states) + te
        a = self.embed_action(actions) + te

        x = torch.stack([r, s, a], dim=2).reshape(B, 3 * T, -1)   # (rtg, state, action) x T
        x = self.ln_in(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_out(x)

        state_tokens = x[:, 1::3]                                  # hidden at state positions
        return squash_to_action_range(self.action_head(state_tokens))


class DecisionTransformerController:
    name = "decision_transformer"

    def __init__(self, obs_dim: int, target_return: float, context_len: int = 24,
                 model_path: str | None = None, device: str = "cpu", **model_kwargs):
        self.device = device
        self.model = ContinuousDecisionTransformer(
            obs_dim, context_len=context_len, **model_kwargs).to(device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.target_return = target_return
        self.context_len = context_len
        self.reset()

    def reset(self):
        self._states, self._actions, self._rtgs, self._ts = [], [], [], []
        self._rtg = self.target_return
        self._t = 0

    def act(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        self._states.append(np.asarray(obs, dtype=np.float32))
        self._actions.append(np.zeros(3, dtype=np.float32))       # placeholder for current step
        self._rtgs.append(self._rtg)
        self._ts.append(self._t)

        K = self.context_len
        states = torch.as_tensor(np.stack(self._states[-K:]))[None].to(self.device)
        actions = torch.as_tensor(np.stack(self._actions[-K:]))[None].to(self.device)
        rtgs = torch.as_tensor(np.array(self._rtgs[-K:], dtype=np.float32))[None, :, None].to(self.device)
        ts = torch.as_tensor(np.array(self._ts[-K:], dtype=np.int64))[None].to(self.device)

        with torch.no_grad():
            pred = self.model(states, actions, rtgs, ts)
        action = pred[0, -1].cpu().numpy().astype(np.float32)
        self._actions[-1] = action
        self._t += 1
        return action

    def update_reward(self, reward: float):
        """Call after env.step to decrement the return-to-go."""
        self._rtg -= reward


# ---------------------------------------------------------------------- training
def train_dt(model: ContinuousDecisionTransformer, trajectories: list[dict],
             epochs: int = 20, steps_per_epoch: int = 500, batch_size: int = 64,
             lr: float = 1e-4, device: str = "cpu", seed: int = 0) -> list[float]:
    """trajectories: list of dicts with keys 'observations' (T,obs), 'actions' (T,3),
    'rewards' (T,). Trains with MSE action regression on sampled context windows."""
    rng = np.random.default_rng(seed)
    K = model.context_len
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()

    rtgs_all = [np.cumsum(tr["rewards"][::-1])[::-1].astype(np.float32) for tr in trajectories]
    losses = []
    for _ in range(epochs):
        ep_loss = 0.0
        for _ in range(steps_per_epoch):
            S, A, R, TS = [], [], [], []
            for _ in range(batch_size):
                i = rng.integers(len(trajectories))
                tr, rtg = trajectories[i], rtgs_all[i]
                T = len(tr["rewards"])
                start = rng.integers(0, max(T - K, 1))
                end = min(start + K, T)
                s = tr["observations"][start:end]
                a = tr["actions"][start:end]
                r = rtg[start:end]
                t = np.arange(start, end)
                pad = K - len(s)
                if pad:  # left-pad
                    s = np.concatenate([np.zeros((pad, s.shape[1]), np.float32), s])
                    a = np.concatenate([np.zeros((pad, a.shape[1]), np.float32), a])
                    r = np.concatenate([np.zeros(pad, np.float32), r])
                    t = np.concatenate([np.zeros(pad, np.int64), t])
                S.append(s), A.append(a), R.append(r), TS.append(t)
            S = torch.as_tensor(np.stack(S), dtype=torch.float32, device=device)
            A = torch.as_tensor(np.stack(A), dtype=torch.float32, device=device)
            R = torch.as_tensor(np.stack(R), dtype=torch.float32, device=device)[..., None]
            TS = torch.as_tensor(np.stack(TS), dtype=torch.int64, device=device)

            pred = model(S, A, R, TS)
            loss = nn.functional.mse_loss(pred, A)                # continuous regression loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            ep_loss += loss.item()
        losses.append(ep_loss / steps_per_epoch)
    model.eval()
    return losses
