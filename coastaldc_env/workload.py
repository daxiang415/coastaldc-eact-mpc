"""Workload dynamics: fixed + flexible split, backlog queue with delay window, SLA penalty.

All loads are expressed in MW (average power over one hourly timestep, so MW == MWh/step).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WORKLOAD_TOL_MWH = 1e-9


@dataclass
class WorkloadConfig:
    it_capacity_mw: float = 10.0          # rated IT compute capacity
    flexible_fraction: float = 0.3        # share of arriving load that is shiftable
    max_delay_hours: int = 24             # SLA delay window for deferred work
    sla_penalty_per_mwh: float = 1.0      # penalty weight per MWh of deadline violation
    enforce_deadlines: bool = True        # serve expiring work first when capacity exists
    enforce_episode_terminal: bool = True # prevent controllable backlog at episode end


@dataclass
class WorkloadState:
    backlog_mwh: float = 0.0
    # age-bucketed backlog queue: queue[k] = MWh deferred k hours ago
    queue: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.queue is None:
            self.queue = np.zeros(0)


class WorkloadModel:
    """Tracks flexible-workload backlog with an age queue and enforces the delay window.

    Action semantics (a_workload in [-1, 1]):
      -1 : defer the entire flexible arrival of this hour;
       0 : execute exactly the flexible arrival (no deferral, no recovery);
      +1 : execute arrival plus recover as much backlog as capacity allows.
    """

    def __init__(self, config: WorkloadConfig | None = None):
        self.cfg = config or WorkloadConfig()
        self.reset()

    def reset(self):
        self.queue = np.zeros(self.cfg.max_delay_hours)  # index = age in hours
        self.total_arrived = 0.0
        self.total_executed = 0.0
        self.total_violated = 0.0
        self.last_mandatory_recovery_mwh = 0.0

    @property
    def backlog_mwh(self) -> float:
        return float(self.queue.sum())

    def deadline_pressure(self) -> float:
        """Weighted urgency in [0, 1]: backlog close to its deadline weighs more."""
        if self.backlog_mwh <= 1e-9:
            return 0.0
        ages = np.arange(len(self.queue))
        w = (ages + 1) / len(self.queue)
        return float(np.clip((self.queue * w).sum() / max(self.backlog_mwh, 1e-9), 0.0, 1.0))

    def _predicted_queue_after_action(self, a_workload: float,
                                      fixed_load_mw: float,
                                      flexible_arrival_mw: float) -> tuple[np.ndarray, bool]:
        """Predict the post-step age queue without mutating model state."""
        cfg = self.cfg
        a = float(np.clip(a_workload, -1.0, 1.0))
        queue = self.queue.copy()
        headroom = max(0.0, cfg.it_capacity_mw - fixed_load_mw)
        expiring = float(queue[-1]) if queue.size else 0.0
        mandatory = min(expiring, headroom) if cfg.enforce_deadlines else 0.0
        self._drain_queue_oldest_first(queue, mandatory)
        headroom -= mandatory

        if a >= 0.0:
            exec_arrival = min(flexible_arrival_mw, headroom)
            deferred_now = flexible_arrival_mw - exec_arrival
            recover_room = max(0.0, headroom - exec_arrival)
            recovery = min(a * float(queue.sum()), recover_room)
            self._drain_queue_oldest_first(queue, recovery)
        else:
            exec_arrival = min((1.0 + a) * flexible_arrival_mw, headroom)
            deferred_now = flexible_arrival_mw - exec_arrival

        violated = float(queue[-1]) if queue.size else 0.0
        if queue.size:
            queue[1:] = queue[:-1]
            queue[0] = deferred_now
        return queue, violated <= 1e-9

    @staticmethod
    def _drain_queue_oldest_first(queue: np.ndarray, amount_mwh: float):
        remaining = max(0.0, float(amount_mwh))
        for k in range(len(queue) - 1, -1, -1):
            take = min(float(queue[k]), remaining)
            queue[k] -= take
            remaining -= take
            if remaining <= 1e-12:
                break

    def _queue_recoverable(self, queue: np.ndarray,
                           future_spare_mwh: np.ndarray) -> bool:
        """Check every deadline while recovering oldest work first."""
        projected = queue.copy()
        for spare in future_spare_mwh:
            self._drain_queue_oldest_first(projected, spare)
            if projected.size and projected[-1] > 1e-9:
                return False
            if projected.size:
                projected[1:] = projected[:-1]
                projected[0] = 0.0
        return float(projected.sum()) <= 1e-9

    def project_recoverable_action(self, a_workload: float,
                                   fixed_load_mw: float,
                                   flexible_arrival_mw: float,
                                   future_spare_mwh) -> tuple[float, bool, bool]:
        """Project an action into the forecast deadline-recoverable set."""
        requested = float(np.clip(a_workload, -1.0, 1.0))
        if not self.cfg.enforce_episode_terminal:
            return requested, False, True

        future_spare = np.atleast_1d(np.asarray(
            future_spare_mwh, dtype=float)).clip(min=0.0)
        requested_queue, deadline_feasible = self._predicted_queue_after_action(
            requested, fixed_load_mw, flexible_arrival_mw)
        if deadline_feasible and self._queue_recoverable(
                requested_queue, future_spare):
            return requested, False, True

        full_queue, full_deadline_feasible = self._predicted_queue_after_action(
            1.0, fixed_load_mw, flexible_arrival_mw)
        physically_feasible = (
            full_deadline_feasible
            and self._queue_recoverable(full_queue, future_spare))
        if not physically_feasible:
            return 1.0, abs(1.0 - requested) > 1e-9, False

        low, high = requested, 1.0
        for _ in range(16):
            mid = 0.5 * (low + high)
            queue, deadline_ok = self._predicted_queue_after_action(
                mid, fixed_load_mw, flexible_arrival_mw)
            if deadline_ok and self._queue_recoverable(queue, future_spare):
                high = mid
            else:
                low = mid
        applied = float(high)
        return applied, abs(applied - requested) > 1e-7, True

    def project_terminal_action(self, a_workload: float,
                                remaining_hours: int) -> tuple[float, bool]:
        """Compatibility helper for callers without capacity forecasts."""
        if remaining_hours > self.cfg.max_delay_hours:
            requested = float(np.clip(a_workload, -1.0, 1.0))
            return requested, False
        applied, intervened, _ = self.project_recoverable_action(
            a_workload, fixed_load_mw=0.0, flexible_arrival_mw=0.0,
            future_spare_mwh=0.0)
        return applied, intervened

    def step(self, a_workload: float, fixed_load_mw: float, flexible_arrival_mw: float):
        """Advance one hour. Returns (executed_it_load_mw, sla_violation_mwh, executed_flexible_mw)."""
        cfg = self.cfg
        a = float(np.clip(a_workload, -1.0, 1.0))
        self.total_arrived += flexible_arrival_mw

        headroom = max(0.0, cfg.it_capacity_mw - fixed_load_mw)

        # Work reaching its deadline has priority over new flexible arrivals.
        # A violation is then caused only by insufficient physical capacity.
        mandatory_recovery = 0.0
        if cfg.enforce_deadlines and self.queue.size:
            mandatory_recovery = min(float(self.queue[-1]), headroom)
            self._drain_oldest_first(mandatory_recovery)
            headroom -= mandatory_recovery
        self.last_mandatory_recovery_mwh = mandatory_recovery

        if a >= 0.0:
            # execute full arrival + recover fraction `a` of backlog (capacity-limited)
            exec_arrival = min(flexible_arrival_mw, headroom)
            deferred_now = flexible_arrival_mw - exec_arrival
            recover_room = max(0.0, headroom - exec_arrival)
            desired_recovery = a * self.backlog_mwh
            recovery = min(desired_recovery, recover_room)
            self._drain_oldest_first(recovery)
        else:
            # defer fraction |a| of the flexible arrival
            exec_arrival = min((1.0 + a) * flexible_arrival_mw, headroom)
            deferred_now = flexible_arrival_mw - exec_arrival
            recovery = 0.0

        # age the queue; anything that exceeds the window is an SLA violation
        violated = float(self.queue[-1])
        self.queue[1:] = self.queue[:-1]
        self.queue[0] = deferred_now
        self.total_violated += violated

        executed_flexible = mandatory_recovery + exec_arrival + recovery
        self.total_executed += executed_flexible
        executed_it_load = fixed_load_mw + executed_flexible

        return executed_it_load, violated, executed_flexible

    def settle_terminal_backlog(self) -> float:
        """Close an episode by classifying all remaining work as unserved.

        Episodic energy comparisons must not receive credit for work deferred
        beyond the accounting boundary. Moving the queue into the violation
        ledger preserves exact workload conservation.
        """
        unserved = self.backlog_mwh
        if unserved <= WORKLOAD_TOL_MWH:
            self.queue.fill(0.0)
            return 0.0
        if unserved > 0.0:
            self.queue.fill(0.0)
            self.total_violated += unserved
        return unserved

    def _drain_oldest_first(self, amount_mwh: float):
        self._drain_queue_oldest_first(self.queue, amount_mwh)

    def sla_penalty(self, violated_mwh: float) -> float:
        return self.cfg.sla_penalty_per_mwh * violated_mwh

    def conservation_error(self) -> float:
        """arrived - executed - backlog - violated; should be ~0 at all times."""
        return self.total_arrived - self.total_executed - self.backlog_mwh - self.total_violated
