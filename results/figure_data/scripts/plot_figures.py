"""Render six Nature-style main-text figures from one CSV per figure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator


PACKAGE = Path(__file__).resolve().parents[1]
PLOT_DATA = PACKAGE / "plot_data"
OUTPUT = PACKAGE / "output"

MM = 1.0 / 25.4
FIGURE_WIDTH_MM = 183.0

COLORS = {
    "eact_mpc": "#D55E00",
    "static_robust_mpc": "#0072B2",
    "nominal_causal_mpc": "#767676",
    "teal": "#009E73",
    "gold": "#C58A00",
    "ink": "#242424",
    "muted": "#666666",
    "grid": "#D8D8D8",
    "light": "#F4F4F4",
    "white": "#FFFFFF",
}

LABELS = {
    "eact_mpc": "EACT-MPC",
    "static_robust_mpc": "Static Robust MPC",
    "nominal_causal_mpc": "Nominal MPC",
}
CONTROLLERS = ["nominal_causal_mpc", "static_robust_mpc", "eact_mpc"]

BENEFIT_CMAP = LinearSegmentedColormap.from_list(
    "benefit", ["#B23A2B", "#F7F7F7", "#2166AC"]
)
PENALTY_CMAP = LinearSegmentedColormap.from_list(
    "penalty", ["#2166AC", "#F7F7F7", "#B23A2B"]
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
        }
    )


def read_data(name: str, expected_min_rows: int = 1) -> pd.DataFrame:
    path = PLOT_DATA / name
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared data: {path}")
    frame = pd.read_csv(path)
    if len(frame) < expected_min_rows:
        raise ValueError(f"{name} contains {len(frame)} rows; expected at least {expected_min_rows}")
    return frame


def clean_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )


def save_figure(fig: plt.Figure, stem: str) -> list[str]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, dpi in [("pdf", None), ("svg", None), ("png", 300), ("tiff", 600)]:
        path = OUTPUT / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.025,
            facecolor="white",
        )
        outputs.append(path.relative_to(PACKAGE).as_posix())
    plt.close(fig)
    return outputs


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    accent: str,
    title_size: float = 7.7,
    detail_size: float = 6.7,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        linewidth=0.75,
        edgecolor="#B8B8B8",
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.add_patch(Rectangle((x, y), 0.008, height, facecolor=accent, edgecolor="none"))
    ax.text(
        x + 0.018,
        y + height * 0.66,
        title,
        ha="left",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        x + 0.018,
        y + height * 0.29,
        detail,
        ha="left",
        va="center",
        fontsize=detail_size,
        color=COLORS["muted"],
        linespacing=1.15,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#8A8A8A") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def figure1() -> list[str]:
    data = read_data("figure1_methodological_framework.csv", 13)
    if set(data.stage) != {"data_plant", "control_loop", "evaluation"}:
        raise ValueError("Figure 1 stage data are incomplete")

    fig = plt.figure(figsize=(FIGURE_WIDTH_MM * MM, 108 * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        (0.02, 0.28, "01", "DATA + PLANT", COLORS["teal"]),
        (0.35, 0.34, "02", "CAUSAL EACT-MPC LOOP", COLORS["eact_mpc"]),
        (0.74, 0.24, "03", "EVALUATION", COLORS["static_robust_mpc"]),
    ]
    for x, width, number, title, color in stages:
        ax.text(x, 0.955, number, fontsize=8, fontweight="bold", color=color, va="center")
        ax.text(x + 0.045, 0.955, title, fontsize=8.5, fontweight="bold", color=COLORS["ink"], va="center")
        ax.plot([x, x + width], [0.925, 0.925], color=color, linewidth=2.2, solid_capstyle="butt")

    left = data.loc[data.stage.eq("data_plant")].sort_values("order")
    left_y = [0.73, 0.50, 0.27]
    for row, y in zip(left.itertuples(index=False), left_y):
        draw_box(ax, 0.025, y, 0.27, 0.15, row.title, row.detail, COLORS["teal"])
    arrow(ax, (0.16, 0.73), (0.16, 0.66))
    arrow(ax, (0.16, 0.50), (0.16, 0.43))

    middle = data.loc[data.stage.eq("control_loop")].sort_values("order")
    mid_y = [0.78, 0.65, 0.52, 0.39, 0.24, 0.09]
    mid_h = [0.10, 0.10, 0.10, 0.10, 0.11, 0.09]
    for row, y, height in zip(middle.itertuples(index=False), mid_y, mid_h):
        draw_box(
            ax,
            0.365,
            y,
            0.31,
            height,
            row.title,
            row.detail,
            COLORS["eact_mpc"] if row.order < 6 else COLORS["gold"],
            title_size=7.4,
            detail_size=6.35,
        )
    for y0, y1 in zip(mid_y[:-1], mid_y[1:]):
        arrow(ax, (0.52, y0), (0.52, y1 + (0.11 if y1 == 0.24 else 0.10)))
    ax.add_patch(
        FancyArrowPatch(
            (0.675, 0.29),
            (0.695, 0.70),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=COLORS["muted"],
            connectionstyle="arc3,rad=-0.28",
        )
    )
    ax.text(0.695, 0.48, "new observation", rotation=90, fontsize=6.2, color=COLORS["muted"], va="center")

    right = data.loc[data.stage.eq("evaluation")].sort_values("order")
    right_y = [0.72, 0.52, 0.32, 0.12]
    for row, y in zip(right.itertuples(index=False), right_y):
        draw_box(ax, 0.75, y, 0.225, 0.14, row.title, row.detail, COLORS["static_robust_mpc"], title_size=7.3, detail_size=6.2)

    arrow(ax, (0.30, 0.35), (0.355, 0.35), COLORS["ink"])
    arrow(ax, (0.68, 0.35), (0.74, 0.35), COLORS["ink"])
    ax.text(0.323, 0.37, "states", fontsize=6.3, color=COLORS["muted"], ha="center")
    ax.text(0.71, 0.37, "records", fontsize=6.3, color=COLORS["muted"], ha="center")
    ax.text(
        0.025,
        0.035,
        "Controller contrasts share forecasts, plant model, objective, constraints, and solver.",
        fontsize=6.8,
        color=COLORS["muted"],
        ha="left",
        va="bottom",
    )
    return save_figure(fig, "figure1_methodological_framework")


def horizontal_forest(
    ax: plt.Axes,
    frame: pd.DataFrame,
    xlabel: str,
    color: str,
    zero: bool = True,
) -> None:
    frame = frame.reset_index(drop=True)
    y = np.arange(len(frame))[::-1]
    values = frame.value.to_numpy(float)
    low = frame.ci95_low.to_numpy(float)
    high = frame.ci95_high.to_numpy(float)
    xerr = np.vstack([values - low, high - values])
    ax.errorbar(
        values,
        y,
        xerr=xerr,
        fmt="o",
        color=color,
        ecolor=color,
        elinewidth=1.2,
        capsize=2.5,
        zorder=3,
    )
    if zero:
        ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(y, frame.condition)
    ax.set_xlabel(xlabel)
    clean_axis(ax, "x")


def figure2() -> list[str]:
    data = read_data("figure2_overall_evidence.csv", 10)
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_MM * MM, 118 * MM))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.11, top=0.96, wspace=0.52, hspace=0.62)

    ax = axes[0, 0]
    frame = data.loc[data.panel.eq("a")]
    colors = [COLORS["static_robust_mpc"], COLORS["eact_mpc"]]
    y = np.arange(len(frame))[::-1]
    ax.barh(y, frame.value, color=colors, height=0.52, zorder=2)
    ax.set_yticks(y, frame.condition)
    ax.set_xlabel("Absolute coverage error (percentage points)")
    ax.set_xlim(0, max(frame.value) * 1.22)
    for yi, value in zip(y, frame.value):
        ax.text(value + 0.08, yi, f"{value:.2f}", va="center", fontsize=7)
    clean_axis(ax, "x")
    panel_label(ax, "a")

    ax = axes[0, 1]
    frame = data.loc[data.panel.eq("b")].reset_index(drop=True)
    x = np.arange(len(frame))
    ax.scatter(x, frame.value, color=COLORS["eact_mpc"], zorder=3)
    for xi, row in enumerate(frame.itertuples(index=False)):
        ax.vlines(xi, row.value, row.ci95_high, color=COLORS["eact_mpc"], linewidth=1.3)
        ax.plot([xi - 0.08, xi + 0.08], [row.ci95_high, row.ci95_high], color=COLORS["eact_mpc"], linewidth=1.3)
        text_x = xi + 0.06 if xi == 0 else xi - 0.02
        text_ha = "left" if xi == 0 else "right"
        ax.text(text_x, row.ci95_high + 0.045, f"upper {row.ci95_high:.2f}%", ha=text_ha, va="bottom", fontsize=6.5)
    ax.axhline(1.0, color=COLORS["ink"], linestyle="--", linewidth=0.9, label="1% margin")
    ax.set_xticks(x, ["Seasonal", "Annual"])
    ax.set_ylabel("Common-objective increase (%)")
    ax.set_ylim(-0.05, 1.16)
    clean_axis(ax, "y")
    panel_label(ax, "b")

    ax = axes[1, 0]
    frame = data.loc[data.panel.eq("c")]
    horizontal_forest(ax, frame, "P95 inlet-temperature reduction (°C)", COLORS["static_robust_mpc"])
    panel_label(ax, "c")

    ax = axes[1, 1]
    frame = data.loc[data.panel.eq("d")]
    horizontal_forest(ax, frame, "Total facility-energy increase (%)", COLORS["eact_mpc"])
    panel_label(ax, "d")
    return save_figure(fig, "figure2_overall_evidence")


def heatmap_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    unit: str,
    cmap: LinearSegmentedColormap,
    show_y: bool,
) -> None:
    countries = sorted(frame.country.unique())
    seasons = ["Q1", "Q2", "Q3", "Q4"]
    matrix = frame.pivot(index="country", columns="season", values="value").reindex(index=countries, columns=seasons)
    values = matrix.to_numpy(float)
    bound = float(np.nanmax(np.abs(values)))
    if bound <= 0:
        bound = 1.0
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(np.arange(4), seasons)
    ax.set_yticks(np.arange(len(countries)))
    ax.set_yticklabels(countries if show_y else [])
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    fmt = ".2f" if unit == "°C" else ".1f"
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=5.4,
                color="white" if luminance < 0.52 else COLORS["ink"],
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.065, fraction=0.045, aspect=24)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=6.1, length=2)
    cbar.set_label(unit, fontsize=6.5, labelpad=1)


def figure3() -> list[str]:
    data = read_data("figure3_country_season.csv", 240)
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_MM * MM, 157 * MM))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.07, top=0.965, wspace=0.18, hspace=0.28)
    specs = [
        ("a", "No shift: objective increase", "%", PENALTY_CMAP),
        ("b", "No shift: P95 reduction", "°C", BENEFIT_CMAP),
        ("c", "Persistent bias: P95 reduction", "°C", BENEFIT_CMAP),
        ("d", "Persistent bias: energy increase", "%", PENALTY_CMAP),
    ]
    for index, (panel, title, unit, cmap) in enumerate(specs):
        ax = axes.flat[index]
        heatmap_panel(fig, ax, data.loc[data.panel.eq(panel)], title, unit, cmap, show_y=index % 2 == 0)
        panel_label(ax, panel, x=-0.11 if index % 2 == 0 else -0.04, y=1.035)
    return save_figure(fig, "figure3_country_season")


def figure6() -> list[str]:
    data = read_data("figure6_mechanism_robustness.csv", 20)
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_MM * MM, 125 * MM))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.12, top=0.96, wspace=0.53, hspace=0.62)

    ax = axes[0, 0]
    beta_p95 = data.loc[
        data.panel.eq("a") & data.metric.eq("p95_reduction_c")
    ].iloc[0]
    beta_cost = data.loc[
        data.panel.eq("a") & data.metric.eq("objective_increase_pct")
    ].iloc[0]
    ax.scatter([0], [0], color="#A0A0A0", s=22, zorder=3)
    ax.annotate(
        r"$\beta_{\min}=0$",
        xy=(0, 0),
        xytext=(0.12, 0.015),
        textcoords="data",
        fontsize=6.7,
        color=COLORS["muted"],
        arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.7),
    )
    ax.errorbar(
        beta_cost.value,
        beta_p95.value,
        xerr=[
            [beta_cost.value - beta_cost.ci95_low],
            [beta_cost.ci95_high - beta_cost.value],
        ],
        yerr=[
            [beta_p95.value - beta_p95.ci95_low],
            [beta_p95.ci95_high - beta_p95.value],
        ],
        fmt="o",
        color=COLORS["eact_mpc"],
        capsize=2.5,
        zorder=3,
    )
    ax.annotate(
        r"$\beta_{\min}=0.10$",
        xy=(beta_cost.value, beta_p95.value),
        xytext=(0.45, 0.17),
        fontsize=6.7,
        color=COLORS["ink"],
        arrowprops=dict(arrowstyle="-", color=COLORS["eact_mpc"], linewidth=0.8),
    )
    ax.axhline(0, color="#888888", linewidth=0.7)
    ax.axvline(0, color="#888888", linewidth=0.7)
    ax.set_xlabel("Common-objective increase (%)")
    ax.set_ylabel("P95 reduction (°C)")
    ax.set_xlim(-0.25, 2.05)
    ax.set_ylim(-0.025, 0.205)
    clean_axis(ax, "both")
    panel_label(ax, "a")

    ax = axes[0, 1]
    frame = data.loc[data.panel.eq("b")]
    horizontal_forest(ax, frame, "P95 reduction (°C)", COLORS["static_robust_mpc"])
    panel_label(ax, "b")

    ax = axes[1, 0]
    frame = data.loc[data.panel.eq("c")]
    horizontal_forest(ax, frame, "Common-objective increase (%)", COLORS["eact_mpc"])
    panel_label(ax, "c")

    ax = axes[1, 1]
    frame = data.loc[data.panel.eq("d")].copy()
    settings = ["low_carbon", "low_total", "primary", "high_total", "high_carbon"]
    labels = ["Low\ncarbon", "Low\ntotal", "Primary", "High\ntotal", "High\ncarbon"]
    x = np.arange(len(settings))
    for condition, color, marker, offset in [
        ("No shift", COLORS["static_robust_mpc"], "o", -0.05),
        ("Persistent bias", COLORS["eact_mpc"], "s", 0.05),
    ]:
        subset = frame.loc[frame.condition.eq(condition)].set_index("setting").reindex(settings)
        values = subset.value.to_numpy(float)
        low = subset.ci95_low.to_numpy(float)
        high = subset.ci95_high.to_numpy(float)
        ax.errorbar(
            x + offset,
            values,
            yerr=np.vstack([values - low, high - values]),
            fmt=marker + "-",
            color=color,
            capsize=2.0,
            label=condition,
        )
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, labels)
    ax.set_ylabel("P95 reduction (°C)")
    ax.legend(loc="lower right")
    clean_axis(ax, "y")
    panel_label(ax, "d")
    return save_figure(fig, "figure6_mechanism_robustness")


def figure4() -> list[str]:
    data = read_data("figure4_capacity_stress.csv", 30)
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_MM * MM, 124 * MM))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.96, wspace=0.44, hspace=0.58)
    availability = [100, 75, 50]
    x = np.arange(3)

    ax = axes[0, 0]
    frame = data.loc[data.panel.eq("a") & data.condition.eq("Persistent bias")]
    for controller in CONTROLLERS:
        subset = frame.loc[frame.controller.eq(controller)].set_index("availability_pct").reindex(availability)
        ax.plot(x, subset.value, marker="o", color=COLORS[controller], label=LABELS[controller])
    ax.set_xticks(x, ["100", "75", "50"])
    ax.set_xlabel("Available conductance (%)")
    ax.set_ylabel("Event hours across 12 weeks")
    ax.set_yscale("symlog", linthresh=1, linscale=0.8)
    ax.set_yticks([0, 1, 5, 20, 50, 200])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.legend(loc="upper left", ncol=1)
    clean_axis(ax, "y")
    panel_label(ax, "a")

    ax = axes[0, 1]
    frame = data.loc[data.panel.eq("b")].set_index("controller").reindex(CONTROLLERS)
    bars = ax.bar(
        np.arange(3),
        frame.value,
        color=[COLORS[c] for c in CONTROLLERS],
        width=0.62,
    )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(3), ["Nominal", "Static", "EACT"])
    ax.set_ylabel("Cumulative excess (°C h)")
    for bar, value in zip(bars, frame.value):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.18, f"{value:.2f}", ha="center", va="bottom", fontsize=6.7)
    ax.text(1.48, 42, "Static to EACT: -91.7%", ha="center", va="center", fontsize=6.6, color=COLORS["ink"])
    clean_axis(ax, "y")
    panel_label(ax, "b")

    for ax, panel, ylabel, zero in [
        (axes[1, 0], "c", "P95 reduction (°C)", True),
        (axes[1, 1], "d", "Total energy increase (%)", True),
    ]:
        frame = data.loc[data.panel.eq(panel)]
        for condition, color, marker, offset in [
            ("No shift", COLORS["static_robust_mpc"], "o", -0.04),
            ("Persistent bias", COLORS["eact_mpc"], "s", 0.04),
        ]:
            subset = frame.loc[frame.condition.eq(condition)].set_index("availability_pct").reindex(availability)
            values = subset.value.to_numpy(float)
            low = subset.ci95_low.to_numpy(float)
            high = subset.ci95_high.to_numpy(float)
            ax.errorbar(
                x + offset,
                values,
                yerr=np.vstack([values - low, high - values]),
                fmt=marker + "-",
                color=color,
                capsize=2.2,
                label=condition,
            )
        if zero:
            ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xticks(x, ["100", "75", "50"])
        ax.set_xlabel("Available conductance (%)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        clean_axis(ax, "y")
        panel_label(ax, panel)
    return save_figure(fig, "figure4_capacity_stress")


def figure5() -> list[str]:
    data = read_data("figure5_representative_trajectory.csv", 504)
    if len(data) != 504:
        raise ValueError(f"Figure 5 requires exactly 504 hourly rows, found {len(data)}")
    fig, axes = plt.subplots(3, 1, figsize=(FIGURE_WIDTH_MM * MM, 132 * MM), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.10, top=0.96, hspace=0.22)
    for controller in CONTROLLERS:
        frame = data.loc[data.algorithm.eq(controller)].sort_values("hour")
        day = frame.hour.to_numpy(float) / 24.0
        axes[0].plot(day, frame.t_inlet_c, color=COLORS[controller], label=LABELS[controller])
        axes[1].plot(day, frame.e_cooling_mwh, color=COLORS[controller])
        axes[2].plot(day, frame.cumulative_excess_degc_h, color=COLORS[controller])

    axes[0].axhline(27.0, color=COLORS["ink"], linestyle="--", linewidth=0.9)
    axes[0].text(6.92, 27.10, "27 °C recommended limit", ha="right", va="bottom", fontsize=6.8, color=COLORS["ink"])
    axes[0].set_ylabel("Inlet temperature\n(°C)")
    axes[0].set_ylim(24.0, 29.35)
    axes[0].legend(loc="upper left", ncol=3, columnspacing=1.0, handlelength=2.2)
    axes[1].set_ylabel("Cooling energy\n(MWh h$^{-1}$)")
    axes[2].set_ylabel("Cumulative excess\n(°C h)")
    axes[2].set_xlabel("Elapsed day")
    axes[2].set_xlim(0, 7)
    axes[2].set_xticks(np.arange(0, 8, 1))
    for label, ax in zip(["a", "b", "c"], axes):
        clean_axis(ax, "y")
        panel_label(ax, label, x=-0.075, y=1.03)
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=5))
    return save_figure(fig, "figure5_representative_trajectory")


def write_build_manifest(outputs: dict[str, list[str]]) -> None:
    data_files = []
    for path in sorted(PLOT_DATA.glob("*.csv")):
        data_files.append(
            {
                "file": path.relative_to(PACKAGE).as_posix(),
                "rows": len(pd.read_csv(path)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "backend": "Python matplotlib",
        "figure_width_mm": FIGURE_WIDTH_MM,
        "style": "Nature-inspired white-background multi-panel scientific figures",
        "data_files": data_files,
        "outputs": outputs,
    }
    (PACKAGE / "figure_build_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    configure_style()
    outputs = {
        "figure1_methodological_framework": figure1(),
        "figure2_overall_evidence": figure2(),
        "figure3_country_season": figure3(),
        "figure4_capacity_stress": figure4(),
        "figure5_representative_trajectory": figure5(),
        "figure6_mechanism_robustness": figure6(),
    }
    write_build_manifest(outputs)
    print(f"Rendered {len(outputs)} figures in {OUTPUT}")


if __name__ == "__main__":
    main()
