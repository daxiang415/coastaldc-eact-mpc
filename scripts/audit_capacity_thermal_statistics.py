"""Audit sparse thermal outcomes in the compound capacity-stress experiment."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    ROOT / "results" / "ashrae_inlet_thermal_capacity_stress_v1"
)
DEFAULT_OUT = ROOT / "results" / "statistical_audit"
DEFAULT_FIGURE_DATA = ROOT / "results" / "figure_data" / "table_data"
DEFAULT_TABLES = ROOT / "results" / "generated_tables"
PAIR_EFFECT_THRESHOLD_DEGC_H = 1e-4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--figure-data-dir", type=Path, default=DEFAULT_FIGURE_DATA
    )
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLES)
    return parser


def paired_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Wilcoxon differences must be finite and nonempty")
    nonzero = values[values != 0.0]
    if nonzero.size == 0:
        return 1.0
    return float(
        wilcoxon(
            nonzero,
            alternative="two-sided",
            zero_method="wilcox",
            method="exact",
        ).pvalue
    )


def load_pairs(results_root: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(results_root / "**" / "episodes_*.csv"), recursive=True))
    if len(paths) != 18:
        raise ValueError(f"Expected 18 episode files, found {len(paths)}")
    episodes = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    selected = episodes.loc[
        (episodes["forecast_stress"] == "adverse_bias")
        & np.isclose(episodes["forecast_stress_scale"], 1.0)
        & np.isclose(episodes["cooling_conductance_multiplier"], 0.5)
        & episodes["algorithm"].isin(["static_robust_mpc", "eact_mpc"])
    ].copy()
    keys = ["country", "start_hour"]
    metrics = [
        "recommended_exceedance_hours",
        "recommended_exceedance_degc_h",
        "p95_t_inlet_c",
        "e_total_mwh",
    ]
    wide = selected[keys + ["algorithm", *metrics]].pivot(
        index=keys, columns="algorithm"
    )
    if len(wide) != 12:
        raise ValueError(f"Expected 12 compound-stress pairs, found {len(wide)}")

    selection_path = results_root / "high_load_window_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    window_rows = []
    for country, windows in selection["windows"].items():
        for item in windows:
            window_rows.append(
                {
                    "country": country,
                    "start_hour": int(item["start_hour"]),
                    "quarter": item["quarter"],
                    "start_timestamp": item["start_timestamp"],
                }
            )
    windows = pd.DataFrame(window_rows)

    pairs = pd.DataFrame(index=wide.index).reset_index()
    pairs = pairs.merge(windows, on=keys, how="left", validate="one_to_one")
    pairs["pair_id"] = pairs["country"] + "_" + pairs["quarter"]
    for metric in metrics:
        pairs[f"static_{metric}"] = wide[(metric, "static_robust_mpc")].to_numpy(float)
        pairs[f"eact_{metric}"] = wide[(metric, "eact_mpc")].to_numpy(float)

    pairs["event_hour_reduction"] = (
        pairs["static_recommended_exceedance_hours"]
        - pairs["eact_recommended_exceedance_hours"]
    )
    pairs["degree_hour_reduction"] = (
        pairs["static_recommended_exceedance_degc_h"]
        - pairs["eact_recommended_exceedance_degc_h"]
    )
    pairs["degree_hour_reduction_solver_scale"] = np.where(
        np.abs(pairs["degree_hour_reduction"]) <= PAIR_EFFECT_THRESHOLD_DEGC_H,
        0.0,
        pairs["degree_hour_reduction"],
    )
    pairs["p95_reduction_c"] = (
        pairs["static_p95_t_inlet_c"] - pairs["eact_p95_t_inlet_c"]
    )
    pairs["facility_energy_increase_mwh"] = (
        pairs["eact_e_total_mwh"] - pairs["static_e_total_mwh"]
    )

    static_total = pairs["static_recommended_exceedance_degc_h"].sum()
    reduction_total = pairs["degree_hour_reduction"].sum()
    pairs["static_degree_hour_share_pct"] = (
        100.0 * pairs["static_recommended_exceedance_degc_h"] / static_total
    )
    pairs["degree_hour_reduction_share_pct"] = (
        100.0 * pairs["degree_hour_reduction"] / reduction_total
    )
    return pairs.sort_values(["country", "quarter"]).reset_index(drop=True)


def aggregate_reduction_pct(static_total: float, eact_total: float) -> float:
    if static_total <= 0:
        return float("nan")
    return float(100.0 * (static_total - eact_total) / static_total)


def statistical_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    event_effect = pairs["event_hour_reduction"].to_numpy(float)
    degree_effect = pairs["degree_hour_reduction"].to_numpy(float)
    degree_sensitivity = pairs[
        "degree_hour_reduction_solver_scale"
    ].to_numpy(float)
    definitions = [
        (
            "Tolerance-adjusted event hours",
            event_effect,
            "Event counts already exclude hourly excess <= 1e-4 degC",
        ),
        (
            "Raw cumulative degree-hours",
            degree_effect,
            "Protocol-defined raw positive excess; no pair-effect threshold",
        ),
        (
            "Degree-hours: solver-scale sensitivity",
            degree_sensitivity,
            "Exploratory sensitivity: pair effects <= 1e-4 degC h set to zero",
        ),
    ]
    rows = []
    for name, values, rule in definitions:
        rows.append(
            {
                "analysis": name,
                "threshold_rule": rule,
                "n_pairs": int(values.size),
                "n_eff_nonzero_pairs": int(np.count_nonzero(values)),
                "wilcoxon_p_value": paired_wilcoxon(values),
                "mean_pair_reduction": float(np.mean(values)),
                "median_pair_reduction": float(np.median(values)),
                "minimum_pair_reduction": float(np.min(values)),
                "maximum_pair_reduction": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def leave_one_out(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, omitted in pairs.iterrows():
        kept = pairs.drop(index=index)
        static_degree = kept["static_recommended_exceedance_degc_h"].sum()
        eact_degree = kept["eact_recommended_exceedance_degc_h"].sum()
        static_events = kept["static_recommended_exceedance_hours"].sum()
        eact_events = kept["eact_recommended_exceedance_hours"].sum()
        degree_raw = kept["degree_hour_reduction"].to_numpy(float)
        degree_sensitivity = kept[
            "degree_hour_reduction_solver_scale"
        ].to_numpy(float)
        event_effect = kept["event_hour_reduction"].to_numpy(float)
        rows.append(
            {
                "omitted_pair": omitted["pair_id"],
                "static_degree_hours": float(static_degree),
                "eact_degree_hours": float(eact_degree),
                "degree_hour_reduction_pct": aggregate_reduction_pct(
                    static_degree, eact_degree
                ),
                "static_event_hours": float(static_events),
                "eact_event_hours": float(eact_events),
                "event_hour_reduction_pct": aggregate_reduction_pct(
                    static_events, eact_events
                ),
                "degree_raw_n_eff": int(np.count_nonzero(degree_raw)),
                "degree_raw_wilcoxon_p": paired_wilcoxon(degree_raw),
                "degree_solver_scale_n_eff": int(
                    np.count_nonzero(degree_sensitivity)
                ),
                "degree_solver_scale_wilcoxon_p": paired_wilcoxon(
                    degree_sensitivity
                ),
                "event_hour_n_eff": int(np.count_nonzero(event_effect)),
                "event_hour_wilcoxon_p": paired_wilcoxon(event_effect),
            }
        )
    return pd.DataFrame(rows)


def tex_number(value: float) -> str:
    if value != 0.0 and abs(value) < 1e-3:
        return rf"\num{{{value:.3e}}}"
    return f"{value:.3f}"


def write_supplementary_table(pairs: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begingroup",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \renewcommand{\arraystretch}{1.08}",
        r"  \begin{longtable}{@{}lrrrrrr@{}}",
        r"    \caption{\textbf{Pair-level thermal exposure under compound forecast-bias and cooling-capacity stress.} All rows are predefined high-load country-quarter weeks at 50\% heat-transfer availability under persistent $1\sigma$ adverse bias. Event hours exclude hourly excess at or below the recorded $10^{-4}\,\si{\degreeCelsius}$ feasibility tolerance; degree-hours retain raw positive excess. The final column gives each pair's share of the Static Robust MPC degree-hour total.}",
        r"    \label{tab:appendix-capacity-pairs} \\",
        r"    \toprule",
        r"    Pair & \multicolumn{2}{c}{Event hours (h)} & \multicolumn{3}{c}{Degree-hours (\si{\degreeCelsius\hour})} & Static share (\%) \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        r"     & Static & EACT & Static & EACT & Reduction & \\",
        r"    \midrule",
        r"    \endfirsthead",
        r"    \multicolumn{7}{@{}l}{\textit{Table \thetable\ continued}} \\",
        r"    \toprule",
        r"    Pair & \multicolumn{2}{c}{Event hours (h)} & \multicolumn{3}{c}{Degree-hours (\si{\degreeCelsius\hour})} & Static share (\%) \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        r"     & Static & EACT & Static & EACT & Reduction & \\",
        r"    \midrule",
        r"    \endhead",
        r"    \bottomrule",
        r"    \endfoot",
    ]
    for row in pairs.itertuples(index=False):
        lines.append(
            f"    {row.pair_id.replace('_', ' ')} & "
            f"{row.static_recommended_exceedance_hours:.0f} & "
            f"{row.eact_recommended_exceedance_hours:.0f} & "
            f"{tex_number(row.static_recommended_exceedance_degc_h)} & "
            f"{tex_number(row.eact_recommended_exceedance_degc_h)} & "
            f"{tex_number(row.degree_hour_reduction)} & "
            f"{row.static_degree_hour_share_pct:.1f} \\\\"
        )
    lines.extend([r"  \end{longtable}", r"\endgroup", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_data_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args.results_root)
    summary = statistical_summary(pairs)
    loo = leave_one_out(pairs)

    pairs_path = args.out_dir / "capacity_compound_stress_pairs.csv"
    summary_path = args.out_dir / "capacity_compound_stress_statistical_audit.csv"
    loo_path = args.out_dir / "capacity_compound_stress_leave_one_out.csv"
    pairs.to_csv(pairs_path, index=False)
    summary.to_csv(summary_path, index=False)
    loo.to_csv(loo_path, index=False)
    pairs.to_csv(
        args.figure_data_dir / "table_s3_capacity_pair_distribution.csv",
        index=False,
    )
    summary.to_csv(
        args.figure_data_dir / "table5_capacity_statistical_audit.csv",
        index=False,
    )
    loo.to_csv(
        args.figure_data_dir / "table_s3_capacity_leave_one_out.csv",
        index=False,
    )
    write_supplementary_table(
        pairs, args.table_dir / "table_s3_capacity_pair_distribution.tex"
    )

    largest = pairs.loc[
        pairs["static_recommended_exceedance_degc_h"].idxmax()
    ]
    audit = {
        "source_root": str(args.results_root.resolve()),
        "condition": "adverse_bias_s1.0, cooling availability 0.5",
        "n_pairs": int(len(pairs)),
        "pair_effect_threshold_degc_h": PAIR_EFFECT_THRESHOLD_DEGC_H,
        "static_degree_hours_total": float(
            pairs["static_recommended_exceedance_degc_h"].sum()
        ),
        "eact_degree_hours_total": float(
            pairs["eact_recommended_exceedance_degc_h"].sum()
        ),
        "degree_hour_reduction_pct": aggregate_reduction_pct(
            pairs["static_recommended_exceedance_degc_h"].sum(),
            pairs["eact_recommended_exceedance_degc_h"].sum(),
        ),
        "largest_static_pair": largest["pair_id"],
        "largest_static_pair_share_pct": float(
            largest["static_degree_hour_share_pct"]
        ),
        "largest_pair_reduction_share_pct": float(
            largest["degree_hour_reduction_share_pct"]
        ),
        "leave_one_out_degree_hour_reduction_pct_min": float(
            loo["degree_hour_reduction_pct"].min()
        ),
        "leave_one_out_degree_hour_reduction_pct_max": float(
            loo["degree_hour_reduction_pct"].max()
        ),
        "outputs": [str(pairs_path), str(summary_path), str(loo_path)],
    }
    (args.out_dir / "capacity_compound_stress_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
