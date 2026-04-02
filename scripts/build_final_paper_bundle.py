#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT_DIR / "paper_summaries" / "tables"
FINAL_DIR = ROOT_DIR / "paper_summaries" / "final_bundle"
FINAL_TABLE_DIR = FINAL_DIR / "tables"
FINAL_FIGURE_DIR = FINAL_DIR / "figures"
PAPER_FIGURE_DIR = ROOT_DIR / "paper" / "figures"
MANIFEST_PATH = FINAL_DIR / "paper_artifact_manifest.csv"
README_PATH = FINAL_DIR / "README.md"

VARIANTS = ["Base", "VI", "VII", "VIII", "VIV", "VV"]
MODEL_ORDER = ["ft_dense", "lightgbm", "snn_head"]
MODEL_LABELS = {
    "ft_dense": "FT-dense",
    "lightgbm": "LightGBM",
    "snn_head": "SNN-head",
}
MODEL_COLORS = {
    "ft_dense": "#1b4965",
    "lightgbm": "#c05621",
    "snn_head": "#2f855a",
}
ABLATION_MODEL_ORDER = ["ft_dense", "matched_ann", "snn_head"]
ABLATION_MODEL_LABELS = {
    "ft_dense": "FT-dense",
    "matched_ann": "Matched ANN",
    "snn_head": "SNN-head",
}
ABLATION_MODEL_COLORS = {
    "ft_dense": MODEL_COLORS["ft_dense"],
    "matched_ann": "#d69e2e",
    "snn_head": MODEL_COLORS["snn_head"],
}
METRIC_LABELS = {
    "roc_auc_mean": "ROC-AUC",
    "average_precision_mean": "Average Precision",
    "balanced_accuracy_at_selected_threshold_mean": "Balanced Accuracy",
    "tpr_at_5pct_fpr_splitwise_mean": "TPR@5%FPR",
    "predictive_equality_age_ratio_mean": "FPR Parity Ratio",
}
PREREQ_SCRIPTS = [
    "scripts/build_baf_headline_test_tables.py",
    "scripts/build_snn_uncertainty_tables.py",
    "scripts/build_cross_variant_fairness_summary.py",
    "scripts/build_baf_model_comparison_tables.py",
    "scripts/build_cross_variant_fairness_summary_with_lightgbm.py",
    "scripts/build_significance_test_tables.py",
]
PREREQ_SCRIPT_ENV_OVERRIDES = {
    "scripts/build_significance_test_tables.py": {
        "SIGNIFICANCE_BOOTSTRAP_REPLICATES": "250",
        "SIGNIFICANCE_MAX_WORKERS": "6",
    }
}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return pd.read_csv(path)


def format_metric(mean_value: float, std_value: float, n_runs: int, digits: int = 4) -> str:
    if n_runs > 1 and pd.notna(std_value):
        return f"{mean_value:.{digits}f} +- {std_value:.{digits}f}"
    return f"{mean_value:.{digits}f}"


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return float("nan")
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def ensure_clean_output_dirs() -> None:
    if FINAL_DIR.exists():
        shutil.rmtree(FINAL_DIR)
    FINAL_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def run_prereq_builders() -> None:
    if os.environ.get("FINAL_BUNDLE_SKIP_PREREQS") == "1":
        print("[final-bundle] skipping prereq builders via FINAL_BUNDLE_SKIP_PREREQS=1", flush=True)
        return
    for script_rel in PREREQ_SCRIPTS:
        script_path = ROOT_DIR / script_rel
        print(f"[final-bundle] running {script_rel}...", flush=True)
        script_env = dict(os.environ)
        script_env.update(PREREQ_SCRIPT_ENV_OVERRIDES.get(script_rel, {}))
        subprocess.run([sys.executable, str(script_path)], check=True, cwd=ROOT_DIR, env=script_env)


def build_main_comparison_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in comparison_df.sort_values(["variant", "model_family"]).iterrows():
        rows.append(
            {
                "variant": row["variant"],
                "model_family": row["model_family"],
                "model_label": row["model_label"],
                "n_runs": int(row["n_runs"]),
                "seed_list": "" if pd.isna(row["seed_list"]) else str(row["seed_list"]),
                "roc_auc_display": format_metric(row["roc_auc_mean"], row["roc_auc_std"], int(row["n_runs"])),
                "average_precision_display": format_metric(
                    row["average_precision_mean"], row["average_precision_std"], int(row["n_runs"])
                ),
                "balanced_accuracy_display": format_metric(
                    row["balanced_accuracy_at_selected_threshold_mean"],
                    row["balanced_accuracy_at_selected_threshold_std"],
                    int(row["n_runs"]),
                ),
                "tpr_at_5pct_fpr_display": format_metric(
                    row["tpr_at_5pct_fpr_splitwise_mean"],
                    row["tpr_at_5pct_fpr_splitwise_std"],
                    int(row["n_runs"]),
                ),
                "predictive_equality_display": format_metric(
                    row["predictive_equality_age_ratio_mean"],
                    row["predictive_equality_age_ratio_std"],
                    int(row["n_runs"]),
                ),
                "source_run_dirs": row["source_run_dirs"],
            }
        )
    return pd.DataFrame(rows)


def build_fairness_table(
    comparison_df: pd.DataFrame,
    fairness_df: pd.DataFrame,
) -> pd.DataFrame:
    merged_df = comparison_df.merge(
        fairness_df[
            [
                "variant",
                "model_family",
                "n_runs",
                "test_age_group_predictive_equality_ratio_min_over_max_mean",
                "test_age_group_predictive_equality_ratio_min_over_max_std",
                "test_income_group_aequitas_fpr_parity_mean",
                "test_income_group_aequitas_fpr_parity_std",
                "test_employment_status_group_aequitas_fpr_parity_mean",
                "test_employment_status_group_aequitas_fpr_parity_std",
            ]
        ],
        on=["variant", "model_family", "n_runs"],
        how="inner",
    )

    rows: list[dict[str, object]] = []
    for _, row in merged_df.sort_values(["variant", "model_family"]).iterrows():
        rows.append(
            {
                "variant": row["variant"],
                "model_family": row["model_family"],
                "model_label": row["model_label"],
                "average_precision_display": format_metric(
                    row["average_precision_mean"], row["average_precision_std"], int(row["n_runs"])
                ),
                "tpr_at_5pct_fpr_display": format_metric(
                    row["tpr_at_5pct_fpr_splitwise_mean"],
                    row["tpr_at_5pct_fpr_splitwise_std"],
                    int(row["n_runs"]),
                ),
                "age_group_predictive_equality_display": format_metric(
                    row["test_age_group_predictive_equality_ratio_min_over_max_mean"],
                    row["test_age_group_predictive_equality_ratio_min_over_max_std"],
                    int(row["n_runs"]),
                ),
                "income_group_parity_display": format_metric(
                    row["test_income_group_aequitas_fpr_parity_mean"],
                    row["test_income_group_aequitas_fpr_parity_std"],
                    int(row["n_runs"]),
                ),
                "employment_status_parity_display": format_metric(
                    row["test_employment_status_group_aequitas_fpr_parity_mean"],
                    row["test_employment_status_group_aequitas_fpr_parity_std"],
                    int(row["n_runs"]),
                ),
            }
        )
    return pd.DataFrame(rows)


def build_significance_table(significance_key_df: pd.DataFrame) -> pd.DataFrame:
    significant_df = significance_key_df.loc[
        significance_key_df["bh_fdr_within_metric_comparison_0_05_significant"]
    ].copy()
    significant_df = significant_df.sort_values(["comparison_key", "metric", "variant"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for _, row in significant_df.iterrows():
        if row["ci_supported_direction"] == row["model_a_family"]:
            supported_model = row["model_a_label"]
        elif row["ci_supported_direction"] == row["model_b_family"]:
            supported_model = row["model_b_label"]
        else:
            supported_model = "inconclusive"

        rows.append(
            {
                "variant": row["variant"],
                "comparison_key": row["comparison_key"],
                "metric": row["metric"],
                "supported_model": supported_model,
                "observed_diff_model_a_minus_model_b": f"{row['observed_diff_model_a_minus_model_b']:.4f}",
                "bootstrap_ci_95": f"[{row['bootstrap_ci_95_lower']:.4f}, {row['bootstrap_ci_95_upper']:.4f}]",
                "p_value_bh_fdr_within_metric_comparison": f"{row['p_value_bh_fdr_within_metric_comparison']:.4f}",
                "bootstrap_replicates": int(row["bootstrap_replicates"]),
            }
        )
    return pd.DataFrame(rows)


def build_vv_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    vv_specs = [
        ("FT-Transformer dense head", ROOT_DIR / "ft_dense_baseline_100_run_VV"),
        ("FT-Transformer SNN head (seed42)", ROOT_DIR / "snn_ftt_100_run_VV_seed42"),
        ("FT-Transformer SNN head (seed52)", ROOT_DIR / "snn_ftt_100_run_VV_seed52"),
        ("LightGBM baseline", ROOT_DIR / "lightgbm_baseline_100_run_VV"),
    ]

    for model_label, run_dir in vv_specs:
        final_metrics_df = load_csv(run_dir / "paper_artifacts" / "tables" / "paper_table_final_metrics.csv")
        valid_row = final_metrics_df.loc[final_metrics_df["split"] == "VALID"].iloc[0]
        test_row = final_metrics_df.loc[final_metrics_df["split"] == "TEST"].iloc[0]
        rows.append(
            {
                "model_label": model_label,
                "source_run_dir": str(run_dir.relative_to(ROOT_DIR)),
                "valid_auc": f"{valid_row['auc']:.4f}",
                "test_auc": f"{test_row['auc']:.4f}",
                "delta_auc": f"{(test_row['auc'] - valid_row['auc']):.4f}",
                "valid_average_precision": f"{valid_row['pr_average_precision_score']:.4f}",
                "test_average_precision": f"{test_row['pr_average_precision_score']:.4f}",
                "delta_average_precision": f"{(test_row['pr_average_precision_score'] - valid_row['pr_average_precision_score']):.4f}",
                "valid_recall_at_5pct_fpr_proxy": f"{valid_row['recall_at_selected_threshold']:.4f}",
                "test_recall_at_5pct_fpr_proxy": f"{test_row['recall_at_selected_threshold']:.4f}",
                "delta_recall_at_5pct_fpr_proxy": f"{(test_row['recall_at_selected_threshold'] - valid_row['recall_at_selected_threshold']):.4f}",
                "test_fpr": f"{test_row['fpr_at_selected_threshold']:.4f}",
            }
        )
    return pd.DataFrame(rows)


def build_matched_ann_ablation_raw_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    run_specs = {
        "ft_dense": lambda variant: [ROOT_DIR / f"ft_dense_baseline_100_run_{variant}"],
        "matched_ann": lambda variant: [
            ROOT_DIR / f"matched_ann_control_100_run_{variant}_seed42",
            ROOT_DIR / f"matched_ann_control_100_run_{variant}_seed52",
        ],
        "snn_head": lambda variant: [
            ROOT_DIR / f"snn_ftt_100_run_{variant}_seed42",
            ROOT_DIR / f"snn_ftt_100_run_{variant}_seed52",
        ],
    }
    seed_specs = {
        "ft_dense": ["42"],
        "matched_ann": ["42", "52"],
        "snn_head": ["42", "52"],
    }

    for variant in VARIANTS:
        for model_family in ABLATION_MODEL_ORDER:
            run_dirs = run_specs[model_family](variant)
            test_rows = []
            for run_dir in run_dirs:
                final_metrics_df = load_csv(run_dir / "paper_artifacts" / "tables" / "paper_table_final_metrics.csv")
                test_rows.append(final_metrics_df.loc[final_metrics_df["split"] == "TEST"].iloc[0])

            auc_values = [float(row["auc"]) for row in test_rows]
            ap_values = [float(row["pr_average_precision_score"]) for row in test_rows]
            recall_values = [float(row["recall_at_selected_threshold"]) for row in test_rows]
            fpr_values = [float(row["fpr_at_selected_threshold"]) for row in test_rows]

            rows.append(
                {
                    "variant": variant,
                    "model_family": model_family,
                    "model_label": ABLATION_MODEL_LABELS[model_family],
                    "n_runs": len(run_dirs),
                    "seed_list": ",".join(seed_specs[model_family][: len(run_dirs)]),
                    "auc_mean": float(np.mean(auc_values)),
                    "auc_std": sample_std(auc_values),
                    "average_precision_mean": float(np.mean(ap_values)),
                    "average_precision_std": sample_std(ap_values),
                    "recall_at_selected_threshold_mean": float(np.mean(recall_values)),
                    "recall_at_selected_threshold_std": sample_std(recall_values),
                    "fpr_at_selected_threshold_mean": float(np.mean(fpr_values)),
                    "fpr_at_selected_threshold_std": sample_std(fpr_values),
                    "source_run_dirs": ";".join(str(run_dir.relative_to(ROOT_DIR)) for run_dir in run_dirs),
                }
            )

    return pd.DataFrame(rows)


def build_matched_ann_ablation_table(ablation_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in ablation_df.sort_values(["variant", "model_family"]).iterrows():
        rows.append(
            {
                "variant": row["variant"],
                "model_family": row["model_family"],
                "model_label": row["model_label"],
                "n_runs": int(row["n_runs"]),
                "seed_list": str(row["seed_list"]),
                "auc_display": format_metric(row["auc_mean"], row["auc_std"], int(row["n_runs"])),
                "average_precision_display": format_metric(
                    row["average_precision_mean"],
                    row["average_precision_std"],
                    int(row["n_runs"]),
                ),
                "recall_at_selected_threshold_display": format_metric(
                    row["recall_at_selected_threshold_mean"],
                    row["recall_at_selected_threshold_std"],
                    int(row["n_runs"]),
                ),
                "fpr_at_selected_threshold_display": format_metric(
                    row["fpr_at_selected_threshold_mean"],
                    row["fpr_at_selected_threshold_std"],
                    int(row["n_runs"]),
                ),
                "source_run_dirs": row["source_run_dirs"],
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: str, extra_dirs: list[Path] | None = None) -> list[Path]:
    outputs = []
    output_dirs = [FINAL_FIGURE_DIR]
    if extra_dirs:
        output_dirs.extend(extra_dirs)
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        for suffix in [".png", ".pdf"]:
            path = output_dir / f"{stem}{suffix}"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            if output_dir == FINAL_FIGURE_DIR:
                outputs.append(path)
    plt.close(fig)
    return outputs


def plot_headline_metrics_grid(comparison_df: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.9), constrained_layout=False)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.84,
        wspace=0.20,
        hspace=0.34,
    )
    axes_flat = axes.flatten()
    x = np.arange(len(VARIANTS))
    metric_columns = [
        "average_precision_mean",
        "balanced_accuracy_at_selected_threshold_mean",
        "tpr_at_5pct_fpr_splitwise_mean",
        "predictive_equality_age_ratio_mean",
    ]

    for ax, metric_column in zip(axes_flat, metric_columns):
        for model_family in MODEL_ORDER:
            model_df = (
                comparison_df.loc[comparison_df["model_family"] == model_family]
                .set_index("variant")
                .loc[VARIANTS]
                .reset_index()
            )
            y = model_df[metric_column].to_numpy(dtype=float)
            std_col = metric_column.replace("_mean", "_std")
            yerr = model_df[std_col].fillna(0.0).to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                marker="o",
                markersize=5.5,
                linewidth=2.2,
                color=MODEL_COLORS[model_family],
                label=MODEL_LABELS[model_family],
            )
            if np.any(yerr > 0.0):
                ax.fill_between(
                    x,
                    y - yerr,
                    y + yerr,
                    color=MODEL_COLORS[model_family],
                    alpha=0.12,
                    linewidth=0.0,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(VARIANTS)
        ax.set_title(METRIC_LABELS[metric_column], fontsize=11)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(alpha=0.2)
        ax.margins(x=0.04)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        fontsize=9,
        handlelength=1.8,
        columnspacing=1.6,
        frameon=False,
    )
    return save_figure(
        fig,
        "paper_figure_headline_metrics_grid",
        extra_dirs=[PAPER_FIGURE_DIR],
    )


def plot_fairness_tradeoff(
    comparison_df: pd.DataFrame,
    fairness_df: pd.DataFrame,
) -> list[Path]:
    merged_df = comparison_df.merge(
        fairness_df[
            [
                "variant",
                "model_family",
                "test_age_group_predictive_equality_ratio_min_over_max_mean",
            ]
        ],
        on=["variant", "model_family"],
        how="inner",
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    for model_family in MODEL_ORDER:
        model_df = merged_df.loc[merged_df["model_family"] == model_family].copy()
        ax.scatter(
            model_df["average_precision_mean"],
            model_df["test_age_group_predictive_equality_ratio_min_over_max_mean"],
            s=60,
            color=MODEL_COLORS[model_family],
            label=MODEL_LABELS[model_family],
        )
        for _, row in model_df.iterrows():
            ax.annotate(
                row["variant"],
                (
                    row["average_precision_mean"],
                    row["test_age_group_predictive_equality_ratio_min_over_max_mean"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("Average Precision")
    ax.set_ylabel("Age-group FPR Parity Ratio")
    ax.set_title("Fairness-Performance Tradeoff on TEST")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    return save_figure(
        fig,
        "paper_figure_fairness_performance_tradeoff",
        extra_dirs=[PAPER_FIGURE_DIR],
    )


def plot_vv_collapse_figure(vv_table_df: pd.DataFrame) -> list[Path]:
    figure_df = vv_table_df.copy()
    figure_df["valid_average_precision"] = figure_df["valid_average_precision"].astype(float)
    figure_df["test_average_precision"] = figure_df["test_average_precision"].astype(float)
    figure_df["valid_recall_at_5pct_fpr_proxy"] = figure_df["valid_recall_at_5pct_fpr_proxy"].astype(float)
    figure_df["test_recall_at_5pct_fpr_proxy"] = figure_df["test_recall_at_5pct_fpr_proxy"].astype(float)

    labels = figure_df["model_label"].tolist()
    x = np.arange(len(labels))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)

    axes[0].bar(x - width / 2, figure_df["valid_average_precision"], width=width, color="#4c78a8", label="VALID")
    axes[0].bar(x + width / 2, figure_df["test_average_precision"], width=width, color="#f58518", label="TEST")
    axes[0].set_title("VV Average Precision")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(
        x - width / 2,
        figure_df["valid_recall_at_5pct_fpr_proxy"],
        width=width,
        color="#4c78a8",
        label="VALID",
    )
    axes[1].bar(
        x + width / 2,
        figure_df["test_recall_at_5pct_fpr_proxy"],
        width=width,
        color="#f58518",
        label="TEST",
    )
    axes[1].set_title("VV Recall at Selected 5% FPR Operating Point")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)

    fig.suptitle("VV VALID -> TEST Collapse from Current Repo Artifacts", fontsize=13)
    return save_figure(
        fig,
        "paper_figure_vv_valid_test_collapse",
        extra_dirs=[PAPER_FIGURE_DIR],
    )


def plot_matched_ann_ablation_figure(ablation_df: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)
    x = np.arange(len(VARIANTS))
    metric_specs = [
        ("average_precision_mean", "average_precision_std", "Average Precision"),
        (
            "recall_at_selected_threshold_mean",
            "recall_at_selected_threshold_std",
            r"Recall at transferred threshold ($\tau^\star$)",
        ),
    ]

    for ax, (mean_col, std_col, title) in zip(axes, metric_specs):
        for model_family in ABLATION_MODEL_ORDER:
            model_df = (
                ablation_df.loc[ablation_df["model_family"] == model_family]
                .set_index("variant")
                .loc[VARIANTS]
                .reset_index()
            )
            y = model_df[mean_col].to_numpy(dtype=float)
            yerr = model_df[std_col].fillna(0.0).to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.0,
                color=ABLATION_MODEL_COLORS[model_family],
                label=ABLATION_MODEL_LABELS[model_family],
            )
            if np.any(yerr > 0.0):
                ax.fill_between(
                    x,
                    y - yerr,
                    y + yerr,
                    color=ABLATION_MODEL_COLORS[model_family],
                    alpha=0.12,
                    linewidth=0.0,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(VARIANTS, rotation=35, ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2)

    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Three-Way FT Readout Ablation on TEST", fontsize=13)
    return save_figure(
        fig,
        "paper_figure_matched_ann_ablation",
        extra_dirs=[PAPER_FIGURE_DIR],
    )


def write_manifest(rows: list[dict[str, str]]) -> None:
    fieldnames = ["artifact_type", "relative_path", "description", "source_builder", "source_inputs"]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme() -> None:
    README_PATH.write_text(
        "\n".join(
            [
                "# Final Paper Bundle",
                "",
                "This directory contains the frozen paper-level tables and figures regenerated from the current repo artifacts.",
                "",
                "Contents:",
                "- `tables/`: compact paper-facing CSV tables derived from the current summary artifacts",
                "- `figures/`: paper-facing figures in both PNG and PDF format",
                "- `paper_artifact_manifest.csv`: manifest linking each bundle artifact to its source builder and inputs",
                "",
                "Primary builder:",
                "- `scripts/build_final_paper_bundle.py`",
                "",
                "Upstream regenerated summary builders:",
                "- `scripts/build_baf_headline_test_tables.py`",
                "- `scripts/build_snn_uncertainty_tables.py`",
                "- `scripts/build_cross_variant_fairness_summary.py`",
                "- `scripts/build_baf_model_comparison_tables.py`",
                "- `scripts/build_cross_variant_fairness_summary_with_lightgbm.py`",
                "- `scripts/build_significance_test_tables.py`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_clean_output_dirs()
    run_prereq_builders()

    comparison_df = load_csv(SUMMARY_DIR / "paper_table_baf_headline_metrics_test_model_comparison_seed_mean_std.csv")
    fairness_df = load_csv(SUMMARY_DIR / "paper_table_fairness_cross_variant_model_seed_mean_std_with_lightgbm.csv")
    significance_key_df = load_csv(SUMMARY_DIR / "paper_table_significance_pairwise_bootstrap_key_claims.csv")
    matched_ann_ablation_raw_df = build_matched_ann_ablation_raw_table()

    main_table_df = build_main_comparison_table(comparison_df)
    fairness_table_df = build_fairness_table(comparison_df, fairness_df)
    significance_table_df = build_significance_table(significance_key_df)
    vv_table_df = build_vv_table()
    matched_ann_ablation_table_df = build_matched_ann_ablation_table(matched_ann_ablation_raw_df)

    main_table_path = FINAL_TABLE_DIR / "paper_final_table_main_comparison.csv"
    fairness_table_path = FINAL_TABLE_DIR / "paper_final_table_fairness_tradeoff.csv"
    significance_table_path = FINAL_TABLE_DIR / "paper_final_table_significance_key_results.csv"
    vv_table_path = FINAL_TABLE_DIR / "paper_final_table_vv_generalization.csv"
    matched_ann_ablation_table_path = FINAL_TABLE_DIR / "paper_final_table_matched_ann_ablation.csv"

    main_table_df.to_csv(main_table_path, index=False)
    fairness_table_df.to_csv(fairness_table_path, index=False)
    significance_table_df.to_csv(significance_table_path, index=False)
    vv_table_df.to_csv(vv_table_path, index=False)
    matched_ann_ablation_table_df.to_csv(matched_ann_ablation_table_path, index=False)

    manifest_rows: list[dict[str, str]] = [
        {
            "artifact_type": "table",
            "relative_path": str(main_table_path.relative_to(ROOT_DIR)),
            "description": "Compact main TEST headline comparison table across FT-dense, LightGBM, and SNN-head.",
            "source_builder": "scripts/build_final_paper_bundle.py",
            "source_inputs": "paper_summaries/tables/paper_table_baf_headline_metrics_test_model_comparison_seed_mean_std.csv",
        },
        {
            "artifact_type": "table",
            "relative_path": str(fairness_table_path.relative_to(ROOT_DIR)),
            "description": "Compact fairness-performance table using TEST AP, TPR@5%FPR, and parity summaries.",
            "source_builder": "scripts/build_final_paper_bundle.py",
            "source_inputs": (
                "paper_summaries/tables/paper_table_baf_headline_metrics_test_model_comparison_seed_mean_std.csv;"
                "paper_summaries/tables/paper_table_fairness_cross_variant_model_seed_mean_std_with_lightgbm.csv"
            ),
        },
        {
            "artifact_type": "table",
            "relative_path": str(significance_table_path.relative_to(ROOT_DIR)),
            "description": "FDR-significant headline comparison results for paper-ready reporting.",
            "source_builder": "scripts/build_final_paper_bundle.py",
            "source_inputs": "paper_summaries/tables/paper_table_significance_pairwise_bootstrap_key_claims.csv",
        },
        {
            "artifact_type": "table",
            "relative_path": str(vv_table_path.relative_to(ROOT_DIR)),
            "description": "VV VALID -> TEST collapse summary from current dense, SNN, and LightGBM run artifacts.",
            "source_builder": "scripts/build_final_paper_bundle.py",
            "source_inputs": (
                "ft_dense_baseline_100_run_VV/paper_artifacts/tables/paper_table_final_metrics.csv;"
                "snn_ftt_100_run_VV_seed42/paper_artifacts/tables/paper_table_final_metrics.csv;"
                "snn_ftt_100_run_VV_seed52/paper_artifacts/tables/paper_table_final_metrics.csv;"
                "lightgbm_baseline_100_run_VV/paper_artifacts/tables/paper_table_final_metrics.csv"
            ),
        },
        {
            "artifact_type": "table",
            "relative_path": str(matched_ann_ablation_table_path.relative_to(ROOT_DIR)),
            "description": "Compact three-way FT readout ablation table across dense, matched ANN, and SNN heads.",
            "source_builder": "scripts/build_final_paper_bundle.py",
            "source_inputs": (
                "ft_dense_baseline_100_run_*/paper_artifacts/tables/paper_table_final_metrics.csv;"
                "matched_ann_control_100_run_*_seed42/paper_artifacts/tables/paper_table_final_metrics.csv;"
                "matched_ann_control_100_run_*_seed52/paper_artifacts/tables/paper_table_final_metrics.csv;"
                "snn_ftt_100_run_*_seed42/paper_artifacts/tables/paper_table_final_metrics.csv;"
                "snn_ftt_100_run_*_seed52/paper_artifacts/tables/paper_table_final_metrics.csv"
            ),
        },
    ]

    for path in plot_headline_metrics_grid(comparison_df):
        manifest_rows.append(
            {
                "artifact_type": "figure",
                "relative_path": str(path.relative_to(ROOT_DIR)),
                "description": "Multi-panel TEST headline metric comparison across BAF variants.",
                "source_builder": "scripts/build_final_paper_bundle.py",
                "source_inputs": "paper_summaries/tables/paper_table_baf_headline_metrics_test_model_comparison_seed_mean_std.csv",
            }
        )
    for path in plot_fairness_tradeoff(comparison_df, fairness_df):
        manifest_rows.append(
            {
                "artifact_type": "figure",
                "relative_path": str(path.relative_to(ROOT_DIR)),
                "description": "Average Precision vs age-group FPR parity ratio tradeoff figure.",
                "source_builder": "scripts/build_final_paper_bundle.py",
                "source_inputs": (
                    "paper_summaries/tables/paper_table_baf_headline_metrics_test_model_comparison_seed_mean_std.csv;"
                    "paper_summaries/tables/paper_table_fairness_cross_variant_model_seed_mean_std_with_lightgbm.csv"
                ),
            }
        )
    for path in plot_vv_collapse_figure(vv_table_df):
        manifest_rows.append(
            {
                "artifact_type": "figure",
                "relative_path": str(path.relative_to(ROOT_DIR)),
                "description": "VV VALID -> TEST collapse figure from current run artifacts.",
                "source_builder": "scripts/build_final_paper_bundle.py",
                "source_inputs": str(vv_table_path.relative_to(ROOT_DIR)),
            }
        )
    for path in plot_matched_ann_ablation_figure(matched_ann_ablation_raw_df):
        manifest_rows.append(
            {
                "artifact_type": "figure",
                "relative_path": str(path.relative_to(ROOT_DIR)),
                "description": "Three-way FT readout ablation figure for Average Precision and recall at the transferred threshold.",
                "source_builder": "scripts/build_final_paper_bundle.py",
                "source_inputs": str(matched_ann_ablation_table_path.relative_to(ROOT_DIR)),
            }
        )

    write_manifest(manifest_rows)
    write_readme()

    print(f"Wrote {main_table_path.relative_to(ROOT_DIR)}")
    print(f"Wrote {fairness_table_path.relative_to(ROOT_DIR)}")
    print(f"Wrote {significance_table_path.relative_to(ROOT_DIR)}")
    print(f"Wrote {vv_table_path.relative_to(ROOT_DIR)}")
    print(f"Wrote {matched_ann_ablation_table_path.relative_to(ROOT_DIR)}")
    print(f"Wrote {MANIFEST_PATH.relative_to(ROOT_DIR)}")
    print(f"Wrote {README_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
