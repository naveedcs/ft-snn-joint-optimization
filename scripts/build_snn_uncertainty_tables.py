#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT_DIR / "paper_summaries" / "tables"

VARIANTS = ["Base", "VI", "VII", "VIII", "VIV", "VV"]
SEEDS = ["42", "52"]
MODEL_LABEL = "FT-Transformer SNN head"
FPR_CAP = 0.05
FLOAT_TOL = 1e-9


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return pd.read_csv(path)


def extract_split_row(df: pd.DataFrame, split_name: str) -> pd.Series:
    rows = df.loc[df["split"] == split_name]
    if rows.empty:
        raise ValueError(f"Missing split={split_name!r} in table with columns {df.columns.tolist()}.")
    return rows.iloc[0]


def compute_tpr_at_fpr_cap(roc_df: pd.DataFrame, cap: float = FPR_CAP) -> float:
    roc_work = roc_df.copy()
    roc_work["fpr"] = pd.to_numeric(roc_work["fpr"], errors="coerce")
    roc_work["tpr"] = pd.to_numeric(roc_work["tpr"], errors="coerce")
    under_cap = roc_work.loc[roc_work["fpr"] <= float(cap) + FLOAT_TOL, "tpr"].dropna()
    if under_cap.empty:
        return 0.0
    return float(under_cap.max())


def extract_predictive_equality_age_ratio(subgroup_df: pd.DataFrame, split_name: str) -> float:
    split_rows = subgroup_df.loc[
        (subgroup_df["split"] == split_name) & (subgroup_df["attribute_name"] == "age_group")
    ].copy()
    if split_rows.empty:
        raise ValueError(f"Missing age_group rows for split={split_name!r} in subgroup metrics.")

    split_rows["fpr"] = pd.to_numeric(split_rows["fpr"], errors="coerce")
    if split_rows["fpr"].isna().any():
        raise ValueError("Age-group subgroup metrics contain non-numeric FPR values.")

    fpr_min = float(split_rows["fpr"].min())
    fpr_max = float(split_rows["fpr"].max())
    return 1.0 if fpr_max == 0.0 else float(fpr_min / fpr_max)


def build_run_level_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for variant in VARIANTS:
        for seed in SEEDS:
            run_dir = ROOT_DIR / f"snn_ftt_100_run_{variant}_seed{seed}"
            tables_dir = run_dir / "paper_artifacts" / "tables"
            curves_dir = run_dir / "paper_artifacts" / "curves"

            final_metrics_df = load_csv(tables_dir / "paper_table_final_metrics.csv")
            confusion_df = load_csv(tables_dir / "paper_table_confusion_summary.csv")
            subgroup_df = load_csv(tables_dir / "paper_table_subgroup_metrics_test.csv")
            roc_valid_df = load_csv(curves_dir / "curve_roc_valid.csv")
            roc_test_df = load_csv(curves_dir / "curve_roc_test.csv")

            valid_final = extract_split_row(final_metrics_df, "VALID")
            test_final = extract_split_row(final_metrics_df, "TEST")
            valid_confusion = extract_split_row(confusion_df, "VALID")
            test_confusion = extract_split_row(confusion_df, "TEST")

            valid_tpr_5 = compute_tpr_at_fpr_cap(roc_valid_df)
            test_tpr_5 = compute_tpr_at_fpr_cap(roc_test_df)
            test_predictive_equality = extract_predictive_equality_age_ratio(subgroup_df, "TEST")

            rows.append(
                {
                    "variant": variant,
                    "model_family": "snn_head",
                    "model_label": MODEL_LABEL,
                    "seed": seed,
                    "source_run_dir": str(run_dir.relative_to(ROOT_DIR)),
                    "valid_auc": float(valid_final["auc"]),
                    "test_auc": float(test_final["auc"]),
                    "delta_test_minus_valid_auc": float(test_final["auc"]) - float(valid_final["auc"]),
                    "valid_average_precision": float(valid_final["pr_average_precision_score"]),
                    "test_average_precision": float(test_final["pr_average_precision_score"]),
                    "delta_test_minus_valid_average_precision": float(test_final["pr_average_precision_score"])
                    - float(valid_final["pr_average_precision_score"]),
                    "valid_pr_auc_trapezoidal": float(valid_final["pr_auc_trapezoidal"]),
                    "test_pr_auc_trapezoidal": float(test_final["pr_auc_trapezoidal"]),
                    "delta_test_minus_valid_pr_auc_trapezoidal": float(test_final["pr_auc_trapezoidal"])
                    - float(valid_final["pr_auc_trapezoidal"]),
                    "valid_recall_at_selected_threshold": float(valid_final["recall_at_selected_threshold"]),
                    "test_recall_at_selected_threshold": float(test_final["recall_at_selected_threshold"]),
                    "delta_test_minus_valid_recall_at_selected_threshold": float(
                        test_final["recall_at_selected_threshold"]
                    )
                    - float(valid_final["recall_at_selected_threshold"]),
                    "valid_fpr_at_selected_threshold": float(valid_final["fpr_at_selected_threshold"]),
                    "test_fpr_at_selected_threshold": float(test_final["fpr_at_selected_threshold"]),
                    "delta_test_minus_valid_fpr_at_selected_threshold": float(
                        test_final["fpr_at_selected_threshold"]
                    )
                    - float(valid_final["fpr_at_selected_threshold"]),
                    "valid_precision_at_selected_threshold": float(valid_final["precision_at_selected_threshold"]),
                    "test_precision_at_selected_threshold": float(test_final["precision_at_selected_threshold"]),
                    "delta_test_minus_valid_precision_at_selected_threshold": float(
                        test_final["precision_at_selected_threshold"]
                    )
                    - float(valid_final["precision_at_selected_threshold"]),
                    "valid_balanced_accuracy_at_selected_threshold": float(valid_confusion["balanced_accuracy"]),
                    "test_balanced_accuracy_at_selected_threshold": float(test_confusion["balanced_accuracy"]),
                    "delta_test_minus_valid_balanced_accuracy_at_selected_threshold": float(
                        test_confusion["balanced_accuracy"]
                    )
                    - float(valid_confusion["balanced_accuracy"]),
                    "valid_tpr_at_5pct_fpr_splitwise": valid_tpr_5,
                    "test_tpr_at_5pct_fpr_splitwise": test_tpr_5,
                    "delta_test_minus_valid_tpr_at_5pct_fpr_splitwise": test_tpr_5 - valid_tpr_5,
                    "test_predictive_equality_age_ratio": test_predictive_equality,
                    "selected_threshold": float(test_final["selected_threshold"]),
                    "positive_rate_pred_test": float(test_final["positive_rate_pred"]),
                }
            )

    return pd.DataFrame(rows).sort_values(["variant", "seed"]).reset_index(drop=True)


def build_seed_aggregate_table(run_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, group_df in run_df.groupby("variant", sort=True):
        row: dict[str, object] = {
            "variant": variant,
            "model_family": "snn_head",
            "model_label": MODEL_LABEL,
            "n_runs": int(len(group_df)),
            "seed_list": ",".join(sorted(group_df["seed"].astype(str).tolist())),
            "source_run_dirs": ";".join(sorted(group_df["source_run_dir"].astype(str).tolist())),
        }
        for column in columns:
            row[f"{column}_mean"] = float(group_df[column].mean())
            row[f"{column}_std"] = float(group_df[column].std(ddof=1)) if len(group_df) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["variant"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_df = build_run_level_table()

    headline_columns = [
        "test_auc",
        "test_average_precision",
        "test_balanced_accuracy_at_selected_threshold",
        "test_tpr_at_5pct_fpr_splitwise",
        "test_predictive_equality_age_ratio",
    ]
    detailed_columns = [
        "valid_auc",
        "test_auc",
        "delta_test_minus_valid_auc",
        "valid_average_precision",
        "test_average_precision",
        "delta_test_minus_valid_average_precision",
        "valid_pr_auc_trapezoidal",
        "test_pr_auc_trapezoidal",
        "delta_test_minus_valid_pr_auc_trapezoidal",
        "valid_recall_at_selected_threshold",
        "test_recall_at_selected_threshold",
        "delta_test_minus_valid_recall_at_selected_threshold",
        "valid_fpr_at_selected_threshold",
        "test_fpr_at_selected_threshold",
        "delta_test_minus_valid_fpr_at_selected_threshold",
        "valid_precision_at_selected_threshold",
        "test_precision_at_selected_threshold",
        "delta_test_minus_valid_precision_at_selected_threshold",
        "valid_balanced_accuracy_at_selected_threshold",
        "test_balanced_accuracy_at_selected_threshold",
        "delta_test_minus_valid_balanced_accuracy_at_selected_threshold",
        "valid_tpr_at_5pct_fpr_splitwise",
        "test_tpr_at_5pct_fpr_splitwise",
        "delta_test_minus_valid_tpr_at_5pct_fpr_splitwise",
        "test_predictive_equality_age_ratio",
        "selected_threshold",
        "positive_rate_pred_test",
    ]
    delta_columns = [
        "delta_test_minus_valid_auc",
        "delta_test_minus_valid_average_precision",
        "delta_test_minus_valid_pr_auc_trapezoidal",
        "delta_test_minus_valid_recall_at_selected_threshold",
        "delta_test_minus_valid_fpr_at_selected_threshold",
        "delta_test_minus_valid_precision_at_selected_threshold",
        "delta_test_minus_valid_balanced_accuracy_at_selected_threshold",
        "delta_test_minus_valid_tpr_at_5pct_fpr_splitwise",
    ]

    headline_df = build_seed_aggregate_table(run_df, headline_columns)
    detailed_df = build_seed_aggregate_table(run_df, detailed_columns)
    delta_df = build_seed_aggregate_table(run_df, delta_columns)

    run_path = OUT_DIR / "paper_table_snn_seed_uncertainty_run_level.csv"
    headline_path = OUT_DIR / "paper_table_snn_seed_uncertainty_test_headline_mean_std.csv"
    detailed_path = OUT_DIR / "paper_table_snn_seed_uncertainty_detailed_mean_std.csv"
    delta_path = OUT_DIR / "paper_table_snn_seed_uncertainty_valid_to_test_delta_mean_std.csv"

    run_df.to_csv(run_path, index=False)
    headline_df.to_csv(headline_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)
    delta_df.to_csv(delta_path, index=False)

    print(f"Wrote {run_path.relative_to(ROOT_DIR)}")
    print(f"Wrote {headline_path.relative_to(ROOT_DIR)}")
    print(f"Wrote {detailed_path.relative_to(ROOT_DIR)}")
    print(f"Wrote {delta_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
