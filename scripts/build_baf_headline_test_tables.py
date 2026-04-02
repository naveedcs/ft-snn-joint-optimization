#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT_DIR / "paper_summaries" / "tables"

VARIANTS = ["Base", "VI", "VII", "VIII", "VIV", "VV"]
MODEL_SPECS = [
    {
        "model_key": "ft_dense",
        "model_family": "ft_dense",
        "model_label": "FT-Transformer dense head",
        "seed": "",
        "run_dir_template": "ft_dense_baseline_100_run_{variant}",
    },
    {
        "model_key": "snn_head_seed42",
        "model_family": "snn_head",
        "model_label": "FT-Transformer SNN head (seed42)",
        "seed": "42",
        "run_dir_template": "snn_ftt_100_run_{variant}_seed42",
    },
    {
        "model_key": "snn_head_seed52",
        "model_family": "snn_head",
        "model_label": "FT-Transformer SNN head (seed52)",
        "seed": "52",
        "run_dir_template": "snn_ftt_100_run_{variant}_seed52",
    },
]
MODEL_FAMILY_LABELS = {
    "ft_dense": "FT-Transformer dense head",
    "snn_head": "FT-Transformer SNN head",
}
FPR_CAP = 0.05
FLOAT_TOL = 1e-9


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return pd.read_csv(path)


def extract_split_row(df: pd.DataFrame, split_name: str) -> pd.Series:
    split_rows = df.loc[df["split"] == split_name]
    if split_rows.empty:
        raise ValueError(f"Missing split={split_name!r} in {df.columns.tolist()}.")
    return split_rows.iloc[0]


def compute_tpr_at_fpr_cap(roc_df: pd.DataFrame, cap: float = FPR_CAP) -> float:
    if "fpr" not in roc_df.columns or "tpr" not in roc_df.columns:
        raise ValueError("ROC curve table must include 'fpr' and 'tpr' columns.")

    roc_work = roc_df.copy()
    roc_work["fpr"] = pd.to_numeric(roc_work["fpr"], errors="coerce")
    roc_work["tpr"] = pd.to_numeric(roc_work["tpr"], errors="coerce")
    under_cap = roc_work.loc[roc_work["fpr"] <= float(cap) + FLOAT_TOL, "tpr"].dropna()
    if under_cap.empty:
        return 0.0
    return float(under_cap.max())


def extract_age_group_predictive_equality_ratio(subgroup_df: pd.DataFrame) -> tuple[float, float, float]:
    age_rows = subgroup_df.loc[subgroup_df["attribute_name"] == "age_group"].copy()
    if age_rows.empty:
        raise ValueError("Missing age_group rows in subgroup metrics table.")

    age_rows["fpr"] = pd.to_numeric(age_rows["fpr"], errors="coerce")
    if age_rows["fpr"].isna().any():
        raise ValueError("Age-group subgroup metrics contain non-numeric FPR values.")

    fpr_min = float(age_rows["fpr"].min())
    fpr_max = float(age_rows["fpr"].max())
    ratio = 1.0 if fpr_max == 0.0 else float(fpr_min / fpr_max)
    return ratio, fpr_min, fpr_max


def extract_age_group_parity_from_attr_summary(attr_df: pd.DataFrame) -> float:
    age_rows = attr_df.loc[attr_df["attribute_name"] == "age_group"]
    if age_rows.empty:
        raise ValueError("Missing age_group row in fairness attribute parity summary table.")
    return float(age_rows.iloc[0]["test_fpr_parity_score"])


def build_run_summary_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for variant in VARIANTS:
        for spec in MODEL_SPECS:
            run_dir = ROOT_DIR / spec["run_dir_template"].format(variant=variant)
            tables_dir = run_dir / "paper_artifacts" / "tables"
            curves_dir = run_dir / "paper_artifacts" / "curves"

            final_metrics_df = load_csv(tables_dir / "paper_table_final_metrics.csv")
            confusion_df = load_csv(tables_dir / "paper_table_confusion_summary.csv")
            subgroup_test_df = load_csv(tables_dir / "paper_table_subgroup_metrics_test.csv")
            attr_summary_df = load_csv(tables_dir / "paper_table_fairness_attr_parity_summary.csv")
            roc_test_df = load_csv(curves_dir / "curve_roc_test.csv")

            test_final = extract_split_row(final_metrics_df, "TEST")
            test_confusion = extract_split_row(confusion_df, "TEST")

            predictive_equality_ratio, age_group_fpr_min, age_group_fpr_max = (
                extract_age_group_predictive_equality_ratio(subgroup_test_df)
            )
            attr_summary_ratio = extract_age_group_parity_from_attr_summary(attr_summary_df)
            if abs(predictive_equality_ratio - attr_summary_ratio) > 1e-6:
                raise ValueError(
                    "Age-group FPR parity ratio mismatch for "
                    f"{run_dir.relative_to(ROOT_DIR)}: subgroup={predictive_equality_ratio} "
                    f"vs attr_summary={attr_summary_ratio}"
                )

            rows.append(
                {
                    "variant": variant,
                    "model_key": spec["model_key"],
                    "model_family": spec["model_family"],
                    "model_label": spec["model_label"],
                    "seed": spec["seed"],
                    "source_run_dir": str(run_dir.relative_to(ROOT_DIR)),
                    "split": "TEST",
                    "roc_auc": float(test_final["auc"]),
                    "average_precision": float(test_final["pr_average_precision_score"]),
                    "balanced_accuracy_at_selected_threshold": float(
                        test_confusion["balanced_accuracy"]
                    ),
                    "tpr_at_5pct_fpr_splitwise": compute_tpr_at_fpr_cap(roc_test_df, FPR_CAP),
                    "predictive_equality_age_ratio": predictive_equality_ratio,
                    "age_group_fpr_min": age_group_fpr_min,
                    "age_group_fpr_max": age_group_fpr_max,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["variant", "model_family", "seed", "model_key"])
        .reset_index(drop=True)
    )


def build_seed_mean_std_table(run_summary_df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy_at_selected_threshold",
        "tpr_at_5pct_fpr_splitwise",
        "predictive_equality_age_ratio",
        "age_group_fpr_min",
        "age_group_fpr_max",
    ]

    rows: list[dict[str, object]] = []
    for (variant, model_family), group_df in run_summary_df.groupby(
        ["variant", "model_family"], sort=True
    ):
        row: dict[str, object] = {
            "variant": variant,
            "model_family": model_family,
            "model_label": MODEL_FAMILY_LABELS.get(model_family, model_family),
            "split": "TEST",
            "n_runs": int(len(group_df)),
            "seed_list": ",".join(
                seed for seed in sorted(group_df["seed"].fillna("").astype(str).unique()) if seed
            ),
            "source_run_dirs": ";".join(sorted(group_df["source_run_dir"].astype(str).tolist())),
        }
        for column in numeric_columns:
            row[f"{column}_mean"] = float(group_df[column].mean())
            row[f"{column}_std"] = (
                float(group_df[column].std(ddof=1)) if len(group_df) > 1 else float("nan")
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["variant", "model_family"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_summary_df = build_run_summary_table()
    seed_mean_std_df = build_seed_mean_std_table(run_summary_df)

    run_summary_path = (
        OUT_DIR / "paper_table_baf_headline_metrics_test_ft_dense_snn_run_summary.csv"
    )
    seed_mean_std_path = (
        OUT_DIR / "paper_table_baf_headline_metrics_test_ft_dense_snn_seed_mean_std.csv"
    )

    run_summary_df.to_csv(run_summary_path, index=False)
    seed_mean_std_df.to_csv(seed_mean_std_path, index=False)

    print(f"Wrote {run_summary_path.relative_to(ROOT_DIR)}")
    print(f"Wrote {seed_mean_std_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
