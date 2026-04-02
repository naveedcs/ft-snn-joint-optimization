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
ATTRIBUTES = ["age_group", "income_group", "employment_status_group"]
MODEL_FAMILY_LABELS = {
    "ft_dense": "FT-Transformer dense head",
    "snn_head": "FT-Transformer SNN head",
}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return pd.read_csv(path)


def extract_split_row(final_metrics_df: pd.DataFrame, split_name: str) -> pd.Series:
    split_rows = final_metrics_df.loc[final_metrics_df["split"] == split_name]
    if split_rows.empty:
        raise ValueError(f"Missing split={split_name!r} in final metrics table.")
    return split_rows.iloc[0]


def extract_attr_scores(attr_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    score_map: dict[str, dict[str, float]] = {}
    for _, row in attr_df.iterrows():
        score_map[str(row["attribute_name"])] = {
            "valid": float(row["valid_fpr_parity_score"]),
            "test": float(row["test_fpr_parity_score"]),
        }
    return score_map


def extract_age_group_metrics(subgroup_df: pd.DataFrame, split_name: str) -> dict[str, float]:
    age_rows = subgroup_df.loc[subgroup_df["attribute_name"] == "age_group"].copy()
    if age_rows.empty:
        raise ValueError(f"Missing age_group rows in subgroup metrics for split={split_name!r}.")

    age_rows["fpr"] = age_rows["fpr"].astype(float)
    age_rows["attribute_value"] = age_rows["attribute_value"].astype(str)

    fpr_by_group = {row["attribute_value"]: float(row["fpr"]) for _, row in age_rows.iterrows()}
    fprs = list(fpr_by_group.values())
    fpr_min = min(fprs)
    fpr_max = max(fprs)
    predictive_equality_ratio = 1.0 if fpr_max == 0 else fpr_min / fpr_max

    return {
        f"{split_name.lower()}_age_group_fpr_lt50": float(fpr_by_group.get("<50", float("nan"))),
        f"{split_name.lower()}_age_group_fpr_ge50": float(fpr_by_group.get(">=50", float("nan"))),
        f"{split_name.lower()}_age_group_fpr_gap_abs": float(fpr_max - fpr_min),
        f"{split_name.lower()}_age_group_predictive_equality_ratio_min_over_max": float(
            predictive_equality_ratio
        ),
    }


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    model_summary_rows: list[dict[str, object]] = []
    attribute_summary_rows: list[dict[str, object]] = []

    for variant in VARIANTS:
        for spec in MODEL_SPECS:
            run_dir = ROOT_DIR / spec["run_dir_template"].format(variant=variant)
            tables_dir = run_dir / "paper_artifacts" / "tables"

            final_metrics_df = load_csv(tables_dir / "paper_table_final_metrics.csv")
            attr_df = load_csv(tables_dir / "paper_table_fairness_attr_parity_summary.csv")
            valid_subgroup_df = load_csv(tables_dir / "paper_table_subgroup_metrics_valid.csv")
            test_subgroup_df = load_csv(tables_dir / "paper_table_subgroup_metrics_test.csv")

            valid_row = extract_split_row(final_metrics_df, "VALID")
            test_row = extract_split_row(final_metrics_df, "TEST")
            attr_scores = extract_attr_scores(attr_df)
            valid_age_metrics = extract_age_group_metrics(valid_subgroup_df, "VALID")
            test_age_metrics = extract_age_group_metrics(test_subgroup_df, "TEST")

            summary_row: dict[str, object] = {
                "variant": variant,
                "model_key": spec["model_key"],
                "model_family": spec["model_family"],
                "model_label": spec["model_label"],
                "seed": spec["seed"],
                "source_run_dir": str(run_dir.relative_to(ROOT_DIR)),
                "valid_overall_aequitas_fpr_parity": float(valid_row["aequitas_fpr_parity_overall"]),
                "test_overall_aequitas_fpr_parity": float(test_row["aequitas_fpr_parity_overall"]),
            }

            for attribute_name in ATTRIBUTES:
                if attribute_name not in attr_scores:
                    raise ValueError(f"Missing attribute={attribute_name!r} in {tables_dir}")
                summary_row[f"valid_{attribute_name}_aequitas_fpr_parity"] = attr_scores[attribute_name]["valid"]
                summary_row[f"test_{attribute_name}_aequitas_fpr_parity"] = attr_scores[attribute_name]["test"]

            summary_row.update(valid_age_metrics)
            summary_row.update(test_age_metrics)
            model_summary_rows.append(summary_row)

            for attribute_name in ATTRIBUTES:
                attribute_summary_rows.append(
                    {
                        "variant": variant,
                        "model_key": spec["model_key"],
                        "model_family": spec["model_family"],
                        "model_label": spec["model_label"],
                        "seed": spec["seed"],
                        "attribute_name": attribute_name,
                        "valid_fpr_parity_score": attr_scores[attribute_name]["valid"],
                        "test_fpr_parity_score": attr_scores[attribute_name]["test"],
                        "source_run_dir": str(run_dir.relative_to(ROOT_DIR)),
                    }
                )

    model_summary_df = (
        pd.DataFrame(model_summary_rows)
        .sort_values(["variant", "model_family", "seed", "model_key"])
        .reset_index(drop=True)
    )
    attribute_summary_df = (
        pd.DataFrame(attribute_summary_rows)
        .sort_values(["attribute_name", "variant", "model_family", "seed", "model_key"])
        .reset_index(drop=True)
    )
    return model_summary_df, attribute_summary_df


def build_seed_aggregate_tables(
    model_summary_df: pd.DataFrame,
    attribute_summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_excluded_columns = {
        "variant",
        "model_key",
        "model_family",
        "model_label",
        "seed",
        "source_run_dir",
    }
    model_numeric_columns = [
        column
        for column in model_summary_df.columns
        if column not in model_excluded_columns and pd.api.types.is_numeric_dtype(model_summary_df[column])
    ]

    model_seed_rows: list[dict[str, object]] = []
    for (variant, model_family), group_df in model_summary_df.groupby(["variant", "model_family"], sort=True):
        row: dict[str, object] = {
            "variant": variant,
            "model_family": model_family,
            "model_label": MODEL_FAMILY_LABELS.get(model_family, model_family),
            "n_runs": int(len(group_df)),
            "seed_list": ",".join(seed for seed in sorted(group_df["seed"].fillna("").astype(str).unique()) if seed),
            "source_run_dirs": ";".join(sorted(group_df["source_run_dir"].astype(str).tolist())),
        }
        for column in model_numeric_columns:
            row[f"{column}_mean"] = float(group_df[column].mean())
            row[f"{column}_std"] = float(group_df[column].std(ddof=1)) if len(group_df) > 1 else float("nan")
        model_seed_rows.append(row)

    attribute_seed_rows: list[dict[str, object]] = []
    for (variant, model_family, attribute_name), group_df in attribute_summary_df.groupby(
        ["variant", "model_family", "attribute_name"],
        sort=True,
    ):
        row = {
            "variant": variant,
            "model_family": model_family,
            "model_label": MODEL_FAMILY_LABELS.get(model_family, model_family),
            "attribute_name": attribute_name,
            "n_runs": int(len(group_df)),
            "seed_list": ",".join(seed for seed in sorted(group_df["seed"].fillna("").astype(str).unique()) if seed),
            "source_run_dirs": ";".join(sorted(group_df["source_run_dir"].astype(str).tolist())),
            "valid_fpr_parity_score_mean": float(group_df["valid_fpr_parity_score"].mean()),
            "valid_fpr_parity_score_std": float(group_df["valid_fpr_parity_score"].std(ddof=1))
            if len(group_df) > 1
            else float("nan"),
            "test_fpr_parity_score_mean": float(group_df["test_fpr_parity_score"].mean()),
            "test_fpr_parity_score_std": float(group_df["test_fpr_parity_score"].std(ddof=1))
            if len(group_df) > 1
            else float("nan"),
        }
        attribute_seed_rows.append(row)

    model_seed_df = (
        pd.DataFrame(model_seed_rows)
        .sort_values(["variant", "model_family"])
        .reset_index(drop=True)
    )
    attribute_seed_df = (
        pd.DataFrame(attribute_seed_rows)
        .sort_values(["attribute_name", "variant", "model_family"])
        .reset_index(drop=True)
    )
    return model_seed_df, attribute_seed_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_summary_df, attribute_summary_df = build_tables()
    model_seed_df, attribute_seed_df = build_seed_aggregate_tables(model_summary_df, attribute_summary_df)

    model_summary_path = OUT_DIR / "paper_table_fairness_cross_variant_model_summary.csv"
    attribute_summary_path = OUT_DIR / "paper_table_fairness_cross_variant_attribute_summary.csv"
    model_seed_path = OUT_DIR / "paper_table_fairness_cross_variant_model_seed_mean_std.csv"
    attribute_seed_path = OUT_DIR / "paper_table_fairness_cross_variant_attribute_seed_mean_std.csv"

    model_summary_df.to_csv(model_summary_path, index=False)
    attribute_summary_df.to_csv(attribute_summary_path, index=False)
    model_seed_df.to_csv(model_seed_path, index=False)
    attribute_seed_df.to_csv(attribute_seed_path, index=False)

    print(f"Wrote {model_summary_path.relative_to(ROOT_DIR)}")
    print(f"Wrote {attribute_summary_path.relative_to(ROOT_DIR)}")
    print(f"Wrote {model_seed_path.relative_to(ROOT_DIR)}")
    print(f"Wrote {attribute_seed_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
