#!/usr/bin/env python3
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT_DIR / "paper_summaries" / "tables"
RUN_SUMMARY_PATH = OUT_DIR / "paper_table_baf_headline_metrics_test_model_comparison_run_summary.csv"
FAMILY_SUMMARY_PATH = OUT_DIR / "paper_table_baf_headline_metrics_test_model_comparison_seed_mean_std.csv"

VARIANTS = ["Base", "VI", "VII", "VIII", "VIV", "VV"]
RUN_SPECS = [
    {
        "model_key": "ft_dense",
        "model_family": "ft_dense",
        "model_label": "FT-Transformer dense head",
        "seed": "",
        "run_dir_template": "ft_dense_baseline_100_run_{variant}",
    },
    {
        "model_key": "lightgbm",
        "model_family": "lightgbm",
        "model_label": "LightGBM baseline",
        "seed": "",
        "run_dir_template": "lightgbm_baseline_100_run_{variant}",
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
RUN_SPECS_BY_KEY = {spec["model_key"]: spec for spec in RUN_SPECS}
MODEL_FAMILY_LABELS = {
    "ft_dense": "FT-Transformer dense head",
    "lightgbm": "LightGBM baseline",
    "snn_head": "FT-Transformer SNN head",
}
COMPARISONS = [
    ("snn_head", "ft_dense"),
    ("snn_head", "lightgbm"),
    ("ft_dense", "lightgbm"),
]
METRIC_COLUMNS = [
    "roc_auc",
    "average_precision",
    "balanced_accuracy_at_selected_threshold",
    "tpr_at_5pct_fpr_splitwise",
    "predictive_equality_age_ratio",
]
FPR_CAP = 0.05
ALPHA = 0.05
CI_LOWER_Q = ALPHA / 2.0
CI_UPPER_Q = 1.0 - (ALPHA / 2.0)
BOOTSTRAP_REPLICATES = int(os.environ.get("SIGNIFICANCE_BOOTSTRAP_REPLICATES", "1000"))
BOOTSTRAP_SEED = int(os.environ.get("SIGNIFICANCE_BOOTSTRAP_SEED", "20260301"))
MAX_WORKERS = int(
    os.environ.get("SIGNIFICANCE_MAX_WORKERS", str(min(len(VARIANTS), os.cpu_count() or 1)))
)
FLOAT_TOL = 1e-6


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return pd.read_csv(path)


def load_predictions_table(run_dir: Path) -> pd.DataFrame:
    predictions_path = run_dir / "paper_artifacts" / "predictions" / "predictions_test.csv"
    df = load_csv(predictions_path).copy()
    if "row_idx" not in df.columns:
        raise ValueError(f"Missing row_idx in predictions table: {predictions_path}")
    if df["split"].nunique() != 1 or df["split"].iloc[0] != "TEST":
        raise ValueError(f"Predictions table is not TEST-only as expected: {predictions_path}")
    return df.sort_values("row_idx").reset_index(drop=True)


def assert_aligned_reference(
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    run_dir: Path,
) -> None:
    reference_columns = ["row_idx", "y_true", "age_group"]
    for column in reference_columns:
        if not reference_df[column].equals(candidate_df[column]):
            raise ValueError(
                f"Prediction alignment mismatch for {run_dir.relative_to(ROOT_DIR)} on column {column!r}."
            )


def compute_tpr_at_fpr_cap(y_true: np.ndarray, y_prob: np.ndarray, cap: float = FPR_CAP) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    under_cap = tpr[fpr <= float(cap) + FLOAT_TOL]
    if under_cap.size == 0:
        return 0.0
    return float(np.max(under_cap))


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    positive_mask = y_true == 1
    negative_mask = ~positive_mask

    tp = int(np.sum(y_pred[positive_mask] == 1))
    fn = int(np.sum(y_pred[positive_mask] == 0))
    tn = int(np.sum(y_pred[negative_mask] == 0))
    fp = int(np.sum(y_pred[negative_mask] == 1))

    tpr = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    tnr = 0.0 if (tn + fp) == 0 else tn / (tn + fp)
    return float((tpr + tnr) / 2.0)


def compute_predictive_equality_age_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    age_group: np.ndarray,
) -> float:
    group_fprs: list[float] = []
    for group_value in np.unique(age_group):
        group_mask = age_group == group_value
        negative_mask = np.logical_and(group_mask, y_true == 0)
        negative_count = int(np.sum(negative_mask))
        if negative_count == 0:
            continue
        fp_count = int(np.sum(np.logical_and(negative_mask, y_pred == 1)))
        group_fprs.append(fp_count / negative_count)

    if not group_fprs:
        raise ValueError("Unable to compute age-group FPR parity ratio: no valid age-group negatives.")

    group_fpr_min = float(min(group_fprs))
    group_fpr_max = float(max(group_fprs))
    return 1.0 if group_fpr_max == 0.0 else float(group_fpr_min / group_fpr_max)


def compute_metric_bundle(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    age_group: np.ndarray,
) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "balanced_accuracy_at_selected_threshold": compute_balanced_accuracy(y_true, y_pred),
        "tpr_at_5pct_fpr_splitwise": compute_tpr_at_fpr_cap(y_true, y_prob, FPR_CAP),
        "predictive_equality_age_ratio": compute_predictive_equality_age_ratio(
            y_true, y_pred, age_group
        ),
    }


def assert_close(label: str, observed: float, expected: float, tol: float = FLOAT_TOL) -> None:
    if pd.isna(expected) and pd.isna(observed):
        return
    if abs(float(observed) - float(expected)) > tol:
        raise ValueError(
            f"Validation mismatch for {label}: observed={observed} expected={expected} tol={tol}"
        )


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    if p_values.empty:
        return p_values.copy()

    values = p_values.astype(float).to_numpy()
    order = np.argsort(values)
    ordered = values[order]
    n_tests = len(ordered)

    adjusted = np.empty(n_tests, dtype=float)
    running = 1.0
    for reverse_idx in range(n_tests - 1, -1, -1):
        rank = reverse_idx + 1
        candidate = ordered[reverse_idx] * n_tests / rank
        running = min(running, candidate)
        adjusted[reverse_idx] = running

    result = np.empty(n_tests, dtype=float)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(result, index=p_values.index)


def build_variant_detail_rows(
    variant: str,
    run_summary_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    print(f"[significance] variant={variant}: loading predictions and validating frozen metrics...", flush=True)

    run_prediction_data: dict[str, dict[str, np.ndarray]] = {}
    run_observed_metrics: dict[str, dict[str, float]] = {}
    reference_df: pd.DataFrame | None = None

    expected_run_rows = run_summary_df.loc[run_summary_df["variant"] == variant].copy()
    if len(expected_run_rows) != len(RUN_SPECS):
        raise ValueError(f"Expected {len(RUN_SPECS)} run-summary rows for {variant}, found {len(expected_run_rows)}.")

    expected_family_rows = family_summary_df.loc[family_summary_df["variant"] == variant].copy()
    if len(expected_family_rows) != len(MODEL_FAMILY_LABELS):
        raise ValueError(
            f"Expected {len(MODEL_FAMILY_LABELS)} family-summary rows for {variant}, found {len(expected_family_rows)}."
        )

    for spec in RUN_SPECS:
        run_dir = ROOT_DIR / spec["run_dir_template"].format(variant=variant)
        prediction_df = load_predictions_table(run_dir)
        if reference_df is None:
            reference_df = prediction_df[["row_idx", "y_true", "age_group"]].copy()
        else:
            assert_aligned_reference(reference_df, prediction_df[["row_idx", "y_true", "age_group"]], run_dir)

        y_true = prediction_df["y_true"].to_numpy(dtype=np.int8, copy=True)
        y_prob = prediction_df["y_prob"].to_numpy(dtype=np.float64, copy=True)
        y_pred = prediction_df["y_pred_at_selected_threshold"].to_numpy(dtype=np.int8, copy=True)
        age_group = prediction_df["age_group"].astype(str).to_numpy(copy=True)

        run_prediction_data[spec["model_key"]] = {
            "y_prob": y_prob,
            "y_pred": y_pred,
        }
        observed_metrics = compute_metric_bundle(y_true, y_prob, y_pred, age_group)
        run_observed_metrics[spec["model_key"]] = observed_metrics

        expected_row = expected_run_rows.loc[expected_run_rows["model_key"] == spec["model_key"]]
        if expected_row.empty:
            raise ValueError(f"Missing frozen run-summary row for variant={variant} model_key={spec['model_key']}.")
        expected_row = expected_row.iloc[0]
        for metric_name in METRIC_COLUMNS:
            assert_close(
                f"{variant} {spec['model_key']} {metric_name}",
                observed_metrics[metric_name],
                float(expected_row[metric_name]),
            )

    if reference_df is None:
        raise ValueError(f"No prediction data loaded for variant {variant}.")

    y_true_reference = reference_df["y_true"].to_numpy(dtype=np.int8, copy=True)
    age_group_reference = reference_df["age_group"].astype(str).to_numpy(copy=True)
    pos_idx = np.flatnonzero(y_true_reference == 1)
    neg_idx = np.flatnonzero(y_true_reference == 0)

    if pos_idx.size == 0 or neg_idx.size == 0:
        raise ValueError(f"Variant {variant} does not have both classes on TEST.")

    family_observed_metrics: dict[str, dict[str, float]] = {
        "ft_dense": run_observed_metrics["ft_dense"],
        "lightgbm": run_observed_metrics["lightgbm"],
        "snn_head": {
            metric_name: float(
                np.mean(
                    [
                        run_observed_metrics["snn_head_seed42"][metric_name],
                        run_observed_metrics["snn_head_seed52"][metric_name],
                    ]
                )
            )
            for metric_name in METRIC_COLUMNS
        },
    }

    for model_family, expected_group in expected_family_rows.groupby("model_family", sort=True):
        expected_row = expected_group.iloc[0]
        for metric_name in METRIC_COLUMNS:
            assert_close(
                f"{variant} {model_family} {metric_name}_mean",
                family_observed_metrics[model_family][metric_name],
                float(expected_row[f"{metric_name}_mean"]),
            )

    print(
        f"[significance] variant={variant}: validated frozen tables; running "
        f"{BOOTSTRAP_REPLICATES} paired stratified bootstrap replicates...",
        flush=True,
    )

    rng = np.random.default_rng(bootstrap_seed)
    run_bootstrap_metrics = {
        model_key: {
            metric_name: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
            for metric_name in METRIC_COLUMNS
        }
        for model_key in RUN_SPECS_BY_KEY
    }

    for bootstrap_idx in range(BOOTSTRAP_REPLICATES):
        sample_idx = np.concatenate(
            [
                pos_idx[rng.integers(0, pos_idx.size, size=pos_idx.size)],
                neg_idx[rng.integers(0, neg_idx.size, size=neg_idx.size)],
            ]
        )
        y_true_sample = y_true_reference[sample_idx]
        age_group_sample = age_group_reference[sample_idx]

        for model_key, prediction_data in run_prediction_data.items():
            metric_bundle = compute_metric_bundle(
                y_true_sample,
                prediction_data["y_prob"][sample_idx],
                prediction_data["y_pred"][sample_idx],
                age_group_sample,
            )
            for metric_name, metric_value in metric_bundle.items():
                run_bootstrap_metrics[model_key][metric_name][bootstrap_idx] = metric_value

    family_bootstrap_metrics: dict[str, dict[str, np.ndarray]] = {
        "ft_dense": run_bootstrap_metrics["ft_dense"],
        "lightgbm": run_bootstrap_metrics["lightgbm"],
        "snn_head": {
            metric_name: np.mean(
                np.vstack(
                    [
                        run_bootstrap_metrics["snn_head_seed42"][metric_name],
                        run_bootstrap_metrics["snn_head_seed52"][metric_name],
                    ]
                ),
                axis=0,
            )
            for metric_name in METRIC_COLUMNS
        },
    }

    rows: list[dict[str, object]] = []
    for model_a_family, model_b_family in COMPARISONS:
        comparison_key = f"{model_a_family}_minus_{model_b_family}"
        for metric_name in METRIC_COLUMNS:
            observed_a = family_observed_metrics[model_a_family][metric_name]
            observed_b = family_observed_metrics[model_b_family][metric_name]
            observed_diff = float(observed_a - observed_b)

            bootstrap_diff = (
                family_bootstrap_metrics[model_a_family][metric_name]
                - family_bootstrap_metrics[model_b_family][metric_name]
            )
            ci_lower = float(np.quantile(bootstrap_diff, CI_LOWER_Q))
            ci_upper = float(np.quantile(bootstrap_diff, CI_UPPER_Q))
            frac_gt_zero = float(np.mean(bootstrap_diff > 0.0))
            frac_lt_zero = float(np.mean(bootstrap_diff < 0.0))
            p_value = float(
                min(
                    1.0,
                    2.0
                    * min(
                        (np.sum(bootstrap_diff <= 0.0) + 1.0) / (BOOTSTRAP_REPLICATES + 1.0),
                        (np.sum(bootstrap_diff >= 0.0) + 1.0) / (BOOTSTRAP_REPLICATES + 1.0),
                    ),
                )
            )

            if ci_lower > 0.0:
                ci_direction = model_a_family
            elif ci_upper < 0.0:
                ci_direction = model_b_family
            else:
                ci_direction = "inconclusive"

            rows.append(
                {
                    "variant": variant,
                    "comparison_key": comparison_key,
                    "model_a_family": model_a_family,
                    "model_a_label": MODEL_FAMILY_LABELS[model_a_family],
                    "model_b_family": model_b_family,
                    "model_b_label": MODEL_FAMILY_LABELS[model_b_family],
                    "metric": metric_name,
                    "observed_model_a": observed_a,
                    "observed_model_b": observed_b,
                    "observed_diff_model_a_minus_model_b": observed_diff,
                    "bootstrap_ci_95_lower": ci_lower,
                    "bootstrap_ci_95_upper": ci_upper,
                    "bootstrap_fraction_diff_gt_zero": frac_gt_zero,
                    "bootstrap_fraction_diff_lt_zero": frac_lt_zero,
                    "p_value_two_sided_sign_bootstrap": p_value,
                    "ci_excludes_zero": bool(ci_lower > 0.0 or ci_upper < 0.0),
                    "ci_supported_direction": ci_direction,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": bootstrap_seed,
                    "bootstrap_scheme": "paired_class_stratified",
                    "source_family_table": str(FAMILY_SUMMARY_PATH.relative_to(ROOT_DIR)),
                    "source_run_table": str(RUN_SUMMARY_PATH.relative_to(ROOT_DIR)),
                }
            )

    print(f"[significance] variant={variant}: completed.", flush=True)
    return rows


def build_summary_table(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (comparison_key, metric), group_df in detail_df.groupby(["comparison_key", "metric"], sort=True):
        model_a_family = group_df["model_a_family"].iloc[0]
        model_b_family = group_df["model_b_family"].iloc[0]

        ci_model_a_variants = sorted(
            group_df.loc[group_df["ci_supported_direction"] == model_a_family, "variant"].astype(str).tolist()
        )
        ci_model_b_variants = sorted(
            group_df.loc[group_df["ci_supported_direction"] == model_b_family, "variant"].astype(str).tolist()
        )
        ci_inconclusive_variants = sorted(
            group_df.loc[group_df["ci_supported_direction"] == "inconclusive", "variant"].astype(str).tolist()
        )

        fdr_model_a_variants = sorted(
            group_df.loc[
                (group_df["bh_fdr_within_metric_comparison_0_05_significant"])
                & (group_df["observed_diff_model_a_minus_model_b"] > 0.0),
                "variant",
            ]
            .astype(str)
            .tolist()
        )
        fdr_model_b_variants = sorted(
            group_df.loc[
                (group_df["bh_fdr_within_metric_comparison_0_05_significant"])
                & (group_df["observed_diff_model_a_minus_model_b"] < 0.0),
                "variant",
            ]
            .astype(str)
            .tolist()
        )
        fdr_inconclusive_variants = sorted(
            group_df.loc[
                ~group_df["bh_fdr_within_metric_comparison_0_05_significant"], "variant"
            ]
            .astype(str)
            .tolist()
        )

        rows.append(
            {
                "comparison_key": comparison_key,
                "model_a_family": model_a_family,
                "model_a_label": MODEL_FAMILY_LABELS[model_a_family],
                "model_b_family": model_b_family,
                "model_b_label": MODEL_FAMILY_LABELS[model_b_family],
                "metric": metric,
                "n_variants": int(len(group_df)),
                "ci_model_a_better_count": int(len(ci_model_a_variants)),
                "ci_model_b_better_count": int(len(ci_model_b_variants)),
                "ci_inconclusive_count": int(len(ci_inconclusive_variants)),
                "ci_model_a_better_variants": ",".join(ci_model_a_variants),
                "ci_model_b_better_variants": ",".join(ci_model_b_variants),
                "ci_inconclusive_variants": ",".join(ci_inconclusive_variants),
                "bh_fdr_within_metric_comparison_model_a_better_count": int(len(fdr_model_a_variants)),
                "bh_fdr_within_metric_comparison_model_b_better_count": int(len(fdr_model_b_variants)),
                "bh_fdr_within_metric_comparison_inconclusive_count": int(len(fdr_inconclusive_variants)),
                "bh_fdr_within_metric_comparison_model_a_better_variants": ",".join(fdr_model_a_variants),
                "bh_fdr_within_metric_comparison_model_b_better_variants": ",".join(fdr_model_b_variants),
                "bh_fdr_within_metric_comparison_inconclusive_variants": ",".join(
                    fdr_inconclusive_variants
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(["comparison_key", "metric"]).reset_index(drop=True)


def main() -> None:
    if BOOTSTRAP_REPLICATES <= 0:
        raise ValueError("SIGNIFICANCE_BOOTSTRAP_REPLICATES must be positive.")
    if MAX_WORKERS <= 0:
        raise ValueError("SIGNIFICANCE_MAX_WORKERS must be positive.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_summary_df = load_csv(RUN_SUMMARY_PATH)
    family_summary_df = load_csv(FAMILY_SUMMARY_PATH)

    detail_rows: list[dict[str, object]] = []
    work_items = [
        {
            "variant": variant,
            "run_summary_df": run_summary_df,
            "family_summary_df": family_summary_df,
            "bootstrap_seed": BOOTSTRAP_SEED + variant_index,
        }
        for variant_index, variant in enumerate(VARIANTS)
    ]

    if MAX_WORKERS == 1:
        for work_item in work_items:
            detail_rows.extend(
                build_variant_detail_rows(
                    variant=work_item["variant"],
                    run_summary_df=work_item["run_summary_df"],
                    family_summary_df=work_item["family_summary_df"],
                    bootstrap_seed=work_item["bootstrap_seed"],
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(
                    build_variant_detail_rows,
                    variant=work_item["variant"],
                    run_summary_df=work_item["run_summary_df"],
                    family_summary_df=work_item["family_summary_df"],
                    bootstrap_seed=work_item["bootstrap_seed"],
                )
                for work_item in work_items
            ]
            for future in futures:
                detail_rows.extend(future.result())

    detail_df = pd.DataFrame(detail_rows).sort_values(
        ["comparison_key", "metric", "variant"]
    ).reset_index(drop=True)
    detail_df["p_value_bh_fdr_within_metric_comparison"] = (
        detail_df.groupby(["comparison_key", "metric"], sort=False)["p_value_two_sided_sign_bootstrap"]
        .transform(benjamini_hochberg)
        .astype(float)
    )
    detail_df["bh_fdr_within_metric_comparison_0_05_significant"] = (
        detail_df["p_value_bh_fdr_within_metric_comparison"] < 0.05
    )

    key_claims_df = detail_df.loc[
        detail_df["comparison_key"].isin(["snn_head_minus_ft_dense", "snn_head_minus_lightgbm"])
    ].reset_index(drop=True)
    summary_df = build_summary_table(detail_df)

    detail_path = OUT_DIR / "paper_table_significance_pairwise_bootstrap_tests.csv"
    key_claims_path = OUT_DIR / "paper_table_significance_pairwise_bootstrap_key_claims.csv"
    summary_path = OUT_DIR / "paper_table_significance_pairwise_bootstrap_summary.csv"

    detail_df.to_csv(detail_path, index=False)
    key_claims_df.to_csv(key_claims_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Wrote {detail_path.relative_to(ROOT_DIR)}")
    print(f"Wrote {key_claims_path.relative_to(ROOT_DIR)}")
    print(f"Wrote {summary_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
