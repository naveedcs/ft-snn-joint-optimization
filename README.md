# FT-SNN Joint Optimization

This repository contains runnable experiment pipelines for the Bank Account Fraud (BAF) variants and the exported artifacts they generate.

## What You Need

- Python `3.10`
- Access to the public Bank Account Fraud (BAF) dataset suite
- GPU access for the FT-based runs if you want the same workflow used in this repo

To recreate the original Python environment used for the recorded runs:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r reproducibility/pip-freeze-2026-02-28.txt
```

If you already have a working environment for these notebooks and scripts, you can skip this step.

## Dataset

This project uses the public Bank Account Fraud (BAF) dataset suite introduced in "Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation".

Dataset source:

- Official repository: `https://github.com/feedzai/bank-account-fraud`
- Public download page: `https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022`

After downloading the dataset, you can point the runs to your local copy with:

- `BAF_DATASET_DIR`
- `BAF_DATASET_PATH`
- `DATASET_DIR` for the matched ANN control script

## How To Run

### 1. SNN-head runs across all 6 variants

```bash
bash run_6_variations_parallel.sh --wait
```

Outputs are written into folders named like:

```text
snn_ftt_100_run_<TAG>_seed<SEED>/
parallel_runs/<timestamp>/
```

### 2. FT-dense baseline runs across all 6 variants

```bash
bash run_6_variations_dense_gpu0_gpu2.sh --wait
```

Outputs are written into folders named like:

```text
ft_dense_baseline_100_run_<TAG>/
parallel_runs_dense_baseline/<timestamp>/
```

### 3. LightGBM baseline runs across all 6 variants

CPU run:

```bash
LIGHTGBM_DEVICE_TYPE=cpu LIGHTGBM_N_JOBS=4 bash run_6_variations_lightgbm_gpu0_gpu2.sh --wait
```

CUDA run, only if your installed `lightgbm` supports CUDA:

```bash
LIGHTGBM_DEVICE_TYPE=cuda LIGHTGBM_GPU_PLATFORM_ID=0 LIGHTGBM_GPU_DEVICE_ID=0 LIGHTGBM_N_JOBS=4 bash run_6_variations_lightgbm_gpu0_gpu2.sh --wait
```

Outputs are written into folders named like:

```text
lightgbm_baseline_100_run_<TAG>/
parallel_runs_lightgbm_baseline/<timestamp>/
```

### 4. Optional matched ANN control runs

```bash
bash run_6_variations_matched_ann_control.sh --wait
```

Outputs are written into folders named like:

```text
matched_ann_control_100_run_<TAG>_seed<SEED>/
parallel_runs_matched_ann_control/<timestamp>/
```

## Monitoring A Run

You can watch logs while a launcher is running:

```bash
tail -f parallel_runs/*/logs/*.log
tail -f parallel_runs_dense_baseline/*/logs/*.log
tail -f parallel_runs_lightgbm_baseline/*/logs/*.log
tail -f parallel_runs_matched_ann_control/*/logs/*.log
```

## Exported Outputs

Each completed run writes exportable artifacts inside its run directory. The main folders to look for are:

```text
paper_artifacts/
csv_exports/
model_artifacts/
optuna/
```

The most useful exported files are usually:

```text
paper_artifacts/tables/paper_table_final_metrics.csv
paper_artifacts/tables/paper_table_fairness_attr_parity_summary.csv
paper_artifacts/predictions/predictions_test.csv
paper_artifacts/predictions/predictions_valid.csv
paper_artifacts/paper_artifact_manifest.csv
```

That `paper_artifact_manifest.csv` file is the quickest way to see everything a run exported.

## Build A Final Bundle For Sharing

After the runs you want are complete, build the consolidated bundle with:

```bash
python3 scripts/build_final_paper_bundle.py
```

This writes the final shareable outputs to:

```text
paper_summaries/final_bundle/
```

That folder includes:

- `tables/`
- `figures/`
- `paper_artifact_manifest.csv`
- `README.md`

If you already generated the prerequisite summary tables and only want to rebuild the final bundle:

```bash
FINAL_BUNDLE_SKIP_PREREQS=1 python3 scripts/build_final_paper_bundle.py
```

## Research References

This project builds on the following key papers and tools:

- Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R., Gama, J., and Bizarro, P. (2022). "Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation." NeurIPS Datasets and Benchmarks.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A. (2021). "Revisiting Deep Learning Models for Tabular Data." NeurIPS.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS.
- Neftci, E. O., Mostafa, H., and Zenke, F. (2019). "Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient-Based Optimization to Spiking Neural Networks." IEEE Signal Processing Magazine.
- Fang, W., Chen, Y., Ding, J., Yu, Z., Masquelier, T., Chen, D., Huang, L., Zhou, H., Li, G., and Tian, Y. (2023). "SpikingJelly: An Open-Source Machine Learning Infrastructure Platform for Spike-Based Intelligence." Science Advances.
- Saleiro, P., Kuester, B., Hinkson, L., London, J., Stevens, A., Anisfeld, A., Rodolfa, K. T., and Ghani, R. (2018). "Aequitas: A Bias and Fairness Audit Toolkit." arXiv.
- Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. (2019). "Optuna: A Next-Generation Hyperparameter Optimization Framework." KDD.

## Reproducibility Files

For GitHub and archival use, the repo already includes:

- `reproducibility/pip-freeze-2026-02-28.txt`
- `reproducibility/environment-notes.md`

These document the environment used for the recorded runs.
