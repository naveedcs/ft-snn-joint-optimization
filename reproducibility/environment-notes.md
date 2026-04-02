# Reproducibility Environment Notes

This repository now includes an exact Python package lock captured from the environment used for the current experiment runs:

- `reproducibility/pip-freeze-2026-02-28.txt`

Capture metadata:

- Capture time (UTC): `2026-02-28T23:31:02Z`
- Repository commit at capture: `ae6fd7c9db68a6735eb31e83ee6067bab446b34f`
- Python executable: `/usr/bin/python`
- Python version: `3.10.12`
- `pip` version: `22.0.2`
- OS / kernel: `Linux 5.15.0-170-generic x86_64 GNU/Linux`
- CUDA toolkit on `PATH`: `Build cuda_12.2.r12.2/compiler.33053471_0`

Core package versions used by the experiment notebooks:

- `torch==2.10.0`
- `xformers==0.0.35`
- `spikingjelly==0.0.0.0.14`
- `lightgbm==4.5.0`
- `optuna==4.0.0`
- `numpy==1.26.3`
- `pandas==2.3.3`
- `scikit-learn==1.5.2`
- `aequitas==1.1.0`
- `matplotlib==3.10.8`
- `seaborn==0.13.2`
- `tensorflow==2.17.0`

Notebook-to-stack mapping:

- `hyper_optimized_ft_snn_100_experiments.ipynb`: `torch`, `spikingjelly`, `xformers`, `optuna`, `aequitas`
- `hyper_optimized_ft_dense_100_experiments.ipynb`: `torch`, `optuna`, `aequitas`
- `hyper_optimized_lightgbm_100_experiments.ipynb`: `lightgbm`, `optuna`, `scikit-learn`, `aequitas`

Known environment caveats:

- The exact package lock is the authoritative dependency record for the current runs.
- The installed `lightgbm==4.5.0` in this environment supports CPU execution, but it does not currently expose the CUDA tree learner. GPU LightGBM reruns require rebuilding LightGBM with CUDA enabled (`-DUSE_CUDA=1`).
- The LightGBM notebook includes a local import guard around optional `dask` imports because the system `dask` package and `pandas==2.3.3` are not cleanly aligned in this environment.

Recommended archival set for paper reproducibility:

- This repository snapshot
- `reproducibility/pip-freeze-2026-02-28.txt`
- `reproducibility/environment-notes.md`
- The per-run `paper_artifact_manifest.csv` files written into each experiment output directory
