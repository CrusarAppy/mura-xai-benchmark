# XAI Benchmark for Musculoskeletal Radiographs (MURA)

A **reproducible benchmark framework** for evaluating explainable-AI (XAI) methods on
musculoskeletal radiograph classification (MURA). MSc thesis implementation.

> **Status:** Phase 1 (vertical slice) — DenseNet121 + Grad-CAM + faithfulness/calibration
> metrics + reproducible results logging. Other backbones, XAI methods, metrics, aggregation,
> statistics, and Layer-7 validation are added in later phases.

## Design (target)
- 3 CNN backbones × 6 XAI methods = **18 standardized configurations**.
- 7 evaluation dimensions: performance, calibration, faithfulness, robustness, agreement, efficiency, reproducibility.
- Aggregation (Pareto → TOPSIS → Borda/weighted-sum) + statistics (ANOVA/Friedman, CIs) + benchmark validation.
- Config-driven; runs on free GPUs (Colab/Kaggle); fully reproducible (seeds, pinned deps, Docker).

See `../PROJECT_MEMORY.md` and `../IMPLEMENTATION_PROMPT.md` for the full plan and rules.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # installs the xai_bench package + deps
```

## Configure the dataset (MURA already downloaded)
Edit `configs/densenet_gradcam.yaml` and set `data.mura_root` to your **MURA-v1.1** directory
(the folder that contains `train_image_paths.csv`, `valid_image_paths.csv`, and the `train/` `valid/`
image folders). The loader verifies the structure and errors clearly if paths are wrong.

## Run Phase 1
```bash
python scripts/run_experiment.py --config configs/densenet_gradcam.yaml
```
This trains DenseNet121, generates Grad-CAM maps on a representative subset, computes
performance + calibration + faithfulness (Deletion/Insertion) + runtime, and appends a
reproducible row to `results/results.csv`.

## Tests (no dataset or GPU required)
```bash
pytest -q          # smoke tests on tiny synthetic data
```

## Repository layout
```
XAI_project/
  configs/                  experiment YAML configs
  src/xai_bench/
    data/mura.py            MURA loader, patient-level split, k-fold CV, transforms
    models/backbones.py     backbone registry (DenseNet121 first)
    models/train.py         training loop, seeds, checkpointing
    explainers/gradcam.py   Grad-CAM (Phase 1)
    evaluation/metrics.py   performance + calibration (Accuracy/F1/AUROC, ECE, Brier)
    evaluation/faithfulness.py  Deletion/Insertion, Average Drop, Increase-in-Confidence
    reporting/results_db.py append results to CSV/parquet
    seeds.py utils.py registry.py
  scripts/run_experiment.py
  tests/
```
