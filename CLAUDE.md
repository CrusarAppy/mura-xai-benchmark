# CLAUDE.md — coding agent context for this repo

Read `../PROJECT_MEMORY.md` for the full project memory and design decisions.

## Rules
- Do NOT hallucinate. Ask before any assumption about the dataset, metric definitions, or protocol.
- MURA is already downloaded — never re-download; read its path from `configs/*.yaml` (`data.mura_root`)
  and verify structure at runtime.
- Config-driven, registry-based, tested, reproducible (seeds, pinned deps, Docker).

## Where things are
- Data/MURA: `src/xai_bench/data/mura.py` (patient-level split — no leakage).
- Backbones: `src/xai_bench/models/backbones.py` (DenseNet121 done; EfficientNet-B0, ConvNeXt-Tiny stubbed-in).
- Explainers: `src/xai_bench/explainers/` (Grad-CAM done).
- Metrics: `src/xai_bench/evaluation/` (performance, calibration, faithfulness done).
- Single run: `scripts/run_experiment.py`. Config: `configs/densenet_gradcam.yaml`.
- Sweep (all 18 configs): `scripts/run_sweep.py`. Config: `configs/sweep.yaml`.
  Shared step: `src/xai_bench/pipeline.py::score_explainer` (explanation + faithfulness + runtime).

## Phase status
- [x] Phase 1: data + DenseNet121 + Grad-CAM + Deletion/Insertion + calibration + results log + tests.
- [x] All six explainers complete: Grad-CAM, Grad-CAM++, Score-CAM, LayerCAM, Integrated Gradients,
      SHAP (GradientSHAP). Set `explain.method` in the config. IG/SHAP are self-contained (pure autograd).
- [x] Sweep runner: 3 backbones (DenseNet121, EfficientNet-B0, ConvNeXt-Tiny) x 6 methods = 18 configs.
      Trains each backbone ONCE per (fold, seed), reuses it for all 6 explainers, appends every row to
      `results/sweep_results.csv`. Checkpoint reuse -> resume-friendly.
- [x] CV + CIs: `run_sweep.py --folds 0,1,2` (batch across Kaggle sessions), then
      `scripts/aggregate_cis.py *.csv` -> mean +/- 95% CI per (backbone, method). Dedups re-runs.
- [x] Repo guard: `scripts/check_repo.py` (all 6 explainers import + no untracked src/*.py).
- [x] Phase A (proposal 3.9.1–3.9.5): AUPRC + Youden-J (`metrics.py`); reliability curve + temperature
      scaling (`calibration.py`, wired into `run_sweep`); dual-baseline faithfulness (blur+mean) and
      GPU-mem per explanation (`pipeline.py`); IG/SHAP baseline = blur (zero/blur/mean selectable) +
      `baseline_sensitivity` diagnostic. New CSV columns: auprc, *_youden, threshold_youden,
      ece_temp_scaled, temperature, deletion_auc_mean, insertion_auc_mean, gpu_mem_mb_per_explanation.
- [ ] Phase B: robustness (perturbation, sanity checks) + agreement (IoU/Dice/SSIM/Spearman) — see IMPLEMENTATION_PLAN_v2.md.
- [ ] Phase C: aggregation (Pareto/TOPSIS/Borda/weight-sensitivity) + stats (Friedman/Nemenyi/Nadeau–Bengio/CD diagram).
- [ ] Phase D: per-anatomy sweep, H4 variance experiment, Docker/determinism, Layer-7 validation.
- [ ] Add robustness (perturbation, sanity checks), agreement (IoU/Dice/SSIM/Spearman), efficiency (GPU mem).
- [ ] Add aggregation (normalize/Pareto/TOPSIS/Borda/sensitivity) and stats (Shapiro/ANOVA/Friedman/CIs).
- [ ] Add Layer-7 benchmark validation checks and full reporting; run all 18 configs.

## Next task
Confirm `data.mura_root`, run `python scripts/run_experiment.py --config configs/densenet_gradcam.yaml`
with `train.quick_debug: true` first to validate the pipeline, then a full run.
