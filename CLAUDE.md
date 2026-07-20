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
- Orchestration: `scripts/run_experiment.py`. Config: `configs/densenet_gradcam.yaml`.

## Phase status
- [x] Phase 1: data + DenseNet121 + Grad-CAM + Deletion/Insertion + calibration + results log + tests.
- [x] CAM family complete: Grad-CAM, Grad-CAM++, Score-CAM, LayerCAM (set `explain.method` in the config).
- [ ] Add attribution methods: Integrated Gradients (Captum), SHAP.
- [ ] Wire in the other two backbones (EfficientNet-B0, ConvNeXt-Tiny) via a config sweep.
- [ ] Add robustness (perturbation, sanity checks), agreement (IoU/Dice/SSIM/Spearman), efficiency (GPU mem).
- [ ] Add aggregation (normalize/Pareto/TOPSIS/Borda/sensitivity) and stats (Shapiro/ANOVA/Friedman/CIs).
- [ ] Add Layer-7 benchmark validation checks and full reporting; run all 18 configs.

## Next task
Confirm `data.mura_root`, run `python scripts/run_experiment.py --config configs/densenet_gradcam.yaml`
with `train.quick_debug: true` first to validate the pipeline, then a full run.
