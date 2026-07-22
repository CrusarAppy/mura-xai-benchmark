# Implementation Plan v2 — closing the gap between the (strengthened) proposal and the code

**Context.** The revised proposal (`proposal_preview.pdf`, 62 pp.) is now methodologically strong and
commits the framework to specific procedures. The current code (`src/xai_bench/…`) implements only a subset.
This plan maps **every methodological commitment in the proposal → the concrete code change that implements it**,
ordered by dependency and value. Nothing here changes the research design; it makes the code deliver what the
document already promises.

---

## 1. Gap analysis — proposal commitment vs. current code

| Proposal § | Commitment | In code today? | Action |
|---|---|---|---|
| 3.9.1 | Accuracy, Precision, Recall, F1, AUROC | ✅ `metrics.classification_metrics` | keep |
| 3.9.1 | **AUPRC**, **Youden-J** operating point, metrics @0.5 **and** @Youden | ❌ | A1 |
| 3.9.2 | ECE, Brier | ✅ `metrics.calibration_metrics` | keep |
| 3.9.2 | **Reliability-diagram data**, **temperature scaling** | ❌ | A2 |
| 3.9.3 | Deletion, Insertion, Avg-Drop, Increase-in-Confidence | ✅ `faithfulness.py` | keep |
| 3.9.3 | **Dual-baseline** (blur **and** dataset-mean), identical perturbation protocol | ⚠️ single baseline | A3 |
| 3.7 | **IG/SHAP baseline = blur/mean** + **baseline-sensitivity** (zero/blur/mean) | ❌ zero baseline only | A4 |
| 3.9.5 | Runtime | ✅ `pipeline.score_explainer` | keep |
| 3.9.5 | **GPU memory** per explanation | ❌ | A5 |
| 3.9.7 | **Agreement: IoU, Dice (top-k% energy), SSIM, Spearman** (method×method, arch×arch) | ❌ | B1 |
| 3.9.4 | **Robustness**: Gaussian noise (2 σ), brightness/contrast; **sanity checks** (cascading + independent param randomisation, label randomisation) | ❌ | B2 |
| 3.8.1 | **Aggregation**: min–max normalise, Pareto, TOPSIS, weighted-sum, Borda, Kendall’s τ, **weight-sensitivity** | ⚠️ only CIs + a heuristic | C1 |
| 3.10 | **Stats**: Shapiro, ANOVA/Friedman, Nemenyi/Wilcoxon+Holm, effect sizes (partial η², Kendall’s W), **Nadeau–Bengio corrected-resampled t**, bootstrap CIs, **critical-difference diagram** | ⚠️ CIs only | C2 |
| 3.8.2 | Regenerate all tables/figures from results DB | ⚠️ partial | C3 |
| 3.9 / 4.1 | **Per-anatomy** reporting across all 7 regions | ❌ wrist-only | D1 |
| H4 | Controlled-vs-uncontrolled **variance experiment** (Levene/Bartlett) | ❌ | D2 |
| 3.10.6 / 4.2 | **Docker**, pinned deps, **deterministic PyTorch**, env capture | ⚠️ partial | D3 |
| Layer 7 | **Benchmark-validation** module (internal/construct validity, stability, extensibility) | ⚠️ implicit | D4 |

Legend: ✅ done · ⚠️ partial · ❌ missing.

---

## 2. Phased plan

### Phase A — per-configuration metrics (extend the sweep; fastest wins)
These slot straight into `pipeline.score_explainer` / `metrics.py` and add columns to `sweep_results.csv`.
No new experiments required beyond a re-run.

**A1 — Classification: AUPRC + Youden-J.** In `metrics.py`, extend `classification_metrics(probs, labels)` to add
`auprc` (`sklearn.metrics.average_precision_score`), compute the Youden-J threshold
(`thr = argmax(tpr − fpr)` from the ROC), and return `precision/recall/f1` at **both** 0.5 and `thr_youden`
(suffix `_youden`), plus `threshold_youden` itself.
*New columns:* `auprc, precision_youden, recall_youden, f1_youden, threshold_youden`.
*Verify:* unit test on a toy imbalanced array; AUPRC ≤ 1, Youden thr ∈ (0,1).

**A2 — Calibration: reliability data + temperature scaling.** New `evaluation/calibration.py`:
`reliability_curve(probs, labels, n_bins)` → per-bin (confidence, accuracy, count) for the diagram;
`temperature_scale(logits, labels)` → fit scalar T on validation logits (LBFGS on NLL), return T and
post-scaling ECE. Requires the sweep to retain **logits** (add to `evaluate_logits`).
*New columns:* `ece_temp_scaled, temperature`. *Artifact:* per-model reliability CSV for plotting.

**A3 — Dual-baseline faithfulness.** In `faithfulness.py`, add a `dataset_mean` baseline alongside
`_blur_baseline`, and make `deletion_insertion` accept `baseline ∈ {blur, mean, zero}`. In
`pipeline.score_explainer`, run deletion/insertion under **both** blur and mean and emit both.
*New columns:* `deletion_auc_mean, insertion_auc_mean` (existing ones become the blur variant).
*Verify:* both baselines produce finite AUCs; document that cross-family ranking is read across baselines.

**A4 — IG/SHAP baseline choice + sensitivity.** Give `IntegratedGradients` / `GradientSHAP` a `baseline`
kwarg (`zero|blur|mean`, default `blur`). In a dedicated **sensitivity pass** (subset of images), compute each
attribution under all three baselines and record how much the map and its deletion-AUC move
(`baseline_sensitivity_ssim`, `baseline_sensitivity_del_range`). Keep the main sweep on the `blur` baseline.
*Verify:* attribution shape/range unchanged; sensitivity ∈ [0,1].

**A5 — GPU memory.** In `score_explainer`, wrap generation with `torch.cuda.reset_peak_memory_stats()` /
`torch.cuda.max_memory_allocated()` (guard for MPS/CPU → `NaN`). *New column:* `gpu_mem_mb_per_explanation`.

### Phase B — new evaluation dimensions (need saliency maps materialised)
Add a lightweight **maps cache**: `score_explainer` optionally writes each method’s (N,H,W) map to
`maps_cache/{backbone}_{method}_f{fold}_s{seed}.npy` so agreement/robustness can reuse them without
re-generating.

**B1 — Explanation agreement (`evaluation/agreement.py`).**
`binarise_topk(sal, k)` — energy-based: keep smallest pixel set holding top-k% of saliency mass (primary k=20).
`iou(a,b)`, `dice(a,b)` on binarised masks; `ssim_map(a,b)` (skimage) on continuous maps; `spearman_map(a,b)`
on flattened importance. Driver computes, per image: (i) **method×method** agreement for a fixed backbone,
(ii) **architecture×architecture** agreement for a fixed method; average over images; run k ∈ {10,20,30} for the
**sensitivity** the proposal promises. *Output:* `results/agreement_{scope}.csv` (pairwise matrices).
*Verify:* IoU(a,a)=1, Dice(a,a)=1, SSIM(a,a)=1; symmetry.

**B2 — Robustness + sanity checks (`evaluation/robustness.py`).**
*Input robustness:* for each test image, perturb with additive Gaussian noise (σ = 2 levels as a fraction of
dynamic range) and brightness/contrast (bounded ± shift); regenerate the map; score stability as SSIM +
Spearman vs. the clean map, **reported per severity level** (not pooled). *New columns:*
`robust_ssim_noise_lo/hi, robust_spearman_noise_lo/hi, robust_ssim_bc, robust_spearman_bc`.
*Sanity checks (Adebayo):* `cascading_randomization` and `independent_randomization` of layers top→down, plus a
`label_randomization` control; measure map divergence (1−SSIM, 1−|Spearman|) as layers are randomised — a
faithful method should diverge. *Output:* `results/sanity_{backbone}_{method}.csv` (divergence vs. depth).
*Verify:* a passing method shows increasing divergence; a constant map fails the check.

### Phase C — aggregation & statistics (operate on the results DB; pure post-processing)

**C1 — Aggregation (`analysis/aggregation.py`).**
`normalize(df, directions)` min–max with lower-is-better inverted; `pareto_front(df)` non-dominated set;
`topsis(df, weights)` closeness-to-ideal; `weighted_sum(df, weights)`; `borda(df)` rank aggregation;
`kendall_tau_between(rankings)` agreement across the three schemes; `weight_sensitivity(df, weight_grid)` →
rank stability under equal / faithfulness-emphasis / efficiency-emphasis weightings. *Output:*
`results/ranking.csv` + `results/ranking_sensitivity.csv`. *Verify:* Pareto set ⊆ full set; TOPSIS ∈ [0,1];
equal-weight TOPSIS agrees with the earlier heuristic within tolerance.

**C2 — Inferential statistics (`analysis/stats.py`).**
`shapiro_per_metric`; `friedman_across_regions` (regions as independent datasets) + `nemenyi_posthoc` +
`critical_difference_diagram` (matplotlib); `rm_anova` (statsmodels) when normal; `wilcoxon_holm` pairwise;
effect sizes `partial_eta_sq`, `kendalls_w`; **`corrected_resampled_ttest`** (Nadeau–Bengio variance
correction for CV folds); `bootstrap_ci` over test images. *Output:* `results/stats_{hypothesis}.csv` + CD-diagram
PNGs. *Verify:* on synthetic data with a known effect, Friedman p is small and the corrected-t variance exceeds
the naive-t variance (the whole point of the correction).

**C3 — Reporting (`analysis/make_report.py`).** One command reads the results DB and regenerates every table
(3.5 evaluation matrix, ranking, per-anatomy) and figure (reliability diagrams, CD diagram, ranking-sensitivity)
so the thesis artefacts are reproducible from data (proposal §3.8.2).

### Phase D — protocol & infrastructure

**D1 — Per-anatomy sweep.** Extend `run_sweep.py` with a `regions` loop (or `--regions all`); tag every row with
`region`; compute Youden-J and all metrics per region. Compute/GPU note: all-7-regions × 18 configs × 5 folds is
large — recommend running region-by-region as separate Kaggle commits and merging (the existing
`aggregate_cis.py` dedup already supports this).

**D2 — H4 reproducibility experiment (`scripts/run_reproducibility.py`).** Two arms on one (backbone, region):
a **controlled** arm (all nuisance factors fixed) repeated R times, and an **uncontrolled** arm (vary seed, CAM
target layer, augmentation, library) repeated R times; compare between-run metric variance with Levene/Bartlett
and a bootstrap CI on the variance ratio. *Output:* `results/h4_variance.csv`. This is the only new *experiment*;
everything else is measurement or post-processing.

**D3 — Reproducibility infra.** Extend `seeds.py` with deterministic PyTorch
(`torch.use_deterministic_algorithms(True)`, cuDNN deterministic, seeded workers — already partly present);
add `Dockerfile` + pinned `requirements.lock`; add `env_capture.py` (writes Python/CUDA/GPU/lib versions into the
results DB per run).

**D4 — Layer-7 benchmark-validation module (`analysis/validation.py`).** Turns the “benchmark validates itself”
claim into code: *internal validity* — assert identical preprocessing/seeds across configs (hash the config);
*construct validity* — report inter-metric correlations to flag redundant dimensions; *stability* — variance/ICC
of each metric across folds & seeds (feeds H4); *extensibility* — extend `check_repo.py` to assert a new
explainer/metric can register and run end-to-end. *Output:* `results/validation_report.json`.

---

## 3. Suggested order & effort

1. **A1–A5** (per-config metrics) — small, high value, only needs a sweep re-run. ½–1 day each.
2. **B1 agreement**, then **B2 robustness** — new dimensions; need the maps cache first. 1–2 days each.
3. **C1 aggregation**, **C2 stats**, **C3 reporting** — pure post-processing on the CSV; unlocks the thesis
   results chapter. 1–2 days each.
4. **D1 per-anatomy** (compute-heavy; schedule around Kaggle quota), **D2 H4**, **D3 infra**, **D4 validation**.

**Result-schema note:** every Phase A/B item adds columns to `sweep_results.csv`; keep `append_result` tolerant of
new columns (it already concatenates by column union), and bump a `schema_version` field so old and new CSVs
merge cleanly.

**Testing:** add one smoke test per new module under `tests/` (agreement identities, TOPSIS bounds, corrected-t
variance ordering, sanity-check divergence). Extend `check_repo.py` to import every new module from a clean
checkout so nothing silently goes untracked (the bug that bit the Kaggle run earlier).
