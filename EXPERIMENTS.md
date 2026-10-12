# EXPERIMENTS — Can an English-trained model work on non-English data?

Single tracking file for the cross-lingual study in `multi/`. Owner: primary orchestrator.

Goal (user): show that an English-trained model performs equivalently on non-English data, OR that
the dimensions carrying language information can be identified and ignored when predicting. Either
branch is an acceptable deliverable.

> **RECOVERY NOTE (2026-10-11).** A buggy cleanup command (`Get-ChildItem -Include` is ignored with
> `-LiteralPath`) deleted all files under `multi/`. **Tracked** project files were restored from git;
> **FLORES embeddings** (`embeddings/`, `embeddings_agg/`, `embeddings_610/`, `embeddings_agg_610/`)
> were rebuilt deterministically. This log and `lang_neutralize.py` were rewritten. Still to rebuild:
> `multieurlex/`, `multieurlex_610/`, `translated/`, `translated_emb/`, `sst2_emb/`, `emotion_emb/`,
> `models/`, `europarl_*_embeddings/`.

## Environment
- win32, PowerShell 5.1; python `C:\Users\pedro\anaconda3\python.exe` 3.13.5; torch 2.10+cu128 (RTX 3060 12GB); transformers 5.15.1; xgboost 3.2.0.
- backbone `EuroBERT/EuroBERT-210m` (768) and `EuroBERT/EuroBERT-610m` (1152); **mean pooling** (user directive: ignore bos).
- MT: `Helsinki-NLP/opus-mt-en-{it,de,fr}` and `-{de,fr}-en`.

## Data
- `flores200_en_it_de_fr.csv` (2006 rows; embeddings are 2009 rows → aligned via git ref `196bbd3`); `flores_agg.csv` (562 URL rows, 11 classes).
- FLORES embeddings `embeddings/<mode>/emb_<lang>.npy` (2009,768) and `embeddings_610/...` (2009,1152); agg likewise (562).
- MultiEURLEX (`nlpaueb/multi_eurlex` parquet): en/de/fr long EU legal docs (median ~450–550 words), 21 level-1 EuroVoc concepts (multi-label), 11000/1000/5000 splits.

## Headline conclusion
- On the **original FLORES probe**: language info is **not** a small set of deletable dims (redundant across ~all 768); but it is a small **subspace** (~30 dirs), and removing it does not hurt (often helps) prediction — branch **B** holds. Branch A (equivalence) only partial there (noisy labels).
- On the **real MultiEURLEX long-doc task**: English-only training reaches **~94% of English micro-F1 / ~91% macro zero-shot**, and **~96–99% AUC with deployable chunked-MT ensembling** — branch **A** essentially holds. Winning recipe: EuroBERT-**610m** mean → `l2_thenZ [both]` (L2-normalize + per-language z-score, applied to English *and* target) → 21-OVR XGBoost (EN only) → `quantile_match_enval` calibration.

## Experiment log

### E1/E1b — language-dim census + INLP (`langdims_probe.py`, `lang_inlp.*`) — DONE
- 4-language linear probe = 1.000 on 768 dims; zeroing top-557 η² dims → 0.999 (only all-768 → 0.25). **Language is redundant across nearly all dims.**
- INLP: language undecodable after **~30 directions** (mean); 60–72% variance retained. Language subspace is a **small linear subspace**, not axis-aligned. Top-3 PCA directions are language axes (removed-PC η²≈0.97).

### E2 — translated eval sets (`translate_datasets.*`) — DONE
- MT SST-2/emotion test+val → it/de/fr (opus-mt). Caveat: Marian flips polarity sometimes.

### E3/E6/E7 — FLORES topic cross-lingual transfer (`crosslingual_topic*`, `crosslingual_inlp.*`) — DONE
- mean pooling; EN→it/de/fr full-transfer macro-F1 (sent) 0.234 → 0.334 with `inlp`; removing more language improves transfer. agg (11-class): full 0.246 → 0.356 (zero90/subspace); EN 0.559, in-language ~0.503.
- Language probe collapse via rank-3 centroid `subspace` is **tautological** (equalizes class means) — INLP (~30 dirs) is the faithful estimate (S5a).

### E4 — cross-lingual sentiment/emotion (`crosslingual_sentiment.*`) — DONE
- SST-2 (mean): EN 0.792 macro; cross-lingual raw it/de/fr 0.596/0.593/0.636; best `centroid` 0.630/0.627/0.649. Emotion: EN 0.370 → best `centroid` it/de/fr 0.203/0.214/0.219. Residual gap ~0.15 (sentiment more language-specific + MT noise).

### E8 — Optuna(AUC) × INLP, EN vs IT, sent & agg (`optuna_inlp_compare.*`) — DONE
- Under INLP, transfer AUC ≈ in-language AUC (IT→EN AUC 0.904 sent / 0.948 agg vs own-test 0.896/0.921).

### E9/E9b/E10/E11 — MultiEURLEX real long-doc task (`multieurlex_*`) — DONE
- Baseline (210m, raw): EN self micro/macro 0.702/0.500; EN→de 0.266/0.081, EN→fr 0.334/0.087.
- Transform search: **`l2_thenZ [both]`** + XGB + global threshold → EN→de 0.653/0.467, fr 0.664/0.493; + depth8/mcw5 + `quantile_match_enval` → de 0.688/0.529, fr 0.683/0.532.
- 610m + `l2_thenZ` + `quantile_match` (FINAL): **de 0.711/0.546/AUC 0.872, fr 0.707/0.547/AUC 0.875** vs EN self 0.752/0.601/0.910 → ~94% micro, ~91% macro.
- Test-time ensembling (0.75·EN + 0.25·target over gold parallel test) closes the AUC gap (de 0.9094/fr 0.9112). Deployable **chunked MT** retains 75%/91% of that gain (de 0.8952/fr 0.9075) — gap is closable deployably.
- Linear language projections (INLP/ABTT) add nothing once `l2_thenZ`+depth are matched; residual is calibration/prior-shift.

### E12/E13 — iterative PC-ablation cascade (`pc_ablation*`, `pc_matrix*`) — DONE
- `pc` = GLOBAL shared ablation (PCA once on stacked languages, zero leading PC, inverse, re-fit; k=2). On plain embeddings zeroing top-2 shared PCs lifts MultiEURLEX de 0.375→0.512 / fr 0.412→0.525; on FLORES sent transfer 0.25→0.31 (k=3), agg 0.25→0.37. Per-language PCA (`pc_perlang`) always hurts (shared basis essential). In-language EN unaffected.
- Corrected train-lang × test-lang matrix: with `pc`/`inlp`/`l2_thenZ`, EN-trained reaches ~0.67×(sent)/0.72×(agg)/0.91×(MultiEURLEX) of the in-language ceiling.
- `pc`+`inlp` combos: mild gains in some settings, redundant with `l2_thenZ`.

### E14 — other avenues (`translate_train.*`, `semi_knn.*`, `agg_pr*`) — DONE/partial
- **Translate-train**: FLORES sent FULLY closes/exceeds the in-language ceiling (de 0.455, fr 0.475 vs ceiling 0.436/0.468) and beats all frozen alignment; but **fails on MultiEURLEX** (chunked MT too lossy).
- **kNN retrieval transfer** (label-free): on FLORES sent, closes the gap and **beats the in-language ceiling** (it 0.413/de 0.402/fr 0.436). Self-training barely moves.
- **Threshold/P-R curves** (FLORES agg): EN model precision/recall/F1 vs threshold, English vs Italian and per method (raw/pc/inlp/pc+inlp/l2_thenZ), 210m vs 610m; cross-lingual opera-ting point is precision-limited and needs a **lower threshold (~0.2)** than English (~0.4). `pc+inlp` puts Italian closest to English.
- **E14b — FLORES gap-close sweep (`flores_close.py`) — DONE.** mean-210m, EN-trained → it/de/fr, micro-F1 vs in-language ceiling (sent ceiling ~0.421, agg ~0.433):

| method | sent mean tgt | agg mean tgt | closes gap? |
|---|---:|---:|:--:|
| raw | 0.291 | 0.277 | 0% |
| l2_thenZ | 0.358 | 0.343 | ~50% |
| pc+inlp | 0.371 | 0.310 | ~20–60% |
| translate-train (XGB) | 0.567 | 0.618 | **≥90%** |
| mlp_align | 0.541 | 0.649 | **≥90%** |
| **kNN (l2_thenZ, k=10)** | **0.602** | **0.676** | **≥90% (matches ceiling)** |

  kNN / MLP-alignment **match or beat** the in-language supervised ceiling; with XGBoost specifically,
  **translate-train** closes ~95%/86% (accuracy). Caveat: these use *unlabeled target* / MT.

### E15 — high-recall / highly-imbalanced: EN-trained vs native (`imbalanced_recall.py`) — DONE (sent/agg)
Framing (user): on highly unbalanced rare classes where missing a positive is unacceptable, is an
**English-trained** model on non-English **comparable to a native-language model** when judged
recall-first (tolerating many false alarms)? mean-pool 210m → `l2_thenZ [both]` → OVR XGB.
- Rare positive classes (FLORES `sent`: business 25, health 68, geography 87, safety 97; `agg` rare topics).
- Metrics: Average Precision; **recall @ FPR {1,5,10%}**; **recall @ {5,20,50,100} false alarms**; FPR needed for recall = 0.90.

**Headline (EN-trained ≈ native at the recall-critical end):**
- **FPR required for 90% recall: mean EN 0.543 vs native 0.540 — essentially identical.**
- recall @100 false alarms: EN 0.769 vs native 0.835 (Δ0.066).
- AP within 0.05 of native in 14/24 cells; EN is comparable-or-better on several rare classes (sent `safety` it/fr, agg `health` all targets).
- `agg` `education and society` degenerate (both models collapse to base rate) — flagged.

**Read:** at the operating point that matters (don't miss the positive), an English-trained detector
spends the **same false-alarm budget** as a native model on highly unbalanced non-English text.

**MultiEURLEX arm (`imbalanced_recall_mel`) — BUG FOUND & corrected.** v1 reported EN ≫ native, but
the native de/fr models had been trained with the **English** train labels; MultiEURLEX **train**
splits are NOT parallel across languages (only the test split is), so the native column was noise.
Verified with the target's own labels: in-language macro-F1 en/en 0.556, de/de 0.546, fr/fr 0.545;
`energy` AP de 0.509 (EN) vs **0.546 (native)**, fr 0.448 vs 0.509; `transport` de 0.749 vs **0.826**.
So the honest result is **English-trained ≈ native (comparable, native modestly ahead)** — exactly the
user's goal. Re-run as v2 with the fix.

### E16 — closing the MultiEURLEX gap with 210m + XGBoost (`mel_close.py`) — DONE
mean-pool 210m, 21-OVR XGBoost; every method trains on EN. **English-trained → de/fr** (micro-F1),
plus the true native ceiling (de→de/fr→fr, verified 0.714/0.712 micro, 0.546/0.545 macro):

| method | de micro | fr micro | de/fr macro | AUC |
|---|---:|---:|---:|---:|
| raw | 0.343 | 0.464 | — | 0.72 |
| l2_thenZ | 0.643 | 0.641 | 0.48/0.49 | 0.86 |
| pc / inlp | 0.63 / 0.60 | 0.64 / 0.63 | — | — |
| mlp_align | 0.331 | 0.283 | — | 0.49 |
| knn_k20 | 0.693 | 0.689 | 0.51/0.51 | 0.87 |
| translate-train | 0.716 | 0.707 | 0.54/0.52 | 0.90 |
| **ensemble (translate-train+knn+l2z)** | **0.728** | **0.710** | **0.545/0.531** | **0.903/0.900** |
| EN self | 0.742 | — | — | — |

**Verdict: English-trained ≈ native (gap essentially closed) on MultiEURLEX with 210m + XGBoost**
(de 0.728 vs 0.714, fr 0.710 vs 0.712 micro; macro 0.545/0.531 vs 0.546/0.545). Winning recipe:
**ensemble of translate-train (MT-augmented EN) + kNN transfer + `l2_thenZ`**. `mlp_align` fails on
long docs; pc/inlp give ~0.63.
> NOTE: `mel_close.md`'s own "native ceiling" row is buggy (near-random — it re-used the same
> cross-language train-label assumption); the true native ceiling above comes from the independent
> verification (E15 mel v2). The English-trained numbers are valid.

### E17 — language dims vs their SHAP contribution (210m mean) — DONE
Per-dim eta^2 across languages (Part A) and mean|SHAP| of the EN XGB (Part B), on sent/agg/mel.

| dataset | total eta² | max eta² | k50 | probe full→-k50→-k95 | Spearman(eta²,SHAP) | top-50 overlap | SHAP mass on k50 dims vs random |
|---|---:|---:|---:|---|---:|---:|---|
| sent | 255.8 | 0.774 | 237 | 1.000→1.000→0.998 | -0.076 | 0/50 | 0.283 vs 0.309 (less) |
| agg | 317.1 | 0.837 | 252 | 1.000→1.000→1.000 | -0.059 | 1/50 | 0.267 vs 0.328 (less) |
| mel | 474.4 | 0.950 | 281 | 1.000→1.000→1.000 | -0.019 | 1/50 | 0.366 vs 0.366 (=) |

**Conclusion:** language info is **redundant across ~all 768 dims** (probe survives zeroing 95% of the
eta² mass) — no small "language dims" set. And the English XGBoost model **does not rely on them**
(SHAP mass ≤ random; Spearman(eta²,SHAP) ≤ 0). The model spends its SHAP budget on task-bearing
directions, not language directions. Figures `lang_shap_dims_{sent,agg,mel}.png`.

### E18 — out-of-the-box proof: one English XGBoost (210m mean) works on all languages — DONE
Constraint (user): **no second model, no MT/translate**; only EuroBERT-210m **mean** + XGBoost. Only
allowed ops = model-free feature normalization on the SAME embeddings. Ladder: `raw` → `l2` →
`global_center`/`global_z` (English stats only) → `perlang_center`/`perlang_z` (each language's own
unlabeled train mean/std, same encoder).

Comparable = target/English ≥ 0.9.

| dataset | metric | EN self | raw/global (no target info) | **perlang_z** |
|---|---|---:|---:|---:|
| sent | AUC | 0.849 | ≤0.89 | it .921 / de .926 / fr .917 |
| agg | AUC | 0.888 | ≤0.89 | it .967 / de .949 / fr .940 |
| mel | micro-F1(+qm calib) | 0.718 | .746/.781 | de .932 / fr .928 |
| mel | AUC | 0.906 | .746/.781 | de .943 / fr .960 |

- **NO-target-info configs never reach 0.9** (English-only stats don't transfer).
- **Mild target sample reaches comparability** on every dataset: AUC ≥0.91 (0.92–0.96), mel F1 0.928–0.932 of English.
- FLORES single-label **F1** tops ~0.8 (noisy labels; in-language ceiling ~0.42) — the representation is fine; it's threshold/prior-shift (fix with per-class `quantile_match`). AUC is comparable everywhere.

**Defensible claim:** one English-trained XGBoost on mean-pooled EuroBERT-210m + model-free
per-language standardization (target text, same encoder; no labels, no MT, no second model) performs
comparably across languages — ranking everywhere, and ~93% F1 on the real long-doc task.

### E19 — FLORES F1 residual: ceiling feasibility (`flores_calib.py`, `flores_ceiling.py`) — DONE
Label-free calibration of the single English XGBoost (`perlang_z` features):
- **agg**: `prior_sinkhorn` lifts **micro-F1 to 0.91/0.92/0.99 of English (≥0.9)** (macro ~0.70).
- **sent**: neither micro (~0.77) nor macro (~0.71) reaches 0.9. `temperature` is a no-op for argmax; AUC
  is invariant across all calibrations (so E18 AUC comparability holds); `BBSE` unstable.
In-language ceilings under the identical protocol (train+val→test): sent micro en .606 / it .562 / de
.537 / fr .582; agg en .619 / it .619 / de .604 / fr .532.
**Feasibility:** `0.9 × English-self` is **provably infeasible** for **sent/de** (micro+macro), **sent/it**
(macro), **agg/fr** (micro+macro) — it exceeds those languages' **own ceilings**. EN-transfer reaches
0.71–0.73 (sent) / 0.81–0.85 (agg) of the target's own ceiling; the rest is ceiling/label-noise bound.
FLORES single-label F1 is thus not a representation/transfer failure.

### E20 — external validation on toxicity (toxi-text-3M, clean 80/10/10) — DONE
`toxi-text-3M` (binary toxic, en/it/de/fr/es, pooled+deduped to remove train/test leakage, re-split
80/10/10; committed as `data/toxi_clean*`). One English-trained XGB on 210m **mean** + `perlang_z`;
native model per target; 3 seeds; thresholds tuned on the **val** splits. Models saved to `models_toxi/`.

| target | AUC EN / native | AP EN/native | FPR@R90 EN/native | target/EN AUC |
|---|---:|---:|---:|---:|
| it | 0.649 / 0.834 | 0.447 / 0.695 | 0.799 / 0.476 | 0.72 |
| de | 0.678 / 0.869 | 0.495 / 0.770 | 0.725 / 0.386 | 0.75 |
| fr | 0.776 / 0.866 | 0.663 / 0.782 | 0.581 / 0.364 | 0.86 |
| es | 0.639 / 0.797 | 0.426 / 0.613 | 0.757 / 0.482 | 0.71 |

**Verdict: NOT comparable** on toxicity (target/EN AUC 0.72–0.86; 0/4 ≥ 0.9). English transfer is
clearly below native here. **So the out-of-the-box comparability holds for TOPIC tasks (FLORES,
MultiEURLEX) but NOT for toxicity** — a task-dependent boundary. (A pre-clean run with a smaller,
partly-overlapping split had looked comparable; the clean, properly-split retrain is the defensible
result.) All 15 models + manifest persisted in `models_toxi/`; outputs `toxi_proof_clean.*`.

## Related work
LIR (Ding 2021, arXiv:2109.04727) — SVD language components; improves Amazon-reviews zero-shot.
Libovický 2020 / Zhao 2021 — subtract language means → smaller transfer gap. Xie EMNLP 2022 — low-rank
language-irrelevant subspace. Gonen/Ravfogel — language-identity subspace / INLP. Translate-train &
self-learning/pseudo-labeling (Xu 2021, CLTP 2022) & test-time adaptation (Tent etc.).

## Reproduction
See individual scripts. Core helper: `lang_neutralize.py`. Winning MultiEURLEX pipeline:
`l2_thenZ [both]` → 21-OVR XGB (depth 8, mcw 5) → `quantile_match_enval`; backbone 610m.
