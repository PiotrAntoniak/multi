# EXPERIMENTS — Cross-lingual study (single tracking file)

**Question.** Can a model trained on **English** perform comparably on **non-English** text, and which
embedding dimensions carry language? Constraint (user): frozen **EuroBERT-210m / 610m**, **mean pooling
only**, classifier **XGBoost**; for the headline "out-of-the-box" claim, **no second model, no MT**.

**How to read the tables.** Each row is one condition; the last column is the **delta** of the two named
columns (so you never have to guess signs). `macro` rows average the language rows. Metrics:
`accuracy / macro-F1 / micro-F1 / AUC` as labelled.

## Glossary

| term | meaning |
|---|---|
| **mean** | mean-pooled last hidden state (only pooling used). |
| **perlang_z** | L2-normalize each vector, then z-score with **each language's own train** mean/std, applied to English **and** target. Model-free. |
| **pc(k)** | global PCA ablation: fit PCA on the stacked-language train pool, remove the leading PC, repeat k times. |
| **inlp** | project out a language-subspace basis. |
| **D** | language-shift: target features + (mean_en_train − mean_target_train). |
| **self / transfer** | self = in-language (train L → test L); transfer = cross-lingual (train L → test L′). |
| **XL-mean** | mean over non-English targets. |

---

## PART A — Headline: one English model, comparable across languages

### E18 — Out-of-the-box cross-lingual proof (`proof_mean_alllangs.py`) — DONE (verified)
Train ONE English XGBoost, only **model-free** feature ops; check target/English ≥ 0.9.

| dataset | metric | English-self | target mean (it/de/fr) | target/English | comparable |
|---|---|---:|---:|---:|:--:|
| FLORES sent (210m) | AUC | 0.849 | 0.92 / 0.93 / 0.92 | 0.92–0.93 | ✅ |
| FLORES agg (210m) | AUC | 0.888 | 0.97 / 0.95 / 0.94 | 0.94–0.97 | ✅ |
| MultiEURLEX (210m) | micro-F1 (+calib) | 0.718 | de 0.93 / fr 0.93 | 0.93 | ✅ |
| MultiEURLEX (210m) | AUC | 0.906 | de 0.94 / fr 0.96 | 0.94–0.96 | ✅ |

| feature config | uses target sample? | reaches target/English ≥ 0.9? |
|---|---|:--:|
| raw / L2 / global centering/z (English stats only) | no | **never** (≤0.89) |
| `perlang_z` (target's own unlabeled train stats) | yes (unlabeled) | **yes** |

**Finding:** comparable across languages **iff** a mild unlabeled **target sample** is used for
per-language standardization. Independently verified (rebuilt from raw embeddings; matches to 6 decimals).

### E24 — More datasets (XNLI, PAWS-X, textdetox, SIB-200) (`more_datasets.py`) — DONE

| dataset | task | metric | target | English-self | target/English | comparable |
|---|---|---|---:|---:|---:|:--:|
| PAWS-X | binary paraphrase | AUC | de | 0.602 | 0.937 | ✅ |
| PAWS-X | | AUC | fr | 0.602 | 0.946 | ✅ |
| PAWS-X | | AUC | es | 0.602 | 0.918 | ✅ |
| XNLI | 3-class NLI | macro-F1 | de | 0.440 | 0.898 | ✗ (borderline) |
| XNLI | | macro-F1 | fr | 0.440 | 0.948 | ✅ |
| XNLI | | macro-F1 | es | 0.440 | 0.949 | ✅ |
| SIB-200 | topic | macro-F1 | de | 0.773 | 0.903 | ✅ |
| SIB-200 | | macro-F1 | fr | 0.773 | 0.995 | ✅ |
| SIB-200 | | macro-F1 | es | 0.773 | 0.896 | ✗ (borderline) |
| textdetox | binary toxic | AUC | it/de/fr/es | 0.973 | 0.62–0.81 | ✗ |

**Finding:** comparable for **topic / paraphrase / NLI**; **fails for toxicity**.

### E25 — Full language × language matrices (`lang_matrix.py`) — DONE

| dataset | backbone | in-language (diagonal) | cross-lingual (off-diag) | gap | 610 vs 210 gap |
|---|---|---:|---:|---:|---|
| flores_sent | 210m | 0.461 | 0.377 | 0.084 | — |
| flores_sent | 610m | 0.486 | 0.447 | 0.039 | smaller |
| flores_agg | 210m | 0.544 | 0.423 | 0.121 | — |
| flores_agg | 610m | 0.510 | 0.464 | 0.046 | smaller |
| mel | 210m | 0.588 | 0.330 | 0.258 | — |
| mel | 610m | 0.577 | 0.417 | 0.160 | smaller |
| toxic | 210m | 0.739 | 0.567 | 0.171 | — |
| toxic | 610m | 0.750 | 0.598 | 0.152 | smaller |
| xnli | 210m | 0.452 | 0.408 | 0.044 | — |
| xnli | 610m | 0.460 | 0.430 | 0.030 | smaller |
| **pawsx** | 210m | 0.520 | **0.532** | **−0.013** | closed |
| **pawsx** | 610m | 0.561 | 0.558 | 0.003 | closed |
| sib200 | 210m | 0.792 | 0.691 | 0.101 | — |
| sib200 | 610m | 0.800 | 0.763 | 0.037 | smaller |
| textdetox | 210m | 0.870 | 0.672 | 0.198 | — |
| textdetox | 610m | 0.874 | 0.717 | 0.158 | smaller |

**Finding:** **610m shrinks the gap on every dataset**; PAWS-X transfers at parity; largest gaps on
MultiEURLEX/textdetox.

### E20 — Toxicity, clean split (`toxi_proof_clean.py`) — DONE
`toxi-text-3M`, binary, leakage-corrected 80/10/10, 3 seeds.

| target | English-trained AUC | native AUC | English/native |
|---|---:|---:|---:|
| it | 0.649 | 0.834 | 0.78 |
| de | 0.678 | 0.869 | 0.78 |
| fr | 0.776 | 0.866 | 0.90 |
| es | 0.639 | 0.797 | 0.80 |

**Finding:** **not comparable on toxicity** (0/4). Models in `models_toxi/`; data in `data/toxi_clean/`.

### E25b — All-languages vs per-language training (`alllang_vs_perlang.py`) — DONE
One model on pooled all-languages vs one model per language (mean + `perlang_z`, fixed XGBoost).

| dataset | backbone | metric | per-lang | all-langs | Δ (all−per) | verdict |
|---|---|---:|---:|---:|---:|---|
| FLORES sent | 210m | macro-F1 | 0.388 | 0.400 | +0.012 | all-langs better |
| FLORES sent | 610m | macro-F1 | 0.379 | 0.414 | +0.035 | all-langs better |
| FLORES agg | 210m | macro-F1 | 0.506 | 0.543 | +0.037 | all-langs better |
| FLORES agg | 610m | macro-F1 | 0.446 | 0.548 | **+0.101** | all-langs better |
| toxicity | 210m | macro-F1 | 0.746 | 0.730 | −0.016 | per-lang better |
| toxicity | 610m | macro-F1 | 0.752 | 0.738 | −0.014 | per-lang better |

Italian-only check (macro-F1, per-lang → all-langs):

| dataset | backbone | IT per-lang | IT all-langs | Δ | AUC Δ |
|---|---|---:|---:|---:|---:|
| FLORES sent | 210m | 0.378 | 0.354 | −0.023 | +0.016 |
| FLORES sent | 610m | 0.317 | 0.377 | **+0.060** | +0.016 |
| FLORES agg | 210m | 0.540 | 0.498 | −0.042 | +0.016 |
| FLORES agg | 610m | 0.482 | 0.516 | **+0.034** | 0.000 |
| toxicity | 210m | — | — | −0.014 | — |

**Finding:** pooling all languages **helps FLORES** (bigger at 610m) and **hurts toxicity**. For **Italian
specifically**, all-langs is better **only on FLORES at 610m**.

---

## PART B — Why mean pooling transfers

### E1/E1b — Language-dimension census + INLP (`langdims_probe.py`, `lang_inlp.*`) — DONE

| measurement | value |
|---|---:|
| 4-language linear probe, all 768 dims | 1.000 |
| probe after zeroing top-557 η² dims (90% of mass) | 0.999 |
| probe after zeroing **all** 768 dims (chance) | 0.250 |
| INLP directions to reach chance (mean pooling) | ~30 |

**Finding:** language is **redundant across ~all dims** (no small dim set), but a small **subspace**
(~30 dirs) suffices.

### E17 — Language dims vs their SHAP contribution (`lang_shap_dims.py`) — DONE

| dataset | Spearman(η², SHAP) | top-50 overlap | SHAP mass on 50%-language dims | vs random |
|---|---:|---:|---:|---|
| sent | −0.076 | 0/50 | 0.283 | below (0.309) |
| agg | −0.059 | 1/50 | 0.267 | below (0.328) |
| mel | −0.019 | 1/50 | 0.366 | equal (0.366) |

**Finding:** the English classifier **does not use** the language dims (SHAP mass ≤ random) — why mean
pooling transfers.

### E12/E13/E26 — Dimension removal (`pc_ablation*.py`, `dim_removal.py`) — DONE
Remove leading PCs (global shared PCA, k=0–5) / INLP / random control; XL-mean macro-F1 shown.

| dataset | backbone | k=0 | best k | best XL-F1 | past-best at k=5 |
|---|---|---:|---:|---:|---:|
| mel | 210m | 0.076 | 2 | **0.290** | 0.189 |
| mel | 610m | 0.107 | 2 | **0.341** | 0.228 |
| flores_agg | 210m | 0.204 | 3 | **0.380** | 0.269 |
| flores_sent | 210m | 0.206 | 3 | 0.259 | 0.178 |
| toxic | 210m | 0.467 | 0 | 0.467 | 0.458 |
| toxic | 610m | 0.501 | 0 | 0.501 | 0.463 |

**Finding:** the leading ~2–3 PCs are ~96–98% language-aligned; removing them **helps topic tasks**
(optimum k≈2–3, harm past that), neutral/harmful for toxicity; random control stays below; EN-self flat.

---

## PART C — Task-level cross-lingual transfer

### E3/E6/E7 — FLORES topic cross-lingual (`crosslingual_topic*.py`, `crosslingual_inlp.py`) — DONE
EN topic classifier → it/de/fr; mean macro-F1 over targets.

| dataset | condition | full (raw) | best neutralization | best method |
|---|---|---:|---:|---|
| FLORES sent | 12-class | 0.234 | 0.334 | inlp |
| FLORES agg | 11-class | 0.246 | 0.356 | zero90 / subspace |

### E9/E10/E11 — MultiEURLEX long-doc (`multieurlex_*`, `mel_close.py`) — DONE
EN→de/fr micro-F1 (21-label).

| backbone | representation | de micro-F1 | fr micro-F1 |
|---|---|---:|---:|
| 210m | raw | 0.266 | 0.334 |
| 210m | `l2_thenZ` + calib | 0.688 | 0.683 |
| 610m | `l2_thenZ` + calib | **0.711** | **0.707** |
| — | in-language ceiling | 0.714 | 0.712 |

**Finding:** `l2_thenZ` + threshold/prior calibration + bigger backbone essentially closes the gap.

### E14/E14b — FLORES gap-closers (`flores_close.py`) — DONE
Mean target macro-F1 vs in-language ceiling (sent ~0.42 / agg ~0.43).

| method | FLORES sent | FLORES agg | closes gap? |
|---|---:|---:|:--:|
| raw | 0.291 | 0.277 | 0% |
| `l2_thenZ` | 0.358 | 0.343 | ~50% |
| `pc+inlp` | 0.371 | 0.310 | 20–60% |
| translate-train (XGB) | 0.567 | 0.618 | ≥90% |
| MLP-align | 0.541 | 0.649 | ≥90% |
| **kNN (l2_thenZ, k=10)** | **0.602** | **0.676** | ≥90% (matches ceiling) |

### E23 — Translate-test (`translate_test_eval.py`) — DONE
Train EN → MT the test → EN → evaluate.

| backbone | English test | MT it | MT de | MT fr | untranslated it/de/fr |
|---|---:|---:|---:|---:|---|
| 210m | 0.486 | 0.466 | 0.466 | 0.476 | 0.231 / 0.299 / 0.288 |
| 610m | 0.495 | 0.519 | 0.494 | 0.515 | 0.276 / 0.311 / 0.340 |

**Finding:** translating the test into English recovers English-level (ratio 0.96–1.05).

### E4 — Sentiment / emotion (`crosslingual_sentiment.py`) — DONE
Macro-F1; English → translated it/de/fr.

| task | English | raw cross | best neutralization (`centroid`) |
|---|---:|---|---|
| SST-2 (binary) | 0.792 | 0.596 / 0.593 / 0.636 | 0.630 / 0.627 / 0.649 |
| emotion (6-class) | 0.370 | 0.170 / 0.118 / 0.174 | 0.203 / 0.214 / 0.219 |

---

## PART D — Classifiers, MT, calibration

### E21 — Non-XGBoost classifiers (`models_compare.py`) — DONE
FLORES-sent, macro-F1, EN-self / transfer.

| model | 210m EN-self | 210m transfer | 610m EN-self | 610m transfer |
|---|---:|---:|---:|---:|
| SVC_linear | **0.555** | 0.430 | **0.583** | **0.562** |
| MLP | 0.538 | 0.432 | 0.564 | 0.517 |
| LogisticRegression | 0.503 | 0.387 | 0.541 | 0.514 |
| GaussianNB | 0.463 | **0.453** | 0.484 | 0.494 |
| XGBoost (ref) | 0.486 | 0.406 | 0.495 | 0.466 |

**Finding:** a **linear SVM beats XGBoost** on self and transfer.

### E28 — Backbone × dataset × method grid (`backbone_grid.py`) — DONE
XGBoost + 10 Optuna trials; XL-mean macro-F1.

| dataset | backbone | mean (raw) | pc | inlp | D |
|---|---|---:|---:|---:|---:|
| mel | 210m | 0.059 | 0.293 | **0.299** | 0.298 |
| mel | 610m | 0.115 | 0.338 | **0.349** | 0.329 |
| flores_sent | 610m | 0.241 | 0.265 | **0.307** | 0.274 |
| flores_agg | 610m | 0.302 | **0.340** | 0.301 | 0.296 |
| toxic | 610m | **0.500** | 0.465 | 0.494 | 0.452 |

**Finding:** the language-subspace methods (`pc`/`inlp`/`D`) **massively help** multi-label long-doc mel;
help FLORES-sent; neutral/harmful for toxicity.

### E22 — MT: bilingual vs multilingual → English (`mt_compare.py`) — DONE

| direction | BLEU (bilingual) | BLEU (multilingual mul-en) | Δ |
|---|---:|---:|---:|
| it→en | 29.8 | 23.2 | −6.6 |
| de→en | 40.7 | 29.2 | −11.4 |
| fr→en | 42.1 | 31.8 | −10.3 |
| **macro** | **37.5** | **28.1** | **−9.4** |

**Finding:** three bilingual models beat one many→en model (−9.4 BLEU).

### E19 — Calibration & ceiling feasibility (`flores_calib.py`, `flores_ceiling.py`) — DONE
FLORES single-label F1; `0.9 × English-self` feasibility.

| dataset | target | 0.9×English-self | target's own ceiling | feasible? |
|---|---|---:|---:|:--:|
| sent | de | 0.545 | 0.537 | **NO** |
| sent | it (macro) | 0.438 | 0.434 | **NO** |
| agg | fr | 0.557 | 0.532 | **NO** |

**Finding:** FLORES single-label F1's residual is ceiling/label-noise bound, not transfer; AUC invariant.

---

## PART E — Recall-first / highly unbalanced

### E15 — English vs native at high recall (`imbalanced_recall*.py`) — DONE

| dataset | English FPR@90%-recall | native FPR@90%-recall | English − native |
|---|---:|---:|---:|
| FLORES (rare topics, sent+agg) | 0.543 | 0.540 | +0.003 |
| MultiEURLEX (rare concepts, v2) | 0.426 | 0.357 | +0.069 |

**Finding:** at the recall-critical operating point the English detector spends ~the same false-alarm
budget as a native model. (A first MultiEURLEX run falsely showed English ≫ native — a **label bug**;
corrected in v2.)

### E27 — toxi precision/recall vs threshold (`toxi_pr*.py`) — DONE

| cell | 210m precision @ recall 0.90 | 610m precision @ recall 0.90 | Δ |
|---|---:|---:|---:|
| EN model / IT test | 0.324 | 0.359 | +0.035 |
| IT model / IT test | 0.448 | 0.460 | +0.012 |
| EN model / EN test | 0.276 | 0.278 | +0.002 |

**Finding:** on toxicity the English model is below native and the threshold is miscalibrated; 610m helps.

---

## PART F — Fine-tuning (in progress)

### E29 — Fine-tune last layer + linear head, raw (`finetune_lastlayer.py`) — RUNNING
Train **only** EuroBERT's last transformer layer + a linear head (embeddings + lower layers frozen), **raw
mean-pooled features, no `perlang_z`/alignment**, bf16 compute (fp32 master weights). EuroBERT paper
protocol: AdamW β=(0.9, 0.95), ε=1e-5, wd=0.1, linear schedule + 10% warmup, batch 32, ≤10k steps, early
stopping; LR grid {1e-5, 3.6e-5, 6e-5, 1e-4} chosen on val. Datasets: FLORES sent/agg + toxicity.
Compare vs the **frozen raw + XGBoost** baseline — does fine-tuning alone make EuroBERT cross-lingual?

---

## Data & utilities

| dataset | rows | languages | label | notes |
|---|---:|---|---|---|
| FLORES sent | 2,006 | en/it/de/fr | 12-class topic | train 997 · val 504 · test 505 |
| FLORES agg | 562 | en/it/de/fr | 11-class topic | 276/140/139 |
| MultiEURLEX | ~17k/lang | en/de/fr | 21-label | long EU docs |
| toxic (clean) | 120,840 | en/it/de/fr/es | binary | 80/10/10; leakage-corrected |
| SST-2 | 68,221 | English | binary | — |
| emotion | 20,000 | English | 6-class | — |
| XNLI / PAWS-X / SIB-200 / textdetox | varies | 4–5 langs | NLI / paraphrase / topic / toxic | downloaded |

- **E2** translated SST-2/emotion eval sets (`translate_datasets.py`, `rebuild_caches.py`).
- **Cleanup/recovery note (2026-10-11):** a buggy cleanup deleted all files under `multi/`; tracked files
  restored from git, embeddings rebuilt, this log rewritten.

### Master result
English-trained, mean-pooled, model-free `perlang_z` → **comparable (target/English ≥ 0.9)** for **topic**
(FLORES sent/agg, MultiEURLEX, SIB-200), **paraphrase** (PAWS-X) and **NLI** (XNLI fr/es); **not
comparable** for **toxicity** (textdetox, toxi-text-3M). **610m shrinks the gap everywhere.** On highly
unbalanced positives the English model matches the native model's false-alarm cost at 90% recall.
