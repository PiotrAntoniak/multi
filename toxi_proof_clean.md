# toxi_proof_clean -- ONE English-trained XGBoost (EuroBERT-210m mean) on the leakage-corrected toxi-text-3M (`data/toxi_clean/`)

Out-of-the-box cross-lingual validation on the **versioned, leakage-corrected** dataset `data/toxi_clean/` (pooled + globally deduped + 80/10/10 re-split, **0 cross-split overlap**).  A single **English-trained** `XGBClassifier` on **EuroBERT-210m masked-mean** embeddings (no MT, no second model) vs a **native** per-language model, evaluated recall-first.  All metrics are **mean+/-sd over seeds [0, 1, 2]**.

Recipe: EuroBERT-210m **mean** pool (masked mean, plain forward, max_length 512, batch 64, **fp32**, GPU) -> `perlang_z` (L2-normalize then per-language z-score; target stats from the target's OWN train rows) -> `XGBClassifier(binary:logistic, hist, max_bin=64, device=cpu, n_jobs=2, max_depth=8, min_child_weight=5, lr=0.1, n_estimators=400)`.  F1-optimal threshold tuned on the **validation** split (EN val for the EN model; target val for the native model).

> **Data.**  `data/toxi_clean/<lang>_{train,validation,test}.parquet`; the leakage fix is baked into the committed data, so this run does NO in-script sampling/dedup.  Test sizes are the full per-language test splits (en/it/de/fr/es = en:3000, it:2310, de:1230, fr:2627, es:2919).

> **Seed note.**  The task-specified XGBoost config has no row/column subsampling and is therefore seed-DETERMINISTIC (verified: identical predictions for random_state 0/1/2).  To obtain a genuinely non-trivial mean+/-sd (so the verdict is not single-seed) each seed trains on a seed-dependent **bootstrap resample** (size=n_train, with replacement) of that model's train rows.  Validation/test rows and every hyperparameter are unchanged; thresholds are still tuned on the fixed validation split.

> **Trained models (fresh).**  Every EN and native model was **trained fresh in this run** on the `data/toxi_clean/` train parquet (threshold tuned on the validation parquet) and **persisted** with XGBoost `save_model` (`ubj`) under `models_toxi/` (15 models, `models_toxi/manifest.json` records lang/split sizes, hyperparameters, seed, val-tuned threshold, and val+test metrics).  No previously saved/leaked model was loaded or reused.

## Per-split positive rates (imbalance)

| lang | train n (pos%) | val n (pos%) | test n (pos%) |
|---|---|---|---|
| en | 24000 (0.117) | 3000 (0.112) | 3000 (0.121) |
| it | 18472 (0.283) | 2310 (0.275) | 2310 (0.294) |
| de | 9836 (0.299) | 1230 (0.302) | 1230 (0.310) |
| fr | 21011 (0.365) | 2627 (0.354) | 2627 (0.369) |
| es | 23349 (0.313) | 2919 (0.318) | 2919 (0.313) |

EN reference (EN-trained on EN test, mean over seeds): **AUC=0.901+/-0.004**, AP=0.689+/-0.007, n_pos/n_neg=363/2637.

## Per-target: EN-trained vs native (test, mean over seeds)

Cells read `EN / native` as `mean+/-sd`.  Lower is better only for `FPR@R90`.

| metric | it | de | fr | es |
|---|---|---|---|---|
| imbalance (pos rate) | 0.294+/-0.000 / 0.294+/-0.000 | 0.310+/-0.000 / 0.310+/-0.000 | 0.369+/-0.000 / 0.369+/-0.000 | 0.313+/-0.000 / 0.313+/-0.000 |
| AUC | 0.649+/-0.011 / 0.834+/-0.002 | 0.678+/-0.013 / 0.869+/-0.002 | 0.776+/-0.008 / 0.866+/-0.000 | 0.639+/-0.005 / 0.797+/-0.006 |
| AP (PR-AUC) | 0.447+/-0.011 / 0.695+/-0.010 | 0.495+/-0.025 / 0.770+/-0.001 | 0.663+/-0.006 / 0.782+/-0.003 | 0.426+/-0.008 / 0.613+/-0.006 |
| accuracy | 0.716+/-0.004 / 0.783+/-0.003 | 0.705+/-0.009 / 0.810+/-0.007 | 0.668+/-0.016 / 0.767+/-0.008 | 0.688+/-0.005 / 0.711+/-0.007 |
| F1 @ val-thr | 0.119+/-0.066 / 0.637+/-0.012 | 0.188+/-0.081 / 0.690+/-0.008 | 0.223+/-0.081 / 0.724+/-0.003 | 0.135+/-0.032 / 0.620+/-0.010 |
| R@FPR1% | 0.046+/-0.008 / 0.181+/-0.019 | 0.069+/-0.020 / 0.258+/-0.027 | 0.083+/-0.005 / 0.165+/-0.017 | 0.029+/-0.010 / 0.080+/-0.011 |
| R@FPR5% | 0.155+/-0.013 / 0.372+/-0.025 | 0.172+/-0.020 / 0.458+/-0.014 | 0.263+/-0.028 / 0.423+/-0.020 | 0.113+/-0.012 / 0.243+/-0.010 |
| R@FPR10% | 0.243+/-0.008 / 0.528+/-0.007 | 0.260+/-0.023 / 0.612+/-0.019 | 0.403+/-0.017 / 0.581+/-0.009 | 0.192+/-0.009 / 0.386+/-0.010 |
| R@FA5 | 0.027+/-0.008 / 0.102+/-0.021 | 0.056+/-0.013 / 0.208+/-0.022 | 0.045+/-0.005 / 0.078+/-0.016 | 0.008+/-0.006 / 0.028+/-0.004 |
| R@FA20 | 0.054+/-0.005 / 0.200+/-0.023 | 0.114+/-0.031 / 0.350+/-0.013 | 0.090+/-0.007 / 0.190+/-0.032 | 0.029+/-0.010 / 0.080+/-0.011 |
| R@FA50 | 0.112+/-0.006 / 0.304+/-0.020 | 0.189+/-0.012 / 0.500+/-0.017 | 0.197+/-0.019 / 0.346+/-0.008 | 0.062+/-0.017 / 0.152+/-0.009 |
| R@FA100 | 0.181+/-0.013 / 0.428+/-0.018 | 0.277+/-0.028 / 0.658+/-0.013 | 0.299+/-0.025 / 0.460+/-0.013 | 0.113+/-0.012 / 0.243+/-0.010 |
| FPR@R90 | 0.799+/-0.027 / 0.476+/-0.022 | 0.725+/-0.043 / 0.386+/-0.023 | 0.581+/-0.027 / 0.364+/-0.004 | 0.757+/-0.016 / 0.482+/-0.019 |

## Ratios / comparability (mean over seeds)

`target/EN` = AUC(EN-trained on target test) / AUC(EN-trained on EN test); bar >= 0.9. `EN/native` = AUC(EN-trained on target) / AUC(native on target); >= 0.9 means the single English model matches the native one.

| target | test n_pos/n_neg | target/EN AUC | EN/native AUC | FPR@R90 EN/native | R@FA100 EN/native |
|---|---|---|---|---|---|
| it | 678/1632 | 0.720+/-0.010 | 0.778+/-0.015 | 0.799+/-0.027 / 0.476+/-0.022 | 0.181+/-0.013 / 0.428+/-0.018 |
| de | 381/849 | 0.752+/-0.016 | 0.780+/-0.016 | 0.725+/-0.043 / 0.386+/-0.023 | 0.277+/-0.028 / 0.658+/-0.013 |
| fr | 969/1658 | 0.861+/-0.013 | 0.896+/-0.010 | 0.581+/-0.027 / 0.364+/-0.004 | 0.299+/-0.025 / 0.460+/-0.013 |
| es | 913/2006 | 0.709+/-0.009 | 0.802+/-0.009 | 0.757+/-0.016 / 0.482+/-0.019 | 0.113+/-0.012 / 0.243+/-0.010 |

## Verdict

- Target/EN AUC ratio >= 0.9 in **0/4** targets (ratios: it=0.720, de=0.752, fr=0.861, es=0.709).
- EN/native AUC ratio >= 0.9 in **0/4** targets (ratios: it=0.778, de=0.780, fr=0.896, es=0.802).
- Mean AP: EN-trained **0.508** vs native **0.715**; mean FPR@R90: EN-trained **0.716** vs native **0.427** (smaller is better).
- **Verdict: NOT comparable** -- the English-trained model is materially below native on the recall-first view for multiple targets.
