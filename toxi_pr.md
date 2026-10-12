# toxi_pr -- precision / recall 2x2 (EN/IT model x EN/IT test)

Toxicity precision/recall for the English model and the Italian model on the English and Italian test splits (`data/toxi_clean/`), using the persisted `models_toxi/{en,it}_seed{0,1,2}.ubj` XGBoost detectors on **EuroBERT-210m masked-mean** embeddings.  Primary operating point = each model's **own val-tuned threshold** (`models_toxi/manifest.json`, F1-optimal on the model's own-language validation split). Metrics are **mean+/-sd over seeds [0, 1, 2]**.

**Encoding convention (matches `toxi_proof_clean.py`).**  For cell (model M, test language L) the transform is `perlang_z` fit on **L's own train rows** and applied to **L's test rows** -- i.e. the transform language is the *test* language and is independent of which model is scored.  Concretely: `EN model x EN test` and `IT model x EN test` both use `fit_perlang_z(emb_en_train)` applied to `emb_en_test`; `EN model x IT test` and `IT model x IT test` both use `fit_perlang_z(emb_it_train)` applied to `emb_it_test`.  `perlang_z` = L2-normalize then per-language z-score.  Embeddings reused from the `opencode/toxi_clean_cache` cache dir under the system TEMP (re-embedded on GPU only if missing).

**Test-set imbalance (positives).** en test n=3000, n_pos=363 (12.1%); it test n=2310, n_pos=678 (29.4%).

## 2x2 @ val-tuned threshold (mean over seeds)

Rows = model, columns = test language.  Cells read `metric (mean+/-sd)`.

### Precision

| model \ test | EN test | IT test |
|---|---|---|
| EN model | 0.674+/-0.072 | 0.687+/-0.050 |
| IT model | 0.312+/-0.018 | 0.627+/-0.012 |

### Recall

| model \ test | EN test | IT test |
|---|---|---|
| EN model | 0.586+/-0.046 | 0.067+/-0.042 |
| IT model | 0.568+/-0.043 | 0.648+/-0.034 |

### F1

| model \ test | EN test | IT test |
|---|---|---|
| EN model | 0.623+/-0.014 | 0.119+/-0.066 |
| IT model | 0.402+/-0.011 | 0.637+/-0.012 |

### Accuracy

| model \ test | EN test | IT test |
|---|---|---|
| EN model | 0.914+/-0.010 | 0.716+/-0.004 |
| IT model | 0.796+/-0.017 | 0.783+/-0.003 |

### AUC (threshold-independent)

| model \ test | EN test | IT test |
|---|---|---|
| EN model | 0.901+/-0.004 | 0.649+/-0.011 |
| IT model | 0.783+/-0.007 | 0.834+/-0.002 |

### Confusion counts @ val-tuned threshold (TP/FP/FN/TN, mean over seeds)

| cell | TP | FP | FN | TN |
|---|---|---|---|---|
| EN model on EN test | 213 | 107 | 150 | 2530 |
| EN model on IT test | 45 | 23 | 633 | 1609 |
| IT model on EN test | 206 | 457 | 157 | 2180 |
| IT model on IT test | 440 | 263 | 238 | 1369 |

### Val-tuned thresholds used (per seed)

| cell | threshold (mean over seeds) |
|---|---|
| EN model on EN test | 0.131+/-0.046 |
| EN model on IT test | 0.131+/-0.046 |
| IT model on EN test | 0.220+/-0.033 |
| IT model on IT test | 0.220+/-0.033 |

## Precision / Recall at fixed thresholds 0.3 / 0.5 / 0.7 (mean over seeds)

Cells read `precision / recall`.

| cell | thr=0.3 | thr=0.5 | thr=0.7 |
|---|---|---|---|
| EN model on EN test | 0.781+/-0.009 / 0.482+/-0.007 | 0.856+/-0.005 / 0.402+/-0.007 | 0.891+/-0.014 / 0.323+/-0.015 |
| EN model on IT test | 0.761+/-0.047 / 0.025+/-0.009 | 0.808+/-0.124 / 0.009+/-0.004 | 1.000+/-0.000 / 0.004+/-0.001 |
| IT model on EN test | 0.352+/-0.007 / 0.452+/-0.036 | 0.433+/-0.017 / 0.260+/-0.030 | 0.533+/-0.029 / 0.137+/-0.033 |
| IT model on IT test | 0.649+/-0.008 / 0.588+/-0.002 | 0.724+/-0.015 / 0.459+/-0.009 | 0.780+/-0.023 / 0.337+/-0.006 |

### F1 at fixed thresholds 0.3 / 0.5 / 0.7 (mean over seeds)

| cell | thr=0.3 | thr=0.5 | thr=0.7 |
|---|---|---|---|
| EN model on EN test | 0.596+/-0.008 | 0.547+/-0.007 | 0.474+/-0.018 |
| EN model on IT test | 0.048+/-0.016 | 0.018+/-0.007 | 0.009+/-0.003 |
| IT model on EN test | 0.396+/-0.018 | 0.325+/-0.028 | 0.216+/-0.042 |
| IT model on IT test | 0.617+/-0.005 | 0.562+/-0.011 | 0.470+/-0.009 |

## Operating point at recall ~= 0.90 (recall-first; mean over seeds)

Smallest-threshold operating point whose recall first reaches the target; precision at that point.

| cell | recall | precision | F1 | threshold | TP/FP/FN/TN |
|---|---|---|---|---|---|
| EN model on EN test | 0.901+/-0.000 | 0.272+/-0.016 | 0.418+/-0.019 | 0.003+/-0.001 | 327/877/36/1760 |
| EN model on IT test | 0.901+/-0.000 | 0.319+/-0.007 | 0.471+/-0.008 | 0.001+/-0.000 | 611/1304/67/328 |
| IT model on EN test | 0.901+/-0.000 | 0.178+/-0.002 | 0.297+/-0.003 | 0.040+/-0.003 | 327/1511/36/1126 |
| IT model on IT test | 0.901+/-0.000 | 0.440+/-0.012 | 0.592+/-0.010 | 0.020+/-0.002 | 611/777/67/855 |

## Interpretation

- On **Italian** at recall~=0.90, the **EN model** has precision **0.319** vs the **IT model** **0.440** -- the EN model gives up **0.121 precision** for the same recall (cross-lingual cost).
- On **English** at recall~=0.90, the **IT model** has precision **0.178** vs the **EN model** **0.272** (reverse-direction cross-lingual cost 0.094).
