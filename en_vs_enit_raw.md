### E42 -- RAW-feature EN vs EN+IT transfer, FLORES sentence -- DONE

**Question.** With only raw mean-pooled EuroBERT-210m features (no MT, no L2, no z-score, no per-language z, no PCA/INLP), how much of an in-language Italian model's test AUC does an English-only model keep?

**Script to reproduce:** `en_vs_enit_raw.py` -- `python en_vs_enit_raw.py --smoke` then `python en_vs_enit_raw.py --launch`; arm C: `python en_vs_enit_raw.py --add-c`. Reproduction status: REPRODUCIBLE (single detached run, seed 0).

**Data.** `flores200_en_it_de_fr.csv`, FLORES sentence only (no `flores_agg`). Train 997 / val 504 / test 505, 12 classes, seed 0. Embeddings `embeddings/mean/emb_{en,it}.npy` (2009x768 float32, sliced to the 2006 kept rows). Features are **raw**: no transform of any kind.

**Protocol.** XGBoost `multi:softprob`, `num_class=12`, `hist`, `max_bin=64`, `device=cpu`, `n_jobs=2`, seed 0. 4 independent Optuna TPE studies (seed 0), **10 trials each**, objective = **validation OVR-macro AUC** (maximise). Search: max_depth 3-10, lr 0.03-0.3 (log), n_estimators 100-800, min_child_weight 1-10, subsample 0.6-1.0, colsample_bytree 0.4-1.0, reg_lambda 0-5.

| arm | train rows | tuned on | best val AUC | test split |
|---|---:|---|---:|---|
| A | 997 (EN) | EN val | 0.8881 | IT test (primary), EN test (self) |
| B | 1994 (EN+IT) | IT val | 0.8572 | IT test |
| C | 997 (IT) | IT val | 0.8440 | IT test (control) |

**Results (test, best params refit on that arm's train rows).**

| cell | accuracy | micro-F1 | macro-F1 | OVR-macro AUC |
|---|---:|---:|---:|---:|
| A: EN train -> EN test (self ref) | 0.5446 | 0.5446 | 0.4149 | **0.8739** |
| A: EN train -> IT test (primary) | 0.3901 | 0.3901 | 0.1799 | **0.7880** |
| B: EN+IT train -> IT test | 0.5168 | 0.5168 | 0.3856 | **0.8370** |
| B: EN+IT train -> EN test | 0.5564 | 0.5564 | 0.4399 | **0.8729** |
| C: IT train -> IT test (control) | 0.5109 | 0.5109 | 0.3741 | **0.8423** |
| D: EN + MT-IT train -> IT test (real IT) | 0.5644 | 0.5644 | 0.4096 | **0.8860** |
| D: EN + MT-IT train -> EN test | 0.6238 | 0.6238 | 0.5409 | **0.9161** |

**Primary comparison.** A(IT)/B(IT) AUC ratio = 0.7880 / 0.8370 = **0.9414** against the **0.975** bar -> **FAIL**. (B(IT)/A(IT) = 1.0622; A(IT)/A(EN) self-retention = 0.9017.)

**Control arm C (IT -> IT, in-language reference).** C reaches **0.8423** AUC on IT test (macro-F1 0.3741, accuracy 0.5109, micro-F1 0.5109).

| question | ratio | value | >= 0.975 | reading |
|---|---|---:|:--:|---|
| 1. difficulty: is IT harder than EN? | C(IT->IT) / A(EN->EN) | **0.9639** | no | Italian is HARDER than English (C/A < 0.975) |
| 2. EN-only retention vs in-language | A(EN->IT) / C(IT->IT) | **0.9355** | **no** | **below the 0.975 bar** |
| 3. EN+IT retention vs in-language | B(EN+IT->IT) / C(IT->IT) | **0.9937** | yes | within the 0.975 bar |
| 4. does adding EN to IT help? | B(EN+IT->IT) / C(IT->IT) | **0.9937** | yes | no resolved effect: B/C in [0.975, 1.0), point estimate slightly negative but inside the bar |

Rows 3 and 4 are the same quotient read two ways: B/C < 1 means English data *hurts* an Italian model, B/C > 1 means it helps, and 0.975-1.025 means no resolved effect.

**English side of arm B (same B model, English test).** B scores **0.8729** AUC on EN test vs A's **0.8739**. B_EN/A_EN = **0.9989** at the 0.975 bar -> **PASS**: adding Italian to an English model does not measurably hurt English.

**Arm D — translate-train (MT EN->IT).** Arm D used the existing flores_close MT cache (no MT re-run). The 1501 EN train+val rows were translated with `Helsinki-NLP/opus-mt-en-it` (greedy, fp16, batch 64, max_length 512), embedded with EuroBERT-210m mean, and the model was trained on EN + MT-IT (3002 rows). **Test text was never translated.**

| question | ratio | value | >= 0.975 | reading |
|---|---|---:|:--:|---|
| 1. MT-IT train vs real-IT train | D(IT) / B(IT) | **1.0586** | yes | D exceeds B by 5.9% but trains on 3002 vs 1994 rows; at equal 1994 rows D'/B = 1.0057 (parity) |
| 2. MT-IT train vs Italian ceiling | D(IT) / C(IT) | **1.0519** | yes | D exceeds the C ceiling by 5.2% with 3002 vs 997 rows; at equal rows D'/C = 0.9994 (parity) |
| 3. English retention of translate-train | D(EN) / A(EN) | **1.0483** | yes | D_EN is 4.8% above A_EN with 3x the rows; at equal rows D'_EN/A_EN = 1.0101 (no English cost) |

Variant: same tuned hyperparameters fit on the **1994 train-only rows** (protocol-matched to A/B/C): D'(IT) AUC **0.8418**, D'(EN) AUC **0.8827**; D'(IT)/B **1.0057**, D'(IT)/C **0.9994**.

Robustness refit (best params refit on train+val, the repo's usual convention):

| cell | accuracy | micro-F1 | macro-F1 | OVR-macro AUC |
|---|---:|---:|---:|---:|
| A -> EN test | 0.6119 | 0.6119 | 0.4935 | 0.9179 |
| A -> IT test | 0.4079 | 0.4079 | 0.2239 | 0.8243 |
| B -> IT test | 0.5644 | 0.5644 | 0.4388 | 0.8809 |
| B -> EN test | 0.6257 | 0.6257 | 0.5273 | 0.9190 |
| C -> IT test | 0.5881 | 0.5881 | 0.4620 | 0.8919 |
| D -> IT test | 0.5644 | 0.5644 | 0.4096 | 0.8860 |
| D -> EN test | 0.6238 | 0.6238 | 0.5409 | 0.9161 |

Refit A(IT)/B(IT) ratio = **0.9358** (bar 0.975 -> FAIL).
Refit A(IT)/C(IT) = **0.9242**, B(IT)/C(IT) = **0.9876**.

**Verdict.** Raw features only: A (EN train -> IT test) reaches **0.7880** AUC vs B (EN+IT train -> IT test) at **0.8370**, so A/B = **0.9414** -- **FAILS** the 0.975 bar (shortfall 5.9 pp of B's AUC). The English self-reference is 0.8739, so the EN-only model keeps 90.2% of its own English AUC on Italian and 94.1% of the in-language Italian model's AUC. No feature transform is involved, so this is the raw-feature floor of the E42 question. **Adding the IT->IT control:** C reaches 0.8423 AUC, so Italian is harder than English (C/A(EN) = 0.9639). Against that in-language reference the EN-only model keeps 0.9355 (A/C, below the 0.975 bar) and the EN+IT model keeps 0.9937 (B/C, above). Adding English to Italian has no resolved effect (B/C = 0.9937). **English side and translate-train:** B scores 0.8729 AUC on the English test vs A's 0.8739 (B_EN/A_EN = 0.9989, above the 0.975 bar), so adding Italian does not measurably hurt the English model. Translate-train (D, 3002 rows of EN + MT-IT) reaches 0.8860 on real Italian (D/B = 1.0586, beats B on row count) and 0.9161 on English (D_EN/A_EN = 1.0483, retained); against the Italian ceiling D/C = 1.0519. D trains on 3x A's rows and 1.5x B's, so the size-matched variant is the fair test: at 1994 rows D'/B = 1.0057 and D'/C = 0.9994, i.e. MT of the training text substitutes for real Italian at equal rows.

**Caveats.**
- Single seed (0). Test = 505 rows; AUC SE on this size is roughly 0.01-0.02, so differences under ~0.02 AUC are not resolved.
- All three arms share the same 12-class label vector because FLORES devtest rows are parallel across languages; B adds English and Italian labels, C adds Italian labels only -- the in-language reference this comparison needs.
- Optuna budget is 10 trials per arm -- enough to beat default params, not enough to claim the space is exhausted.
- The main numbers refit the best params on the arm's train rows only (literal to the spec); the train+val refit is reported separately as a robustness check.
- Ties in val AUC across trials are broken by Optuna's own ordering; no early stopping.
- Arm C is the in-language control, not a fourth transfer cell: it sees Italian labels, so C/A(EN) measures corpus difficulty (IT vs EN) on this split, not transfer.
- C introduces one more tuned model on 997 rows / 504-row val, so A/C and B/C inherit the same single-seed and small-test noise as the A/B ratio; compare verdicts, not the third decimal.
- B/C compares two models that differ in training data only after each was tuned on the same IT val with the same budget, so it is the cleanest of the three comparisons.
- Arm D trains on synthetic Italian: the MT text is fluent but is not a human translation, so D is an upper bound on what translation of training text can buy.
- D's hyperparameters are chosen on real IT val using a 1994-row train-only fit, then refit on 3002 rows (EN train+val + MT train+val). That keeps val out of the tuning signal while honouring the 3002-row train size; the protocol-matched 1994-row variant is reported alongside.
- B_EN uses arm B's stored best params (no re-tuning on English), so it measures the Italian-augmented model's English behaviour, not a best-case English model.
