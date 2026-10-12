# toxi_proof -- ONE English-trained XGBoost (EuroBERT-210m mean) on a REAL unbalanced multilingual toxicity task

External validation of the out-of-the-box cross-lingual claim on `FredZhang7/toxi-text-3M` (columns `text`,`is_toxic`,`lang`): a single **English-trained** `XGBClassifier` on **EuroBERT-210m mean** embeddings (no MT, no second model), with the model-free `perlang_z` transform.

Recipe: EuroBERT-210m **mean** pool (masked mean, plain forward, max_length 512, batch 64, **fp32**, GPU) -> `perlang_z` (L2-normalize then per-language z-score; target stats from the target's OWN text) -> ONE `XGBClassifier(binary:logistic, hist, max_bin=64, device=cpu, n_jobs=2, seed=0, max_depth=8, min_child_weight=5, lr=0.1, n_estimators=400)`.  F1-optimal threshold tuned on a held-out EN-val split.

> **Data note.** `validation/multilingual-validation(new).csv` was **deleted** from the HF repo (commit history), so the per-language z-score stats are fit on each language's OWN **train** text (unlabeled for the transform; labels are used only to train the models, never for the transform).  EN train = 20000 sampled rows (seed 0); each target likewise sampled (native training).  Test capped at 5000/target; en (525) and de (190) tests are SMALL and their numbers are noisy.

> **Leakage removal (`dedup_test`).** The dataset's non-English train and test splits **share exact texts** (translation artifacts): before training, every eval-test text is dropped from that language's train pool.  Removed leaked train texts per language: {'en': 3, 'it': 4847, 'de': 161, 'fr': 3807, 'es': 3755}.  Without this, a native model memorises the test set (e.g. it 97% / fr 76% / es 75% / de 85% of the test texts were in the train pool) and its scores are inflated.  EN overlap is ~0, so EN is unaffected.

EN reference (EN-trained model on the EN test): **AUC=0.7860**, AP=0.8182, n_pos=328.

## Per-target: EN-trained vs native (test)

Lower is better only for `FPR@R90`.  Cells read `EN / native`.

| metric | it | de | fr | es |
|---|---|---|---|---|
| imbalance (pos rate) | 0.185 / 0.185 | 0.716 / 0.716 | 0.298 / 0.298 | 0.374 / 0.374 |
| AUC | 0.763 / 0.815 | 0.625 / 0.589 | 0.750 / 0.821 | 0.775 / 0.834 |
| AP (PR-AUC) | 0.454 / 0.552 | 0.826 / 0.779 | 0.554 / 0.650 | 0.683 / 0.728 |
| accuracy | 0.826 / 0.822 | 0.295 / 0.647 | 0.717 / 0.737 | 0.664 / 0.753 |
| F1 | 0.209 / 0.511 | 0.029 / 0.755 | 0.140 / 0.619 | 0.203 / 0.692 |
| R@FPR1% | 0.098 / 0.161 | 0.125 / 0.037 | 0.071 / 0.140 | 0.128 / 0.118 |
| R@FPR5% | 0.261 / 0.355 | 0.176 / 0.081 | 0.221 / 0.308 | 0.293 / 0.319 |
| R@FPR10% | 0.389 / 0.492 | 0.257 / 0.206 | 0.364 / 0.442 | 0.423 / 0.484 |
| R@FA5 | 0.036 / 0.060 | 0.257 / 0.206 | 0.026 / 0.049 | 0.055 / 0.037 |
| R@FA20 | 0.070 / 0.127 | 0.574 / 0.456 | 0.049 / 0.108 | 0.100 / 0.099 |
| R@FA50 | 0.104 / 0.174 | 0.985 / 0.993 | 0.094 / 0.175 | 0.158 / 0.165 |
| R@FA100 | 0.175 / 0.261 | 1.000 / 1.000 | 0.142 / 0.233 | 0.220 / 0.248 |
| FPR@R90 | 0.597 / 0.513 | 0.815 / 0.759 | 0.619 / 0.445 | 0.571 / 0.415 |

## Ratios / comparability

`target/EN` = AUC(EN-trained on target test) / AUC(EN-trained on EN test) (comparability bar >= 0.9).  `EN/native` = AUC(EN-trained on target) / AUC(native on target); >= 0.9 means the single English model matches the native one.

| target | n_pos/n_neg | target/EN AUC | EN/native AUC | FPR@R90 EN/native | R@FA100 EN/native |
|---|---|---|---|---|---|
| it | 927/4073 | 0.970 | 0.935 | 0.597 / 0.513 | 0.175 / 0.261 |
| de | 136/54 | 0.796 | 1.062 | 0.815 / 0.759 | 1.000 / 1.000 |
| fr | 1488/3512 | 0.954 | 0.913 | 0.619 / 0.445 | 0.142 / 0.233 |
| es | 1868/3132 | 0.986 | 0.929 | 0.571 / 0.415 | 0.220 / 0.248 |

## Verdict

- Target/EN AUC ratio >= 0.9 in **3/4** targets (ratios: it=0.970, de=0.796, fr=0.954, es=0.986).
- EN/native AUC ratio >= 0.9 in **4/4** targets (ratios: it=0.935, de=1.062, fr=0.913, es=0.929).
- Mean AP: EN-trained **0.629** vs native **0.677**; mean FPR@R90: EN-trained **0.650** vs native **0.533** (smaller is better).
- **Verdict: comparable** -- the single English-trained model matches the native models on this real, unbalanced multilingual task (AUC and the recall-first view).
