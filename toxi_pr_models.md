# toxi_pr_models -- precision/recall vs decision-score percentile for NON-XGBoost classifiers

Same toxicity task and figure layout as `toxi_pr.png` (XGBoost), swapped detector. Data `data/toxi_clean/`, **RAW mean** embeddings `emb_bf16/toxic/{210m,610m}/emb_<lang>_<split>.npy` (**no `perlang_z`** -- chosen to match the raw fine-tune convention and keep it simple). `210m` = solid, `610m` = dotted. Val threshold = F1-optimal on the model's own-language validation split.

**Sweep axis = decision-score percentile q (0..100), not raw probability 0..1.** The old raw-probability sweep was invalid for `GaussianNB` (saturated `predict_proba`, F1-opt val threshold at 0.99-1.00) and `KNeighborsClassifier` (k=5 -> only 6 discrete probabilities). Each model now uses a monotone, continuous decision score (see `toxi_pr_models.py` docstring): `SVC_linear`/`LogisticRegression` = `decision_function`; `MLP`/`RF` = `predict_proba[:,1]`; `GaussianNB` = log-likelihood-ratio; `KNN` = continuous distance-weighted positive fraction. `SVC_linear` uses the liblinear PRIMAL linear SVM (`LinearSVC(loss='squared_hinge', dual=False)`) -- plain libsvm `SVC(kernel='linear')` was infeasible at n=24k x 768 (>11 min/fit, non-convergent).


## P@recall=0.90 (EN-on-IT and IT-on-IT) and AUC, per model x backbone

`gap` = |P(EN/IT) - P(IT/IT)| at recall 0.90 (smaller = EN-on-IT closer to IT-on-IT). AUC and P@R90 are ranking-based (invariant to the monotone score rescaling); for KNN the score function itself changed (discrete fraction -> continuous distance-weighted fraction), so its AUC/P@R90 are recomputed.

| model | backbone | AUC EN/EN | AUC EN/IT | AUC IT/IT | P@R90 EN/EN | P@R90 EN/IT | P@R90 IT/IT | gap |
|---|---|---|---|---|---|---|---|---|
| XGBoost | 210m | 0.907 | 0.621 | 0.842 | 0.271 | 0.318 | 0.450 | 0.131 |
| XGBoost | 610m | 0.913 | 0.715 | 0.843 | 0.287 | 0.360 | 0.467 | 0.107 |
| SVC_linear | 210m | 0.896 | 0.633 | 0.823 | 0.264 | 0.321 | 0.442 | 0.121 |
| SVC_linear | 610m | 0.906 | 0.706 | 0.822 | 0.299 | 0.341 | 0.442 | 0.101 |
| MLPClassifier | 210m | 0.882 | 0.572 | 0.812 | 0.253 | 0.307 | 0.430 | 0.123 |
| MLPClassifier | 610m | 0.892 | 0.693 | 0.822 | 0.265 | 0.346 | 0.454 | 0.108 |
| LogisticRegression | 210m | 0.897 | 0.635 | 0.823 | 0.257 | 0.323 | 0.443 | 0.120 |
| LogisticRegression | 610m | 0.905 | 0.711 | 0.823 | 0.298 | 0.346 | 0.444 | 0.097 |
| RandomForestClassifier | 210m | 0.856 | 0.686 | 0.804 | 0.189 | 0.333 | 0.394 | 0.061 |
| RandomForestClassifier | 610m | 0.845 | 0.692 | 0.795 | 0.189 | 0.346 | 0.398 | 0.053 |
| KNeighborsClassifier | 210m | 0.810 | 0.454 | 0.777 | 0.121 | 0.294 | 0.414 | 0.120 |
| KNeighborsClassifier | 610m | 0.795 | 0.648 | 0.782 | 0.121 | 0.294 | 0.418 | 0.124 |
| GaussianNB | 210m | 0.851 | 0.663 | 0.729 | 0.235 | 0.329 | 0.344 | 0.015 |
| GaussianNB | 610m | 0.811 | 0.679 | 0.705 | 0.221 | 0.346 | 0.324 | 0.022 |

## Closest EN-on-IT to IT-on-IT @ recall 0.90 (per backbone)

**210m**: `GaussianNB` (gap 0.015; EN/IT 0.329 vs IT/IT 0.344), `RandomForestClassifier` (gap 0.061; EN/IT 0.333 vs IT/IT 0.394), `LogisticRegression` (gap 0.120; EN/IT 0.323 vs IT/IT 0.443), `KNeighborsClassifier` (gap 0.120; EN/IT 0.294 vs IT/IT 0.414), `SVC_linear` (gap 0.121; EN/IT 0.321 vs IT/IT 0.442), `MLPClassifier` (gap 0.123; EN/IT 0.307 vs IT/IT 0.430), `XGBoost` (gap 0.131; EN/IT 0.318 vs IT/IT 0.450)

**610m**: `GaussianNB` (gap 0.022; EN/IT 0.346 vs IT/IT 0.324), `RandomForestClassifier` (gap 0.053; EN/IT 0.346 vs IT/IT 0.398), `LogisticRegression` (gap 0.097; EN/IT 0.346 vs IT/IT 0.444), `SVC_linear` (gap 0.101; EN/IT 0.341 vs IT/IT 0.442), `XGBoost` (gap 0.107; EN/IT 0.360 vs IT/IT 0.467), `MLPClassifier` (gap 0.108; EN/IT 0.346 vs IT/IT 0.454), `KNeighborsClassifier` (gap 0.124; EN/IT 0.294 vs IT/IT 0.418)

## Grid figure paths (rows=models, cols={210m,610m}; vs decision-score percentile)

* `toxi_pr_models_grid.png` -- ONE combined grid: rows = classifier x {Precision, Recall} (14 rows for 7 classifiers), cols = backbone
* `toxi_pr_models_grid_precision.png` -- Precision only, overlaying EN-on-EN / EN-on-IT / IT-on-IT
* `toxi_pr_models_grid_recall.png` -- Recall only, overlaying EN-on-EN / EN-on-IT / IT-on-IT

## Per-model figure paths

* `toxi_pr_XGBoost.png`
* `toxi_pr_SVC_linear.png`
* `toxi_pr_MLPClassifier.png`
* `toxi_pr_LogisticRegression.png`
* `toxi_pr_RandomForestClassifier.png`
* `toxi_pr_KNeighborsClassifier.png`
* `toxi_pr_GaussianNB.png`

