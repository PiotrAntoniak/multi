# models_compare - non-XGBoost classifiers on FLORES-12 (sentence), EN + IT

Features: frozen EuroBERT **mean** sentence embeddings + `perlang_z [both]` (L2-normalize + per-language **train** z-score). Final models train on **train+val** (1501) and are evaluated on the seed-0 **test** (505). Metrics: macro-F1 (primary), accuracy, OVR-macro AUC. `LogisticRegression` (C in {0.1,1,10}) and `KNN` (k in {5,15}) pick their config by own-language val macro-F1; all others use sklearn defaults. `XGBoost` is the project reference (flores_calib.XGB_COMMON).

Runtime adaptations (documented): `GradientBoosting` uses `max_features='sqrt'` (sklearn's `None` default needed ~745 s/fit on the 1152-d backbone; `sqrt` ~24 s), and `SVC` uses `decision_function_shape='ovr'` scores for AUC instead of `probability=True` (the 5-fold Platt CV). All other models are sklearn defaults.

Backbones: **210m** = `embeddings/mean` (768-d); **610m** = `embeddings_610/mean` (1152-d). CPU, n_jobs=2.


## Backbone 210m

| model | EN-self F1 | EN-self acc | EN-self AUC | IT-self F1 | EN->IT F1 | IT->EN F1 | transfer F1 |
|---|---|---|---|---|---|---|---|
| SVC_linear | 0.555 | 0.598 | 0.921 | 0.478 | 0.407 | 0.454 | 0.430 |
| MLP | 0.538 | 0.632 | 0.922 | 0.455 | 0.413 | 0.451 | 0.432 |
| SVC_rbf | 0.514 | 0.650 | 0.929 | 0.459 | 0.334 | 0.416 | 0.375 |
| SGD | 0.512 | 0.588 | 0.917 | 0.429 | 0.332 | 0.392 | 0.362 |
| LogisticRegression | 0.503 | 0.562 | 0.905 | 0.452 | 0.338 | 0.435 | 0.387 |
| LDA | 0.501 | 0.539 | 0.876 | 0.378 | 0.216 | 0.210 | 0.213 |
| XGBoost | 0.486 | 0.606 | 0.904 | 0.434 | 0.366 | 0.446 | 0.406 |
| HistGradientBoosting | 0.486 | 0.614 | 0.898 | 0.399 | 0.317 | 0.321 | 0.319 |
| Ridge | 0.484 | 0.549 | 0.866 | 0.362 | 0.217 | 0.183 | 0.200 |
| KNN | 0.471 | 0.596 | 0.919 | 0.483 | 0.404 | 0.473 | 0.439 |
| GaussianNB | 0.463 | 0.517 | 0.902 | 0.390 | 0.464 | 0.443 | 0.453 |
| LinearSVC | 0.447 | 0.521 | 0.862 | 0.354 | 0.202 | 0.248 | 0.225 |
| GradientBoosting | 0.395 | 0.539 | 0.873 | 0.369 | 0.250 | 0.282 | 0.266 |
| ExtraTrees | 0.391 | 0.560 | 0.918 | 0.278 | 0.205 | 0.189 | 0.197 |
| RandomForest | 0.388 | 0.550 | 0.904 | 0.325 | 0.204 | 0.195 | 0.199 |
| DecisionTree | 0.235 | 0.329 | 0.591 | 0.201 | 0.120 | 0.164 | 0.142 |
| QDA | 0.082 | 0.099 | 0.506 | 0.083 | 0.085 | 0.075 | 0.080 |

* **Best non-XGBoost EN-self**: `SVC_linear` (macro-F1 0.555) vs XGBoost 0.486.
* **Best non-XGBoost transfer** (mean of EN->IT and IT->EN): `GaussianNB` (macro-F1 0.453) vs XGBoost 0.406.
* **Beats/matches XGBoost EN-self**: `SVC_linear`, `MLP`, `SVC_rbf`, `SGD`, `LogisticRegression`, `LDA`.
* **Beats/matches XGBoost transfer**: `GaussianNB`, `KNN`, `MLP`, `SVC_linear`.

EN-self ranking (macro-F1): SVC_linear (0.555) > MLP (0.538) > SVC_rbf (0.514) > SGD (0.512) > LogisticRegression (0.503) > LDA (0.501) > HistGradientBoosting (0.486) > Ridge (0.484) > KNN (0.471) > GaussianNB (0.463) > LinearSVC (0.447) > GradientBoosting (0.395) > ExtraTrees (0.391) > RandomForest (0.388) > DecisionTree (0.235) > QDA (0.082)

Transfer ranking (macro-F1, mean EN->IT & IT->EN): GaussianNB (0.453) > KNN (0.439) > MLP (0.432) > SVC_linear (0.430) > LogisticRegression (0.387) > SVC_rbf (0.375) > SGD (0.362) > HistGradientBoosting (0.319) > GradientBoosting (0.266) > LinearSVC (0.225) > LDA (0.213) > Ridge (0.200) > RandomForest (0.199) > ExtraTrees (0.197) > DecisionTree (0.142) > QDA (0.080)


## Backbone 610m

| model | EN-self F1 | EN-self acc | EN-self AUC | IT-self F1 | EN->IT F1 | IT->EN F1 | transfer F1 |
|---|---|---|---|---|---|---|---|
| SVC_linear | 0.583 | 0.636 | 0.939 | 0.539 | 0.519 | 0.606 | 0.562 |
| SGD | 0.571 | 0.636 | 0.936 | 0.507 | 0.460 | 0.542 | 0.501 |
| MLP | 0.564 | 0.671 | 0.939 | 0.517 | 0.492 | 0.543 | 0.517 |
| LogisticRegression | 0.541 | 0.630 | 0.935 | 0.550 | 0.474 | 0.553 | 0.514 |
| SVC_rbf | 0.523 | 0.663 | 0.942 | 0.457 | 0.447 | 0.492 | 0.469 |
| LinearSVC | 0.521 | 0.590 | 0.888 | 0.479 | 0.361 | 0.440 | 0.400 |
| KNN | 0.506 | 0.638 | 0.938 | 0.445 | 0.469 | 0.536 | 0.502 |
| XGBoost | 0.495 | 0.612 | 0.910 | 0.428 | 0.461 | 0.472 | 0.466 |
| GaussianNB | 0.484 | 0.552 | 0.914 | 0.503 | 0.456 | 0.532 | 0.494 |
| HistGradientBoosting | 0.469 | 0.598 | 0.893 | 0.392 | 0.394 | 0.427 | 0.410 |
| Ridge | 0.436 | 0.467 | 0.826 | 0.330 | 0.246 | 0.270 | 0.258 |
| LDA | 0.423 | 0.450 | 0.835 | 0.367 | 0.251 | 0.292 | 0.272 |
| GradientBoosting | 0.409 | 0.549 | 0.881 | 0.360 | 0.295 | 0.369 | 0.332 |
| RandomForest | 0.319 | 0.507 | 0.910 | 0.264 | 0.248 | 0.261 | 0.255 |
| ExtraTrees | 0.294 | 0.501 | 0.902 | 0.281 | 0.226 | 0.228 | 0.227 |
| DecisionTree | 0.174 | 0.265 | 0.558 | 0.171 | 0.135 | 0.186 | 0.161 |
| QDA | 0.074 | 0.089 | 0.495 | 0.077 | 0.081 | 0.085 | 0.083 |

* **Best non-XGBoost EN-self**: `SVC_linear` (macro-F1 0.583) vs XGBoost 0.495.
* **Best non-XGBoost transfer** (mean of EN->IT and IT->EN): `SVC_linear` (macro-F1 0.562) vs XGBoost 0.466.
* **Beats/matches XGBoost EN-self**: `SVC_linear`, `SGD`, `MLP`, `LogisticRegression`, `SVC_rbf`, `LinearSVC`, `KNN`.
* **Beats/matches XGBoost transfer**: `SVC_linear`, `MLP`, `LogisticRegression`, `KNN`, `SGD`, `GaussianNB`, `SVC_rbf`.

EN-self ranking (macro-F1): SVC_linear (0.583) > SGD (0.571) > MLP (0.564) > LogisticRegression (0.541) > SVC_rbf (0.523) > LinearSVC (0.521) > KNN (0.506) > GaussianNB (0.484) > HistGradientBoosting (0.469) > Ridge (0.436) > LDA (0.423) > GradientBoosting (0.409) > RandomForest (0.319) > ExtraTrees (0.294) > DecisionTree (0.174) > QDA (0.074)

Transfer ranking (macro-F1, mean EN->IT & IT->EN): SVC_linear (0.562) > MLP (0.517) > LogisticRegression (0.514) > KNN (0.502) > SGD (0.501) > GaussianNB (0.494) > SVC_rbf (0.469) > HistGradientBoosting (0.410) > LinearSVC (0.400) > GradientBoosting (0.332) > LDA (0.272) > Ridge (0.258) > RandomForest (0.255) > ExtraTrees (0.227) > DecisionTree (0.161) > QDA (0.083)


## Config

```json
{
  "emb": {
    "210m": "embeddings/mean",
    "610m": "embeddings_610/mean"
  },
  "feature": "perlang_z [both] (L2-normalize + per-language TRAIN z-score)",
  "train": "train+val (1501)",
  "test": "seed-0 test (505)",
  "metrics": [
    "accuracy",
    "macro_f1",
    "auc_ovr_macro"
  ],
  "xgb_reference": {
    "objective": "multi:softprob",
    "tree_method": "hist",
    "device": "cpu",
    "max_bin": 64,
    "n_jobs": 2,
    "seed": 0,
    "max_depth": 8,
    "min_child_weight": 5,
    "learning_rate": 0.1,
    "n_estimators": 400
  },
  "light_grid": {
    "LogisticRegression": [
      0.1,
      1.0,
      10.0
    ],
    "KNN": [
      5,
      15
    ]
  },
  "runtime_notes": [
    "GradientBoosting uses max_features='sqrt' (sklearn default None was ~745 s/fit on 610m; sqrt is ~24 s/fit). All other models keep sklearn defaults.",
    "SVC uses decision_function_shape='ovr' + decision_function for AUC (no probability=True 5-fold CV) to keep per-fit time ~1 s."
  ],
  "seed": 0,
  "n_jobs": 2,
  "smoke": false
}
```
