# FLORES label-free calibration of ONE English XGBoost (EuroBERT-210m mean + `perlang_z`)

One English-trained XGBoost per dataset (softprob, depth 8 / mcw 5 / lr 0.1 / n 400, hist, max_bin 64, cpu, n_jobs 2, seed 0), trained on the EN **train** rows; the EN **val** rows fit the calibrations. Features = `perlang_z` (L2-normalize + per-language *train* z-score). All calibrations are **label-free** (target labels never used). Ratio denominator = **English self under the baseline**; comparable when ratio >= 0.9.

Calibrations: `baseline` argmax; `temperature` EN-val NLL temperature; `prior_sinkhorn` IPF so the target softmax distribution matches the English prior; `quantile_hardprior` EN-val quantile thresholds + hard prior matching; `bbse` BBSE target-prior estimate + prior correction.


## sent (12 classes, splits {'train': 997, 'val': 504, 'test': 505})

English prior (from EN train labels): 0.018, 0.075, 0.048, 0.083, 0.044, 0.032, 0.067, 0.050, 0.064, 0.183, 0.071, 0.264

| method | EN micro | EN macro | it micro | de micro | fr micro | it/EN micro | de/EN micro | fr/EN micro | it macro | de macro | fr macro | it/EN macro | de/EN macro | fr/EN macro | min ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.537 | 0.403 | 0.414 | 0.434 | 0.436 | 0.771 | 0.808 | 0.812 | 0.282 | 0.283 | 0.283 | 0.698 | 0.703 | 0.702 | **0.698** |
| `temperature` | 0.537 | 0.403 | 0.414 | 0.434 | 0.436 | 0.771 | 0.808 | 0.812 | 0.282 | 0.283 | 0.283 | 0.698 | 0.703 | 0.702 | **0.698** |
| `prior_sinkhorn` | 0.535 | 0.409 | 0.408 | 0.432 | 0.420 | 0.760 | 0.804 | 0.782 | 0.281 | 0.290 | 0.294 | 0.698 | 0.719 | 0.730 | **0.698** |
| `quantile_hardprior` | 0.505 | 0.397 | 0.398 | 0.392 | 0.420 | 0.742 | 0.731 | 0.782 | 0.299 | 0.288 | 0.348 | 0.741 | 0.714 | 0.862 | **0.714** |
| `bbse` | 0.487 | 0.319 | 0.400 | 0.432 | 0.392 | 0.745 | 0.804 | 0.731 | 0.219 | 0.235 | 0.180 | 0.543 | 0.584 | 0.446 | **0.446** |

Ratio denominator = EN baseline (micro 0.5366 / macro 0.4032). Accuracy == micro-F1 for single-label argmax; full acc/AUC/predicted-distribution per cell are in `flores_calib.json`.

**English self (baseline): micro-F1 0.5366, macro-F1 0.4032, AUC 0.8486.** For single-label argmax, accuracy == micro-F1 exactly. AUC is invariant across every calibration (each is a per-class monotone transform of the scores): 
| lang | AUC | AUC/EN |
|---|---:|---:|
| en (self) | 0.8486 | 1.000 |
| it | 0.7819 | 0.921 |
| de | 0.7859 | 0.926 |
| fr | 0.7784 | 0.917 |

- Best calibration for **micro-F1** (max of min target/EN micro ratio): `baseline` -> min micro ratio **0.771** (reaches >=0.9: **False**).
- Best calibration for **macro-F1**: `quantile_hardprior` -> min macro ratio **0.714** (reaches >=0.9: **False**).
- Best over **both** (max of the global min ratio): `quantile_hardprior` -> min ratio **0.714**; reaches >=0.9 on micro **False** / macro **False** / both **False**.

- `temperature` is **identical to `baseline`** for single-label argmax: a single global temperature is strictly order-preserving, so it cannot change the argmax (or F1).
- `bbse` is **unstable** here: the EN-val confusion matrix is ill-conditioned (negative target-prior estimates), so prior correction degrades even English (report as unstable; see `fit_info.*.bbse_info` in the JSON).
- `prior_sinkhorn` matches the target *softmax* distribution to the English prior; `quantile_hardprior` matches the *hard* argmax distribution. The latter raises rare-class (macro) F1 but trades away micro-F1 / accuracy.

**Verdict sent: comparability NOT reached on micro-F1 and NOT reached on macro-F1.**


## agg (11 classes, splits {'train': 276, 'val': 140, 'test': 139})

English prior (from EN train labels): 0.083, 0.036, 0.091, 0.051, 0.040, 0.065, 0.058, 0.065, 0.203, 0.080, 0.228

| method | EN micro | EN macro | it micro | de micro | fr micro | it/EN micro | de/EN micro | fr/EN micro | it macro | de macro | fr macro | it/EN macro | de/EN macro | fr/EN macro | min ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.554 | 0.501 | 0.439 | 0.504 | 0.547 | 0.792 | 0.909 | 0.987 | 0.278 | 0.392 | 0.459 | 0.555 | 0.782 | 0.914 | **0.555** |
| `temperature` | 0.554 | 0.501 | 0.439 | 0.504 | 0.547 | 0.792 | 0.909 | 0.987 | 0.278 | 0.392 | 0.459 | 0.555 | 0.782 | 0.914 | **0.555** |
| `prior_sinkhorn` | 0.568 | 0.489 | 0.504 | 0.511 | 0.547 | 0.909 | 0.922 | 0.987 | 0.349 | 0.366 | 0.412 | 0.697 | 0.731 | 0.821 | **0.697** |
| `quantile_hardprior` | 0.568 | 0.494 | 0.410 | 0.518 | 0.504 | 0.740 | 0.935 | 0.909 | 0.307 | 0.437 | 0.429 | 0.613 | 0.871 | 0.855 | **0.613** |
| `bbse` | 0.165 | 0.185 | 0.309 | 0.165 | 0.439 | 0.558 | 0.299 | 0.792 | 0.222 | 0.179 | 0.246 | 0.442 | 0.357 | 0.491 | **0.299** |

Ratio denominator = EN baseline (micro 0.5540 / macro 0.5014). Accuracy == micro-F1 for single-label argmax; full acc/AUC/predicted-distribution per cell are in `flores_calib.json`.

**English self (baseline): micro-F1 0.5540, macro-F1 0.5014, AUC 0.8882.** For single-label argmax, accuracy == micro-F1 exactly. AUC is invariant across every calibration (each is a per-class monotone transform of the scores): 
| lang | AUC | AUC/EN |
|---|---:|---:|
| en (self) | 0.8882 | 1.000 |
| it | 0.8587 | 0.967 |
| de | 0.8431 | 0.949 |
| fr | 0.8348 | 0.940 |

- Best calibration for **micro-F1** (max of min target/EN micro ratio): `prior_sinkhorn` -> min micro ratio **0.909** (reaches >=0.9: **True**).
- Best calibration for **macro-F1**: `prior_sinkhorn` -> min macro ratio **0.697** (reaches >=0.9: **False**).
- Best over **both** (max of the global min ratio): `prior_sinkhorn` -> min ratio **0.697**; reaches >=0.9 on micro **True** / macro **False** / both **False**.

- `temperature` is **identical to `baseline`** for single-label argmax: a single global temperature is strictly order-preserving, so it cannot change the argmax (or F1).
- `bbse` is **unstable** here: the EN-val confusion matrix is ill-conditioned (negative target-prior estimates), so prior correction degrades even English (report as unstable; see `fit_info.*.bbse_info` in the JSON).
- `prior_sinkhorn` matches the target *softmax* distribution to the English prior; `quantile_hardprior` matches the *hard* argmax distribution. The latter raises rare-class (macro) F1 but trades away micro-F1 / accuracy.

**Verdict agg: comparability reached on micro-F1 and NOT reached on macro-F1.**


## Global conclusion

- **sent**: best label-free calibration `quantile_hardprior` -> min target/EN ratio **0.714** (micro best `baseline` 0.771, macro best `quantile_hardprior` 0.714); >=0.9 on both micro and macro: **False**.
- **agg**: best label-free calibration `prior_sinkhorn` -> min target/EN ratio **0.697** (micro best `prior_sinkhorn` 0.909, macro best `prior_sinkhorn` 0.697); >=0.9 on both micro and macro: **False**.

### Honest residual

Label-free calibration moves the FLORES F1 ratios but does **not** fully close the gap. The residual is **not** a threshold/prior problem any more: once the target predicted distribution is (soft- or hard-)matched to the English prior -- which is the true target prior here, because FLORES is parallel -- the best target micro still tops at ~0.76-0.83 (sent) / ~0.91+ (agg) of English, and macro at ~0.70-0.86. The residual tracks the target's own ranking quality: the target OVR-macro AUC is ~0.78 (sent) / ~0.84-0.86 (agg) versus EN ~0.85 / ~0.89, i.e. within-class ranking degrades in transfer, and no per-class offset can repair within-class ordering. FLORES single-label labels are also noisy (the in-language ceiling is low), so a sizeable part of the English-vs-target F1 gap is irreducible for this representation without target labels. Matching marginals maximises one of micro/macro at the other's expense (a precision/recall trade), so no single calibration makes both >=0.9.
