# Out-of-the-box cross-lingual proof -- EuroBERT-210m mean pooling + ONE English XGBoost

Model = one English-trained XGBoost per dataset; features are **mean-pooled** EuroBERT-210m with **model-free** transforms only (no MT, no second model, no target labels). English is the sole model language; every target test set is scored by that single model. Metric: FLORES micro/macro-F1 (argmax) + OVR-macro AUC; MultiEURLEX micro/macro-F1 at the EN-val global threshold + `quantile_match_enval` + OVR-macro AUC. Ratio = target / English self; **comparable when ratio >= 0.9**.

## sent (single-label multiclass, 12 classes/concepts)

train_n=997 val_n=504 test_n=505 | headline metric = `auc` | languages = en, it, de, fr

| config | en | it | de | fr | spread | it/EN | de/EN | fr/EN | min-ratio |
|---|---|---|---|---|---|---|---|---|---|
| `raw` | 0.856 | 0.759 | 0.763 | 0.768 | 0.097 | 0.886 | 0.891 | 0.897 | 0.886 |
| `l2` | 0.849 | 0.765 | 0.756 | 0.776 | 0.093 | 0.902 | 0.890 | 0.915 | 0.890 |
| `global_center` | 0.856 | 0.759 | 0.763 | 0.768 | 0.097 | 0.886 | 0.891 | 0.897 | 0.886 |
| `global_z` | 0.856 | 0.759 | 0.763 | 0.768 | 0.097 | 0.886 | 0.891 | 0.897 | 0.886 |
| `perlang_center` | 0.856 | 0.778 | 0.770 | 0.778 | 0.086 | 0.909 | 0.899 | 0.909 | 0.899 |
| `perlang_z` | 0.849 | 0.782 | 0.786 | 0.778 | 0.070 | 0.921 | 0.926 | 0.917 | 0.917 |

**micro_f1**

| config | en | it | de | fr | spread | it/EN | de/EN | fr/EN |
|---|---|---|---|---|---|---|---|---|
| `raw` | 0.535 | 0.378 | 0.384 | 0.396 | 0.156 | 0.707 | 0.719 | 0.741 |
| `l2` | 0.537 | 0.374 | 0.392 | 0.416 | 0.162 | 0.697 | 0.731 | 0.775 |
| `global_center` | 0.535 | 0.378 | 0.384 | 0.396 | 0.156 | 0.707 | 0.719 | 0.741 |
| `global_z` | 0.535 | 0.378 | 0.384 | 0.396 | 0.156 | 0.707 | 0.719 | 0.741 |
| `perlang_center` | 0.535 | 0.396 | 0.408 | 0.438 | 0.139 | 0.741 | 0.763 | 0.819 |
| `perlang_z` | 0.537 | 0.414 | 0.434 | 0.436 | 0.123 | 0.771 | 0.808 | 0.812 |

**macro_f1**

| config | en | it | de | fr | spread | it/EN | de/EN | fr/EN |
|---|---|---|---|---|---|---|---|---|
| `raw` | 0.411 | 0.158 | 0.263 | 0.211 | 0.254 | 0.384 | 0.639 | 0.512 |
| `l2` | 0.403 | 0.163 | 0.235 | 0.219 | 0.240 | 0.405 | 0.583 | 0.543 |
| `global_center` | 0.411 | 0.158 | 0.263 | 0.211 | 0.254 | 0.384 | 0.639 | 0.512 |
| `global_z` | 0.411 | 0.158 | 0.263 | 0.211 | 0.254 | 0.384 | 0.639 | 0.512 |
| `perlang_center` | 0.411 | 0.237 | 0.272 | 0.263 | 0.174 | 0.577 | 0.661 | 0.639 |
| `perlang_z` | 0.403 | 0.282 | 0.283 | 0.283 | 0.122 | 0.698 | 0.703 | 0.702 |

**auc**

| config | en | it | de | fr | spread | it/EN | de/EN | fr/EN |
|---|---|---|---|---|---|---|---|---|
| `raw` | 0.856 | 0.759 | 0.763 | 0.768 | 0.097 | 0.886 | 0.891 | 0.897 |
| `l2` | 0.849 | 0.765 | 0.756 | 0.776 | 0.093 | 0.902 | 0.890 | 0.915 |
| `global_center` | 0.856 | 0.759 | 0.763 | 0.768 | 0.097 | 0.886 | 0.891 | 0.897 |
| `global_z` | 0.856 | 0.759 | 0.763 | 0.768 | 0.097 | 0.886 | 0.891 | 0.897 |
| `perlang_center` | 0.856 | 0.778 | 0.770 | 0.778 | 0.086 | 0.909 | 0.899 | 0.909 |
| `perlang_z` | 0.849 | 0.782 | 0.786 | 0.778 | 0.070 | 0.921 | 0.926 | 0.917 |

comparable configs (headline `auc` >= 0.9 for every target): `perlang_z`

- comparable on `micro_f1`: NONE
- comparable on `macro_f1`: NONE
- comparable on `auc`: `perlang_z`

simplest comparable (headline `auc`): **perlang_z** | simplest NO-target-info: **None** | simplest mild-target-sample: **perlang_z**
simplest comparable on AUC: **perlang_z** | on micro-F1: **None**

## agg (single-label multiclass, 11 classes/concepts)

train_n=276 val_n=140 test_n=139 | headline metric = `auc` | languages = en, it, de, fr

| config | en | it | de | fr | spread | it/EN | de/EN | fr/EN | min-ratio |
|---|---|---|---|---|---|---|---|---|---|
| `raw` | 0.895 | 0.824 | 0.807 | 0.795 | 0.100 | 0.921 | 0.902 | 0.888 | 0.888 |
| `l2` | 0.888 | 0.827 | 0.798 | 0.788 | 0.100 | 0.931 | 0.898 | 0.887 | 0.887 |
| `global_center` | 0.895 | 0.824 | 0.807 | 0.795 | 0.100 | 0.921 | 0.902 | 0.888 | 0.888 |
| `global_z` | 0.895 | 0.824 | 0.807 | 0.795 | 0.100 | 0.921 | 0.902 | 0.888 | 0.888 |
| `perlang_center` | 0.895 | 0.859 | 0.815 | 0.836 | 0.080 | 0.960 | 0.911 | 0.935 | 0.911 |
| `perlang_z` | 0.888 | 0.859 | 0.843 | 0.835 | 0.053 | 0.967 | 0.949 | 0.940 | 0.940 |

**micro_f1**

| config | en | it | de | fr | spread | it/EN | de/EN | fr/EN |
|---|---|---|---|---|---|---|---|---|
| `raw` | 0.568 | 0.288 | 0.281 | 0.266 | 0.302 | 0.506 | 0.494 | 0.468 |
| `l2` | 0.554 | 0.345 | 0.317 | 0.352 | 0.237 | 0.623 | 0.571 | 0.636 |
| `global_center` | 0.568 | 0.288 | 0.281 | 0.266 | 0.302 | 0.506 | 0.494 | 0.468 |
| `global_z` | 0.568 | 0.288 | 0.281 | 0.266 | 0.302 | 0.506 | 0.494 | 0.468 |
| `perlang_center` | 0.568 | 0.468 | 0.532 | 0.475 | 0.101 | 0.823 | 0.937 | 0.835 |
| `perlang_z` | 0.554 | 0.439 | 0.504 | 0.547 | 0.115 | 0.792 | 0.909 | 0.987 |

**macro_f1**

| config | en | it | de | fr | spread | it/EN | de/EN | fr/EN |
|---|---|---|---|---|---|---|---|---|
| `raw` | 0.508 | 0.195 | 0.200 | 0.197 | 0.313 | 0.384 | 0.393 | 0.388 |
| `l2` | 0.501 | 0.225 | 0.268 | 0.247 | 0.277 | 0.448 | 0.535 | 0.494 |
| `global_center` | 0.508 | 0.195 | 0.200 | 0.197 | 0.313 | 0.384 | 0.393 | 0.388 |
| `global_z` | 0.508 | 0.195 | 0.200 | 0.197 | 0.313 | 0.384 | 0.393 | 0.388 |
| `perlang_center` | 0.508 | 0.347 | 0.423 | 0.356 | 0.162 | 0.681 | 0.831 | 0.700 |
| `perlang_z` | 0.501 | 0.279 | 0.392 | 0.459 | 0.223 | 0.555 | 0.782 | 0.914 |

**auc**

| config | en | it | de | fr | spread | it/EN | de/EN | fr/EN |
|---|---|---|---|---|---|---|---|---|
| `raw` | 0.895 | 0.824 | 0.807 | 0.795 | 0.100 | 0.921 | 0.902 | 0.888 |
| `l2` | 0.888 | 0.827 | 0.798 | 0.788 | 0.100 | 0.931 | 0.898 | 0.887 |
| `global_center` | 0.895 | 0.824 | 0.807 | 0.795 | 0.100 | 0.921 | 0.902 | 0.888 |
| `global_z` | 0.895 | 0.824 | 0.807 | 0.795 | 0.100 | 0.921 | 0.902 | 0.888 |
| `perlang_center` | 0.895 | 0.859 | 0.815 | 0.836 | 0.080 | 0.960 | 0.911 | 0.935 |
| `perlang_z` | 0.888 | 0.859 | 0.843 | 0.835 | 0.053 | 0.967 | 0.949 | 0.940 |

comparable configs (headline `auc` >= 0.9 for every target): `perlang_center`, `perlang_z`

- comparable on `micro_f1`: NONE
- comparable on `macro_f1`: NONE
- comparable on `auc`: `perlang_center`, `perlang_z`

simplest comparable (headline `auc`): **perlang_center** | simplest NO-target-info: **None** | simplest mild-target-sample: **perlang_center**
simplest comparable on AUC: **perlang_center** | on micro-F1: **None**

## mel (21-dim multi-label, 21 classes/concepts)

train_n=10000 val_n=1000 test_n=5000 | headline metric = `micro_f1_qm` | languages = en, de, fr

| config | en | de | fr | spread | de/EN | fr/EN | min-ratio |
|---|---|---|---|---|---|---|---|
| `raw` | 0.717 | 0.535 | 0.560 | 0.182 | 0.746 | 0.781 | 0.746 |
| `l2` | 0.717 | 0.529 | 0.568 | 0.189 | 0.737 | 0.792 | 0.737 |
| `global_center` | 0.717 | 0.535 | 0.560 | 0.182 | 0.746 | 0.781 | 0.746 |
| `global_z` | 0.717 | 0.535 | 0.560 | 0.182 | 0.746 | 0.781 | 0.746 |
| `perlang_center` | 0.717 | 0.658 | 0.659 | 0.059 | 0.918 | 0.919 | 0.918 |
| `perlang_z` | 0.718 | 0.669 | 0.667 | 0.051 | 0.932 | 0.928 | 0.928 |

**micro_f1**

| config | en | de | fr | spread | de/EN | fr/EN |
|---|---|---|---|---|---|---|
| `raw` | 0.735 | 0.319 | 0.446 | 0.415 | 0.434 | 0.607 |
| `l2` | 0.709 | 0.240 | 0.354 | 0.468 | 0.339 | 0.500 |
| `global_center` | 0.735 | 0.319 | 0.446 | 0.415 | 0.434 | 0.607 |
| `global_z` | 0.735 | 0.319 | 0.446 | 0.415 | 0.434 | 0.607 |
| `perlang_center` | 0.735 | 0.623 | 0.622 | 0.113 | 0.847 | 0.846 |
| `perlang_z` | 0.732 | 0.651 | 0.652 | 0.081 | 0.890 | 0.891 |

**macro_f1**

| config | en | de | fr | spread | de/EN | fr/EN |
|---|---|---|---|---|---|---|
| `raw` | 0.550 | 0.101 | 0.151 | 0.449 | 0.184 | 0.274 |
| `l2` | 0.514 | 0.064 | 0.095 | 0.450 | 0.125 | 0.185 |
| `global_center` | 0.550 | 0.101 | 0.151 | 0.449 | 0.184 | 0.274 |
| `global_z` | 0.550 | 0.101 | 0.151 | 0.449 | 0.184 | 0.274 |
| `perlang_center` | 0.550 | 0.340 | 0.344 | 0.210 | 0.619 | 0.625 |
| `perlang_z` | 0.556 | 0.390 | 0.403 | 0.166 | 0.701 | 0.726 |

**auc**

| config | en | de | fr | spread | de/EN | fr/EN |
|---|---|---|---|---|---|---|
| `raw` | 0.909 | 0.758 | 0.766 | 0.151 | 0.834 | 0.843 |
| `l2` | 0.906 | 0.750 | 0.782 | 0.156 | 0.828 | 0.864 |
| `global_center` | 0.909 | 0.758 | 0.766 | 0.151 | 0.834 | 0.843 |
| `global_z` | 0.909 | 0.758 | 0.766 | 0.151 | 0.834 | 0.843 |
| `perlang_center` | 0.909 | 0.848 | 0.861 | 0.061 | 0.933 | 0.947 |
| `perlang_z` | 0.906 | 0.854 | 0.870 | 0.052 | 0.943 | 0.960 |

**micro_f1_qm**

| config | en | de | fr | spread | de/EN | fr/EN |
|---|---|---|---|---|---|---|
| `raw` | 0.717 | 0.535 | 0.560 | 0.182 | 0.746 | 0.781 |
| `l2` | 0.717 | 0.529 | 0.568 | 0.189 | 0.737 | 0.792 |
| `global_center` | 0.717 | 0.535 | 0.560 | 0.182 | 0.746 | 0.781 |
| `global_z` | 0.717 | 0.535 | 0.560 | 0.182 | 0.746 | 0.781 |
| `perlang_center` | 0.717 | 0.658 | 0.659 | 0.059 | 0.918 | 0.919 |
| `perlang_z` | 0.718 | 0.669 | 0.667 | 0.051 | 0.932 | 0.928 |

**macro_f1_qm**

| config | en | de | fr | spread | de/EN | fr/EN |
|---|---|---|---|---|---|---|

comparable configs (headline `micro_f1_qm` >= 0.9 for every target): `perlang_center`, `perlang_z`

- comparable on `micro_f1`: NONE
- comparable on `macro_f1`: NONE
- comparable on `auc`: `perlang_center`, `perlang_z`
- comparable on `micro_f1_qm`: `perlang_center`, `perlang_z`
- comparable on `macro_f1_qm`: NONE

simplest comparable (headline `micro_f1_qm`): **perlang_center** | simplest NO-target-info: **None** | simplest mild-target-sample: **perlang_center**
simplest comparable on AUC: **perlang_center** | on micro-F1: **None**
