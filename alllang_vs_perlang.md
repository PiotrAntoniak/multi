# all-langs vs per-lang training -- EuroBERT mean pooling, fixed XGBoost

FLORES (sent + agg) and TOXICITY are two SEPARATE problems (their data is never mixed). For each dataset and backbone: **all-langs** = ONE XGBoost trained on the pooled TRAIN rows of every language; **per-lang** = one XGBoost trained per language on its own TRAIN rows. Both are scored on EVERY language's held-out TEST split. Features: `perlang_z [both]` on every row (L2-normalize, then z-score with that row's own language TRAIN mean/std). Params fixed (hist, max_bin=64, cpu, n_jobs=2, seed=0, depth 8, mcw 5, lr 0.1, n400); no Optuna. Metrics: accuracy, macro-F1, micro-F1 / positive-class F1, OVR-macro AUC. `delta = all-langs - per-lang`; headline is macro-F1.

## flores_sent / 210m  (multiclass)

pooled all-langs TRAIN rows = 3988 | per-lang train rows = {'en': 997, 'it': 997, 'de': 997, 'fr': 997} | test rows (per language) = {'en': 505, 'it': 505, 'de': 505, 'fr': 505}

**accuracy**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.535 | 0.541 | +0.006 |
| it | 0.499 | 0.499 | +0.000 |
| de | 0.489 | 0.487 | -0.002 |
| fr | 0.523 | 0.549 | +0.026 |
| **macro** | **0.511** | **0.519** | **+0.007** |

**macro_f1 (**headline**)**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.395 | 0.421 | +0.025 |
| it | 0.378 | 0.354 | -0.023 |
| de | 0.374 | 0.388 | +0.015 |
| fr | 0.404 | 0.437 | +0.033 |
| **macro** | **0.388** | **0.400** | **+0.012** |

**micro_f1**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.535 | 0.541 | +0.006 |
| it | 0.499 | 0.499 | +0.000 |
| de | 0.489 | 0.487 | -0.002 |
| fr | 0.523 | 0.549 | +0.026 |
| **macro** | **0.511** | **0.519** | **+0.007** |

**auc**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.855 | 0.872 | +0.017 |
| it | 0.838 | 0.855 | +0.016 |
| de | 0.837 | 0.843 | +0.006 |
| fr | 0.825 | 0.855 | +0.030 |
| **macro** | **0.839** | **0.856** | **+0.017** |

verdict (macro_f1 macro): **all-langs helps** -- macro delta +0.012; per-language wins(all>per)=3 losses=1 out of 4.

## flores_sent / 610m  (multiclass)

pooled all-langs TRAIN rows = 3988 | per-lang train rows = {'en': 997, 'it': 997, 'de': 997, 'fr': 997} | test rows (per language) = {'en': 505, 'it': 505, 'de': 505, 'fr': 505}

**accuracy**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.517 | 0.541 | +0.024 |
| it | 0.450 | 0.495 | +0.046 |
| de | 0.505 | 0.543 | +0.038 |
| fr | 0.481 | 0.531 | +0.050 |
| **macro** | **0.488** | **0.527** | **+0.039** |

**macro_f1 (**headline**)**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.430 | 0.413 | -0.017 |
| it | 0.317 | 0.377 | +0.060 |
| de | 0.409 | 0.460 | +0.050 |
| fr | 0.361 | 0.409 | +0.048 |
| **macro** | **0.379** | **0.414** | **+0.035** |

**micro_f1**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.517 | 0.541 | +0.024 |
| it | 0.450 | 0.495 | +0.046 |
| de | 0.505 | 0.543 | +0.038 |
| fr | 0.481 | 0.531 | +0.050 |
| **macro** | **0.488** | **0.527** | **+0.039** |

**auc**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.841 | 0.876 | +0.035 |
| it | 0.830 | 0.846 | +0.016 |
| de | 0.847 | 0.858 | +0.010 |
| fr | 0.826 | 0.842 | +0.015 |
| **macro** | **0.836** | **0.855** | **+0.019** |

verdict (macro_f1 macro): **all-langs helps** -- macro delta +0.035; per-language wins(all>per)=3 losses=1 out of 4.

## flores_agg / 210m  (multiclass)

pooled all-langs TRAIN rows = 1104 | per-lang train rows = {'en': 276, 'it': 276, 'de': 276, 'fr': 276} | test rows (per language) = {'en': 139, 'it': 139, 'de': 139, 'fr': 139}

**accuracy**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.590 | 0.640 | +0.050 |
| it | 0.604 | 0.590 | -0.014 |
| de | 0.597 | 0.633 | +0.036 |
| fr | 0.504 | 0.626 | +0.122 |
| **macro** | **0.574** | **0.622** | **+0.049** |

**macro_f1 (**headline**)**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.546 | 0.570 | +0.023 |
| it | 0.540 | 0.498 | -0.042 |
| de | 0.526 | 0.559 | +0.033 |
| fr | 0.413 | 0.547 | +0.134 |
| **macro** | **0.506** | **0.543** | **+0.037** |

**micro_f1**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.590 | 0.640 | +0.050 |
| it | 0.604 | 0.590 | -0.014 |
| de | 0.597 | 0.633 | +0.036 |
| fr | 0.504 | 0.626 | +0.122 |
| **macro** | **0.574** | **0.622** | **+0.049** |

**auc**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.889 | 0.912 | +0.023 |
| it | 0.880 | 0.896 | +0.016 |
| de | 0.887 | 0.914 | +0.026 |
| fr | 0.868 | 0.894 | +0.027 |
| **macro** | **0.881** | **0.904** | **+0.023** |

verdict (macro_f1 macro): **all-langs helps** -- macro delta +0.037; per-language wins(all>per)=3 losses=1 out of 4.

## flores_agg / 610m  (multiclass)

pooled all-langs TRAIN rows = 1104 | per-lang train rows = {'en': 276, 'it': 276, 'de': 276, 'fr': 276} | test rows (per language) = {'en': 139, 'it': 139, 'de': 139, 'fr': 139}

**accuracy**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.532 | 0.647 | +0.115 |
| it | 0.590 | 0.590 | +0.000 |
| de | 0.540 | 0.597 | +0.058 |
| fr | 0.540 | 0.640 | +0.101 |
| **macro** | **0.550** | **0.619** | **+0.068** |

**macro_f1 (**headline**)**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.399 | 0.587 | +0.188 |
| it | 0.482 | 0.516 | +0.034 |
| de | 0.447 | 0.507 | +0.060 |
| fr | 0.457 | 0.580 | +0.123 |
| **macro** | **0.446** | **0.548** | **+0.101** |

**micro_f1**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.532 | 0.647 | +0.115 |
| it | 0.590 | 0.590 | +0.000 |
| de | 0.540 | 0.597 | +0.058 |
| fr | 0.540 | 0.640 | +0.101 |
| **macro** | **0.550** | **0.619** | **+0.068** |

**auc**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.849 | 0.900 | +0.051 |
| it | 0.869 | 0.901 | +0.032 |
| de | 0.889 | 0.894 | +0.005 |
| fr | 0.894 | 0.911 | +0.017 |
| **macro** | **0.875** | **0.901** | **+0.026** |

verdict (macro_f1 macro): **all-langs helps** -- macro delta +0.101; per-language wins(all>per)=4 losses=0 out of 4.

## toxic / 210m  (binary)

pooled all-langs TRAIN rows = 96668 | per-lang train rows = {'en': 24000, 'it': 18472, 'de': 9836, 'fr': 21011, 'es': 23349} | test rows (per language) = {'en': 3000, 'it': 2310, 'de': 1230, 'fr': 2627, 'es': 2919}

**accuracy**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.924 | 0.916 | -0.008 |
| it | 0.787 | 0.777 | -0.010 |
| de | 0.831 | 0.816 | -0.015 |
| fr | 0.788 | 0.789 | +0.002 |
| es | 0.758 | 0.748 | -0.010 |
| **macro** | **0.817** | **0.809** | **-0.008** |

**macro_f1 (**headline**)**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.771 | 0.744 | -0.028 |
| it | 0.709 | 0.695 | -0.014 |
| de | 0.791 | 0.770 | -0.021 |
| fr | 0.770 | 0.770 | +0.000 |
| es | 0.689 | 0.671 | -0.018 |
| **macro** | **0.746** | **0.730** | **-0.016** |

**f1**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.584 | 0.533 | -0.051 |
| it | 0.559 | 0.538 | -0.022 |
| de | 0.699 | 0.667 | -0.032 |
| fr | 0.705 | 0.703 | -0.002 |
| es | 0.542 | 0.512 | -0.030 |
| **macro** | **0.618** | **0.591** | **-0.027** |

**auc**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.903 | 0.902 | -0.000 |
| it | 0.839 | 0.823 | -0.016 |
| de | 0.881 | 0.865 | -0.017 |
| fr | 0.870 | 0.864 | -0.005 |
| es | 0.809 | 0.800 | -0.009 |
| **macro** | **0.860** | **0.851** | **-0.009** |

verdict (macro_f1 macro): **per-lang helps** -- macro delta -0.016; per-language wins(all>per)=1 losses=4 out of 5.

## toxic / 610m  (binary)

pooled all-langs TRAIN rows = 96668 | per-lang train rows = {'en': 24000, 'it': 18472, 'de': 9836, 'fr': 21011, 'es': 23349} | test rows (per language) = {'en': 3000, 'it': 2310, 'de': 1230, 'fr': 2627, 'es': 2919}

**accuracy**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.927 | 0.922 | -0.005 |
| it | 0.795 | 0.789 | -0.006 |
| de | 0.834 | 0.822 | -0.012 |
| fr | 0.796 | 0.789 | -0.007 |
| es | 0.754 | 0.744 | -0.010 |
| **macro** | **0.821** | **0.813** | **-0.008** |

**macro_f1 (**headline**)**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.783 | 0.769 | -0.014 |
| it | 0.722 | 0.711 | -0.011 |
| de | 0.795 | 0.775 | -0.020 |
| fr | 0.779 | 0.768 | -0.012 |
| es | 0.680 | 0.666 | -0.014 |
| **macro** | **0.752** | **0.738** | **-0.014** |

**f1**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.605 | 0.580 | -0.025 |
| it | 0.581 | 0.562 | -0.019 |
| de | 0.704 | 0.672 | -0.033 |
| fr | 0.719 | 0.698 | -0.021 |
| es | 0.526 | 0.505 | -0.021 |
| **macro** | **0.627** | **0.603** | **-0.024** |

**auc**

| language | per-lang | all-langs | delta (all-per) |
|---|---|---|---|
| en | 0.910 | 0.901 | -0.008 |
| it | 0.844 | 0.838 | -0.006 |
| de | 0.880 | 0.876 | -0.003 |
| fr | 0.875 | 0.875 | -0.001 |
| es | 0.815 | 0.801 | -0.014 |
| **macro** | **0.865** | **0.858** | **-0.007** |

verdict (macro_f1 macro): **per-lang helps** -- macro delta -0.014; per-language wins(all>per)=0 losses=5 out of 5.
