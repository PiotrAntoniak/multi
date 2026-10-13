# backbone_grid -- XGBoost across EuroBERT-210m / 610m x datasets x methods

Mean pooling only; XGBoost with 10 Optuna TPE trials (seed 0) per cell; objective = val AUC (macro OVR). Trained on English, evaluated on the English test (EN self) and on every non-English target test; XL-mean = mean over non-English targets. B = backbone.

## macro_f1

| dataset | backbone | method | EN self macro_f1 | XL-mean macro_f1 |
|---|---|---|---:|---:|
| flores_sent | 210m | mean (raw) | 0.375 | 0.199 |
| flores_sent | 210m | pc (global PCA k=2) | 0.378 | 0.207 |
| flores_sent | 210m | inlp (~30-dir) | 0.386 | 0.206 |
| flores_sent | 210m | D (language-shift) | 0.375 | 0.209 |
| flores_sent | 610m | mean (raw) | 0.403 | 0.241 |
| flores_sent | 610m | pc (global PCA k=2) | 0.403 | 0.265 |
| flores_sent | 610m | inlp (~30-dir) | 0.391 | 0.307 |
| flores_sent | 610m | D (language-shift) | 0.403 | 0.274 |
| flores_agg | 210m | mean (raw) | 0.493 | 0.294 |
| flores_agg | 210m | pc (global PCA k=2) | 0.457 | 0.282 |
| flores_agg | 210m | inlp (~30-dir) | 0.492 | 0.292 |
| flores_agg | 210m | D (language-shift) | 0.493 | 0.293 |
| flores_agg | 610m | mean (raw) | 0.406 | 0.302 |
| flores_agg | 610m | pc (global PCA k=2) | 0.403 | 0.340 |
| flores_agg | 610m | inlp (~30-dir) | 0.429 | 0.301 |
| flores_agg | 610m | D (language-shift) | 0.406 | 0.296 |
| mel | 210m | mean (raw) | 0.509 | 0.059 |
| mel | 210m | pc (global PCA k=2) | 0.521 | 0.293 |
| mel | 210m | inlp (~30-dir) | 0.523 | 0.299 |
| mel | 210m | D (language-shift) | 0.509 | 0.298 |
| mel | 610m | mean (raw) | 0.501 | 0.115 |
| mel | 610m | pc (global PCA k=2) | 0.517 | 0.338 |
| mel | 610m | inlp (~30-dir) | 0.495 | 0.349 |
| mel | 610m | D (language-shift) | 0.501 | 0.329 |
| sst2 | 210m | mean (raw) | 0.805 | n/a |
| sst2 | 210m | pc (global PCA k=2) | 0.809 | n/a |
| sst2 | 210m | inlp (~30-dir) | 0.805 | n/a |
| sst2 | 210m | D (language-shift) | 0.805 | n/a |
| sst2 | 610m | mean (raw) | 0.765 | n/a |
| sst2 | 610m | pc (global PCA k=2) | 0.762 | n/a |
| sst2 | 610m | inlp (~30-dir) | 0.765 | n/a |
| sst2 | 610m | D (language-shift) | 0.765 | n/a |
| emotion | 210m | mean (raw) | 0.365 | n/a |
| emotion | 210m | pc (global PCA k=2) | 0.374 | n/a |
| emotion | 210m | inlp (~30-dir) | 0.365 | n/a |
| emotion | 210m | D (language-shift) | 0.365 | n/a |
| emotion | 610m | mean (raw) | 0.358 | n/a |
| emotion | 610m | pc (global PCA k=2) | 0.399 | n/a |
| emotion | 610m | inlp (~30-dir) | 0.358 | n/a |
| emotion | 610m | D (language-shift) | 0.358 | n/a |
| toxic | 210m | mean (raw) | 0.770 | 0.470 |
| toxic | 210m | pc (global PCA k=2) | 0.750 | 0.450 |
| toxic | 210m | inlp (~30-dir) | 0.767 | 0.482 |
| toxic | 210m | D (language-shift) | 0.770 | 0.447 |
| toxic | 610m | mean (raw) | 0.784 | 0.500 |
| toxic | 610m | pc (global PCA k=2) | 0.774 | 0.465 |
| toxic | 610m | inlp (~30-dir) | 0.788 | 0.494 |
| toxic | 610m | D (language-shift) | 0.784 | 0.452 |

## auc

| dataset | backbone | method | EN self auc | XL-mean auc |
|---|---|---|---:|---:|
| flores_sent | 210m | mean (raw) | 0.884 | 0.811 |
| flores_sent | 210m | pc (global PCA k=2) | 0.878 | 0.808 |
| flores_sent | 210m | inlp (~30-dir) | 0.882 | 0.817 |
| flores_sent | 210m | D (language-shift) | 0.884 | 0.821 |
| flores_sent | 610m | mean (raw) | 0.890 | 0.829 |
| flores_sent | 610m | pc (global PCA k=2) | 0.881 | 0.827 |
| flores_sent | 610m | inlp (~30-dir) | 0.879 | 0.837 |
| flores_sent | 610m | D (language-shift) | 0.890 | 0.830 |
| flores_agg | 210m | mean (raw) | 0.924 | 0.881 |
| flores_agg | 210m | pc (global PCA k=2) | 0.920 | 0.886 |
| flores_agg | 210m | inlp (~30-dir) | 0.920 | 0.887 |
| flores_agg | 210m | D (language-shift) | 0.924 | 0.874 |
| flores_agg | 610m | mean (raw) | 0.906 | 0.885 |
| flores_agg | 610m | pc (global PCA k=2) | 0.911 | 0.891 |
| flores_agg | 610m | inlp (~30-dir) | 0.911 | 0.875 |
| flores_agg | 610m | D (language-shift) | 0.906 | 0.883 |
| mel | 210m | mean (raw) | 0.912 | 0.807 |
| mel | 210m | pc (global PCA k=2) | 0.909 | 0.866 |
| mel | 210m | inlp (~30-dir) | 0.910 | 0.866 |
| mel | 210m | D (language-shift) | 0.912 | 0.864 |
| mel | 610m | mean (raw) | 0.912 | 0.839 |
| mel | 610m | pc (global PCA k=2) | 0.911 | 0.880 |
| mel | 610m | inlp (~30-dir) | 0.909 | 0.881 |
| mel | 610m | D (language-shift) | 0.912 | 0.881 |
| sst2 | 210m | mean (raw) | 0.894 | n/a |
| sst2 | 210m | pc (global PCA k=2) | 0.894 | n/a |
| sst2 | 210m | inlp (~30-dir) | 0.894 | n/a |
| sst2 | 210m | D (language-shift) | 0.894 | n/a |
| sst2 | 610m | mean (raw) | 0.856 | n/a |
| sst2 | 610m | pc (global PCA k=2) | 0.853 | n/a |
| sst2 | 610m | inlp (~30-dir) | 0.856 | n/a |
| sst2 | 610m | D (language-shift) | 0.856 | n/a |
| emotion | 210m | mean (raw) | 0.824 | n/a |
| emotion | 210m | pc (global PCA k=2) | 0.826 | n/a |
| emotion | 210m | inlp (~30-dir) | 0.824 | n/a |
| emotion | 210m | D (language-shift) | 0.824 | n/a |
| emotion | 610m | mean (raw) | 0.832 | n/a |
| emotion | 610m | pc (global PCA k=2) | 0.830 | n/a |
| emotion | 610m | inlp (~30-dir) | 0.832 | n/a |
| emotion | 610m | D (language-shift) | 0.832 | n/a |
| toxic | 210m | mean (raw) | 0.908 | 0.706 |
| toxic | 210m | pc (global PCA k=2) | 0.907 | 0.691 |
| toxic | 210m | inlp (~30-dir) | 0.910 | 0.698 |
| toxic | 210m | D (language-shift) | 0.908 | 0.685 |
| toxic | 610m | mean (raw) | 0.913 | 0.730 |
| toxic | 610m | pc (global PCA k=2) | 0.912 | 0.704 |
| toxic | 610m | inlp (~30-dir) | 0.913 | 0.705 |
| toxic | 610m | D (language-shift) | 0.913 | 0.728 |

## micro_f1

| dataset | backbone | method | EN self micro_f1 | XL-mean micro_f1 |
|---|---|---|---:|---:|
| flores_sent | 210m | mean (raw) | 0.531 | 0.406 |
| flores_sent | 210m | pc (global PCA k=2) | 0.531 | 0.407 |
| flores_sent | 210m | inlp (~30-dir) | 0.525 | 0.391 |
| flores_sent | 210m | D (language-shift) | 0.531 | 0.415 |
| flores_sent | 610m | mean (raw) | 0.549 | 0.426 |
| flores_sent | 610m | pc (global PCA k=2) | 0.533 | 0.437 |
| flores_sent | 610m | inlp (~30-dir) | 0.550 | 0.475 |
| flores_sent | 610m | D (language-shift) | 0.549 | 0.457 |
| flores_agg | 210m | mean (raw) | 0.597 | 0.468 |
| flores_agg | 210m | pc (global PCA k=2) | 0.583 | 0.496 |
| flores_agg | 210m | inlp (~30-dir) | 0.612 | 0.492 |
| flores_agg | 210m | D (language-shift) | 0.597 | 0.508 |
| flores_agg | 610m | mean (raw) | 0.576 | 0.482 |
| flores_agg | 610m | pc (global PCA k=2) | 0.568 | 0.489 |
| flores_agg | 610m | inlp (~30-dir) | 0.576 | 0.496 |
| flores_agg | 610m | D (language-shift) | 0.576 | 0.501 |
| mel | 210m | mean (raw) | 0.702 | 0.242 |
| mel | 210m | pc (global PCA k=2) | 0.711 | 0.591 |
| mel | 210m | inlp (~30-dir) | 0.707 | 0.596 |
| mel | 210m | D (language-shift) | 0.702 | 0.588 |
| mel | 610m | mean (raw) | 0.711 | 0.314 |
| mel | 610m | pc (global PCA k=2) | 0.722 | 0.629 |
| mel | 610m | inlp (~30-dir) | 0.705 | 0.639 |
| mel | 610m | D (language-shift) | 0.711 | 0.622 |
| sst2 | 210m | mean (raw) | 0.808 | n/a |
| sst2 | 210m | pc (global PCA k=2) | 0.812 | n/a |
| sst2 | 210m | inlp (~30-dir) | 0.808 | n/a |
| sst2 | 210m | D (language-shift) | 0.808 | n/a |
| sst2 | 610m | mean (raw) | 0.770 | n/a |
| sst2 | 610m | pc (global PCA k=2) | 0.767 | n/a |
| sst2 | 610m | inlp (~30-dir) | 0.770 | n/a |
| sst2 | 610m | D (language-shift) | 0.770 | n/a |
| emotion | 210m | mean (raw) | 0.559 | n/a |
| emotion | 210m | pc (global PCA k=2) | 0.560 | n/a |
| emotion | 210m | inlp (~30-dir) | 0.559 | n/a |
| emotion | 210m | D (language-shift) | 0.559 | n/a |
| emotion | 610m | mean (raw) | 0.565 | n/a |
| emotion | 610m | pc (global PCA k=2) | 0.577 | n/a |
| emotion | 610m | inlp (~30-dir) | 0.565 | n/a |
| emotion | 610m | D (language-shift) | 0.565 | n/a |
| toxic | 210m | mean (raw) | 0.923 | 0.694 |
| toxic | 210m | pc (global PCA k=2) | 0.919 | 0.689 |
| toxic | 210m | inlp (~30-dir) | 0.923 | 0.695 |
| toxic | 210m | D (language-shift) | 0.923 | 0.687 |
| toxic | 610m | mean (raw) | 0.929 | 0.701 |
| toxic | 610m | pc (global PCA k=2) | 0.926 | 0.693 |
| toxic | 610m | inlp (~30-dir) | 0.928 | 0.698 |
| toxic | 610m | D (language-shift) | 0.929 | 0.691 |

## subset_acc (mel only)

| dataset | backbone | method | EN self subset_acc | XL-mean subset_acc |
|---|---|---|---:|---:|
| flores_sent | 210m | mean (raw) | n/a | n/a |
| flores_sent | 210m | pc (global PCA k=2) | n/a | n/a |
| flores_sent | 210m | inlp (~30-dir) | n/a | n/a |
| flores_sent | 210m | D (language-shift) | n/a | n/a |
| flores_sent | 610m | mean (raw) | n/a | n/a |
| flores_sent | 610m | pc (global PCA k=2) | n/a | n/a |
| flores_sent | 610m | inlp (~30-dir) | n/a | n/a |
| flores_sent | 610m | D (language-shift) | n/a | n/a |
| flores_agg | 210m | mean (raw) | n/a | n/a |
| flores_agg | 210m | pc (global PCA k=2) | n/a | n/a |
| flores_agg | 210m | inlp (~30-dir) | n/a | n/a |
| flores_agg | 210m | D (language-shift) | n/a | n/a |
| flores_agg | 610m | mean (raw) | n/a | n/a |
| flores_agg | 610m | pc (global PCA k=2) | n/a | n/a |
| flores_agg | 610m | inlp (~30-dir) | n/a | n/a |
| flores_agg | 610m | D (language-shift) | n/a | n/a |
| mel | 210m | mean (raw) | 0.118 | 0.002 |
| mel | 210m | pc (global PCA k=2) | 0.114 | 0.063 |
| mel | 210m | inlp (~30-dir) | 0.113 | 0.067 |
| mel | 210m | D (language-shift) | 0.118 | 0.068 |
| mel | 610m | mean (raw) | 0.142 | 0.017 |
| mel | 610m | pc (global PCA k=2) | 0.142 | 0.084 |
| mel | 610m | inlp (~30-dir) | 0.138 | 0.094 |
| mel | 610m | D (language-shift) | 0.142 | 0.085 |
| sst2 | 210m | mean (raw) | n/a | n/a |
| sst2 | 210m | pc (global PCA k=2) | n/a | n/a |
| sst2 | 210m | inlp (~30-dir) | n/a | n/a |
| sst2 | 210m | D (language-shift) | n/a | n/a |
| sst2 | 610m | mean (raw) | n/a | n/a |
| sst2 | 610m | pc (global PCA k=2) | n/a | n/a |
| sst2 | 610m | inlp (~30-dir) | n/a | n/a |
| sst2 | 610m | D (language-shift) | n/a | n/a |
| emotion | 210m | mean (raw) | n/a | n/a |
| emotion | 210m | pc (global PCA k=2) | n/a | n/a |
| emotion | 210m | inlp (~30-dir) | n/a | n/a |
| emotion | 210m | D (language-shift) | n/a | n/a |
| emotion | 610m | mean (raw) | n/a | n/a |
| emotion | 610m | pc (global PCA k=2) | n/a | n/a |
| emotion | 610m | inlp (~30-dir) | n/a | n/a |
| emotion | 610m | D (language-shift) | n/a | n/a |
| toxic | 210m | mean (raw) | n/a | n/a |
| toxic | 210m | pc (global PCA k=2) | n/a | n/a |
| toxic | 210m | inlp (~30-dir) | n/a | n/a |
| toxic | 210m | D (language-shift) | n/a | n/a |
| toxic | 610m | mean (raw) | n/a | n/a |
| toxic | 610m | pc (global PCA k=2) | n/a | n/a |
| toxic | 610m | inlp (~30-dir) | n/a | n/a |
| toxic | 610m | D (language-shift) | n/a | n/a |

## EN self / XL-mean side by side (macro-F1)

| dataset | method | 210m EN | 210m XL | 610m EN | 610m XL |
|---|---|---:|---:|---:|---:|
| flores_sent | mean (raw) | 0.375 | 0.199 | 0.403 | 0.241 |
| flores_sent | pc (global PCA k=2) | 0.378 | 0.207 | 0.403 | 0.265 |
| flores_sent | inlp (~30-dir) | 0.386 | 0.206 | 0.391 | 0.307 |
| flores_sent | D (language-shift) | 0.375 | 0.209 | 0.403 | 0.274 |
| flores_agg | mean (raw) | 0.493 | 0.294 | 0.406 | 0.302 |
| flores_agg | pc (global PCA k=2) | 0.457 | 0.282 | 0.403 | 0.340 |
| flores_agg | inlp (~30-dir) | 0.492 | 0.292 | 0.429 | 0.301 |
| flores_agg | D (language-shift) | 0.493 | 0.293 | 0.406 | 0.296 |
| mel | mean (raw) | 0.509 | 0.059 | 0.501 | 0.115 |
| mel | pc (global PCA k=2) | 0.521 | 0.293 | 0.517 | 0.338 |
| mel | inlp (~30-dir) | 0.523 | 0.299 | 0.495 | 0.349 |
| mel | D (language-shift) | 0.509 | 0.298 | 0.501 | 0.329 |
| sst2 | mean (raw) | 0.805 | n/a | 0.765 | n/a |
| sst2 | pc (global PCA k=2) | 0.809 | n/a | 0.762 | n/a |
| sst2 | inlp (~30-dir) | 0.805 | n/a | 0.765 | n/a |
| sst2 | D (language-shift) | 0.805 | n/a | 0.765 | n/a |
| emotion | mean (raw) | 0.365 | n/a | 0.358 | n/a |
| emotion | pc (global PCA k=2) | 0.374 | n/a | 0.399 | n/a |
| emotion | inlp (~30-dir) | 0.365 | n/a | 0.358 | n/a |
| emotion | D (language-shift) | 0.365 | n/a | 0.358 | n/a |
| toxic | mean (raw) | 0.770 | 0.470 | 0.784 | 0.500 |
| toxic | pc (global PCA k=2) | 0.750 | 0.450 | 0.774 | 0.465 |
| toxic | inlp (~30-dir) | 0.767 | 0.482 | 0.788 | 0.494 |
| toxic | D (language-shift) | 0.770 | 0.447 | 0.784 | 0.452 |

## Failed cells

- none

