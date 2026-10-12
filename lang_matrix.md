# Language transfer matrices (train L -> test L') -- EuroBERT mean pooling

Every language is trained and every language is tested (held-out **test** split; train-on-`L` evaluate-on-`L'`). Features = EuroBERT **mean** pooling + the **`perlang_z`** transform (per language: L2-normalize, then fit mean/std on that language's own TRAIN rows and apply to its own rows -- each language self-normalizes; the trained model sees its own language's stats, the evaluated language uses its own). XGBoost fixed params (`hist, max_bin=64, cpu, n_jobs=2, seed=0, depth8, mcw5, lr0.1, n400`). Trained on train(+val); **macro-F1** is the primary cell, **AUC** the secondary. Rows = trained-on language, cols = evaluated-on language; **diagonal = in-language**. English-only datasets skipped: sst2 (English-only); emotion (English-only).

Train/val are capped at 20000/5000 rows per language (seed-0 permutation subset) for tractability; test is the full standard split. textdetox has no predefined split -> 80/10/10 per language (stratified, seed 0).

## Summary -- mean cross-lingual (off-diagonal) vs in-language (diagonal) macro-F1

| dataset | backbone | in-language macro-F1 (diag) | cross-lingual macro-F1 (off-diag) | gap (diag-off) | off-diag AUC | N |
|---|---|---:|---:|---:|---:|---:|
| flores_sent | 210m | 0.461 | 0.377 | 0.084 | 0.846 | 4 |
| flores_sent | 610m | 0.486 | 0.447 | 0.039 | 0.879 | 4 |
| flores_agg | 210m | 0.544 | 0.423 | 0.121 | 0.850 | 4 |
| flores_agg | 610m | 0.510 | 0.464 | 0.046 | 0.884 | 4 |
| mel | 210m | 0.588 | 0.330 | 0.258 | 0.872 | 3 |
| mel | 610m | 0.577 | 0.417 | 0.160 | 0.897 | 3 |
| toxic | 210m | 0.739 | 0.567 | 0.171 | 0.725 | 5 |
| toxic | 610m | 0.750 | 0.598 | 0.152 | 0.752 | 5 |
| xnli | 210m | 0.452 | 0.408 | 0.044 | 0.595 | 4 |
| xnli | 610m | 0.460 | 0.430 | 0.030 | 0.618 | 4 |
| pawsx | 210m | 0.520 | 0.532 | -0.013 | 0.570 | 4 |
| pawsx | 610m | 0.561 | 0.558 | 0.003 | 0.604 | 4 |
| sib200 | 210m | 0.792 | 0.691 | 0.101 | 0.937 | 4 |
| sib200 | 610m | 0.800 | 0.763 | 0.037 | 0.955 | 4 |
| textdetox | 210m | 0.870 | 0.672 | 0.198 | 0.726 | 5 |
| textdetox | 610m | 0.874 | 0.717 | 0.158 | 0.787 | 5 |

## flores_sent (multiclass, 12 classes) -- languages: en, it, de, fr

**macro_f1 / EuroBERT-210m**

| trained \ tested | en | it | de | fr |
|---|---|---|---|---|
| en | **0.483** | 0.363 | 0.351 | 0.359 |
| it | 0.409 | **0.412** | 0.360 | 0.434 |
| de | 0.393 | 0.360 | **0.474** | 0.374 |
| fr | 0.432 | 0.359 | 0.324 | **0.475** |

**macro_f1 / EuroBERT-610m**

| trained \ tested | en | it | de | fr |
|---|---|---|---|---|
| en | **0.512** | 0.448 | 0.417 | 0.461 |
| it | 0.472 | **0.441** | 0.445 | 0.488 |
| de | 0.451 | 0.431 | **0.496** | 0.430 |
| fr | 0.442 | 0.430 | 0.455 | **0.496** |

**auc / EuroBERT-210m**

| trained \ tested | en | it | de | fr |
|---|---|---|---|---|
| en | **0.908** | 0.847 | 0.827 | 0.829 |
| it | 0.874 | **0.877** | 0.833 | 0.829 |
| de | 0.878 | 0.837 | **0.885** | 0.857 |
| fr | 0.867 | 0.845 | 0.833 | **0.865** |

**auc / EuroBERT-610m**

| trained \ tested | en | it | de | fr |
|---|---|---|---|---|
| en | **0.913** | 0.869 | 0.868 | 0.865 |
| it | 0.896 | **0.890** | 0.883 | 0.879 |
| de | 0.905 | 0.875 | **0.892** | 0.868 |
| fr | 0.899 | 0.872 | 0.869 | **0.887** |


## flores_agg (multiclass, 11 classes) -- languages: en, it, de, fr

**macro_f1 / EuroBERT-210m**

| trained \ tested | en | it | de | fr |
|---|---|---|---|---|
| en | **0.564** | 0.376 | 0.449 | 0.399 |
| it | 0.541 | **0.559** | 0.509 | 0.505 |
| de | 0.380 | 0.435 | **0.590** | 0.315 |
| fr | 0.434 | 0.411 | 0.324 | **0.464** |

**macro_f1 / EuroBERT-610m**

| trained \ tested | en | it | de | fr |
|---|---|---|---|---|
| en | **0.466** | 0.427 | 0.391 | 0.473 |
| it | 0.511 | **0.539** | 0.426 | 0.431 |
| de | 0.521 | 0.470 | **0.476** | 0.498 |
| fr | 0.476 | 0.505 | 0.442 | **0.559** |

**auc / EuroBERT-210m**

| trained \ tested | en | it | de | fr |
|---|---|---|---|---|
| en | **0.913** | 0.827 | 0.864 | 0.833 |
| it | 0.885 | **0.892** | 0.876 | 0.873 |
| de | 0.848 | 0.835 | **0.890** | 0.824 |
| fr | 0.831 | 0.871 | 0.836 | **0.884** |

**auc / EuroBERT-610m**

| trained \ tested | en | it | de | fr |
|---|---|---|---|---|
| en | **0.888** | 0.896 | 0.844 | 0.873 |
| it | 0.881 | **0.882** | 0.869 | 0.886 |
| de | 0.898 | 0.897 | **0.892** | 0.873 |
| fr | 0.892 | 0.900 | 0.900 | **0.918** |


## mel (multilabel, 21 classes) -- languages: en, de, fr

**macro_f1 / EuroBERT-210m**

| trained \ tested | en | de | fr |
|---|---|---|---|
| en | **0.606** | 0.341 | 0.354 |
| de | 0.335 | **0.591** | 0.263 |
| fr | 0.366 | 0.322 | **0.566** |

**macro_f1 / EuroBERT-610m**

| trained \ tested | en | de | fr |
|---|---|---|---|
| en | **0.600** | 0.371 | 0.449 |
| de | 0.441 | **0.567** | 0.393 |
| fr | 0.465 | 0.381 | **0.562** |

**auc / EuroBERT-210m**

| trained \ tested | en | de | fr |
|---|---|---|---|
| en | **0.929** | 0.866 | 0.881 |
| de | 0.866 | **0.922** | 0.859 |
| fr | 0.892 | 0.870 | **0.924** |

**auc / EuroBERT-610m**

| trained \ tested | en | de | fr |
|---|---|---|---|
| en | **0.928** | 0.889 | 0.891 |
| de | 0.905 | **0.919** | 0.891 |
| fr | 0.910 | 0.896 | **0.924** |


## toxic (binary, 2 classes) -- languages: en, it, de, fr, es

**macro_f1 / EuroBERT-210m**

| trained \ tested | en | it | de | fr | es |
|---|---|---|---|---|---|
| en | **0.763** | 0.426 | 0.424 | 0.412 | 0.424 |
| it | 0.633 | **0.703** | 0.596 | 0.591 | 0.566 |
| de | 0.676 | 0.582 | **0.770** | 0.647 | 0.579 |
| fr | 0.666 | 0.649 | 0.644 | **0.774** | 0.626 |
| es | 0.594 | 0.567 | 0.518 | 0.523 | **0.683** |

**macro_f1 / EuroBERT-610m**

| trained \ tested | en | it | de | fr | es |
|---|---|---|---|---|---|
| en | **0.769** | 0.426 | 0.436 | 0.417 | 0.435 |
| it | 0.691 | **0.718** | 0.596 | 0.628 | 0.605 |
| de | 0.699 | 0.634 | **0.799** | 0.689 | 0.612 |
| fr | 0.702 | 0.662 | 0.676 | **0.782** | 0.643 |
| es | 0.659 | 0.586 | 0.592 | 0.570 | **0.681** |

**auc / EuroBERT-210m**

| trained \ tested | en | it | de | fr | es |
|---|---|---|---|---|---|
| en | **0.903** | 0.634 | 0.664 | 0.756 | 0.634 |
| it | 0.794 | **0.833** | 0.751 | 0.780 | 0.710 |
| de | 0.784 | 0.711 | **0.876** | 0.764 | 0.694 |
| fr | 0.831 | 0.711 | 0.710 | **0.873** | 0.703 |
| es | 0.727 | 0.700 | 0.702 | 0.736 | **0.801** |

**auc / EuroBERT-610m**

| trained \ tested | en | it | de | fr | es |
|---|---|---|---|---|---|
| en | **0.914** | 0.702 | 0.678 | 0.741 | 0.696 |
| it | 0.819 | **0.847** | 0.744 | 0.807 | 0.726 |
| de | 0.816 | 0.723 | **0.886** | 0.784 | 0.703 |
| fr | 0.865 | 0.759 | 0.757 | **0.876** | 0.728 |
| es | 0.789 | 0.729 | 0.726 | 0.748 | **0.813** |


## xnli (multiclass, 3 classes) -- languages: en, de, fr, es

**macro_f1 / EuroBERT-210m**

| trained \ tested | en | de | fr | es |
|---|---|---|---|---|
| en | **0.451** | 0.401 | 0.417 | 0.422 |
| de | 0.409 | **0.442** | 0.418 | 0.406 |
| fr | 0.405 | 0.400 | **0.464** | 0.417 |
| es | 0.418 | 0.374 | 0.406 | **0.450** |

**macro_f1 / EuroBERT-610m**

| trained \ tested | en | de | fr | es |
|---|---|---|---|---|
| en | **0.453** | 0.411 | 0.432 | 0.436 |
| de | 0.435 | **0.458** | 0.439 | 0.429 |
| fr | 0.426 | 0.406 | **0.473** | 0.433 |
| es | 0.437 | 0.434 | 0.442 | **0.456** |

**auc / EuroBERT-210m**

| trained \ tested | en | de | fr | es |
|---|---|---|---|---|
| en | **0.643** | 0.590 | 0.600 | 0.606 |
| de | 0.595 | **0.627** | 0.597 | 0.585 |
| fr | 0.595 | 0.579 | **0.644** | 0.599 |
| es | 0.612 | 0.580 | 0.603 | **0.637** |

**auc / EuroBERT-610m**

| trained \ tested | en | de | fr | es |
|---|---|---|---|---|
| en | **0.649** | 0.604 | 0.623 | 0.619 |
| de | 0.620 | **0.637** | 0.621 | 0.621 |
| fr | 0.613 | 0.609 | **0.652** | 0.621 |
| es | 0.621 | 0.618 | 0.626 | **0.638** |


## pawsx (binary, 2 classes) -- languages: en, de, fr, es

**macro_f1 / EuroBERT-210m**

| trained \ tested | en | de | fr | es |
|---|---|---|---|---|
| en | **0.510** | 0.518 | 0.529 | 0.514 |
| de | 0.549 | **0.524** | 0.531 | 0.553 |
| fr | 0.511 | 0.521 | **0.527** | 0.526 |
| es | 0.534 | 0.549 | 0.551 | **0.517** |

**macro_f1 / EuroBERT-610m**

| trained \ tested | en | de | fr | es |
|---|---|---|---|---|
| en | **0.575** | 0.545 | 0.561 | 0.570 |
| de | 0.562 | **0.543** | 0.542 | 0.552 |
| fr | 0.572 | 0.559 | **0.559** | 0.550 |
| es | 0.573 | 0.556 | 0.553 | **0.567** |

**auc / EuroBERT-210m**

| trained \ tested | en | de | fr | es |
|---|---|---|---|---|
| en | **0.530** | 0.563 | 0.571 | 0.568 |
| de | 0.561 | **0.556** | 0.567 | 0.578 |
| fr | 0.538 | 0.561 | **0.563** | 0.595 |
| es | 0.560 | 0.585 | 0.588 | **0.559** |

**auc / EuroBERT-610m**

| trained \ tested | en | de | fr | es |
|---|---|---|---|---|
| en | **0.628** | 0.599 | 0.603 | 0.607 |
| de | 0.605 | **0.592** | 0.595 | 0.607 |
| fr | 0.609 | 0.589 | **0.606** | 0.602 |
| es | 0.611 | 0.603 | 0.613 | **0.621** |


## sib200 (multiclass, 7 classes) -- languages: en, de, fr, es

**macro_f1 / EuroBERT-210m**

| trained \ tested | en | de | fr | es |
|---|---|---|---|---|
| en | **0.804** | 0.661 | 0.726 | 0.688 |
| de | 0.699 | **0.823** | 0.663 | 0.682 |
| fr | 0.700 | 0.645 | **0.786** | 0.671 |
| es | 0.748 | 0.681 | 0.725 | **0.755** |

**macro_f1 / EuroBERT-610m**

| trained \ tested | en | de | fr | es |
|---|---|---|---|---|
| en | **0.803** | 0.766 | 0.738 | 0.752 |
| de | 0.772 | **0.813** | 0.758 | 0.720 |
| fr | 0.801 | 0.785 | **0.788** | 0.800 |
| es | 0.796 | 0.707 | 0.760 | **0.795** |

**auc / EuroBERT-210m**

| trained \ tested | en | de | fr | es |
|---|---|---|---|---|
| en | **0.975** | 0.927 | 0.949 | 0.946 |
| de | 0.947 | **0.966** | 0.923 | 0.936 |
| fr | 0.951 | 0.909 | **0.957** | 0.946 |
| es | 0.948 | 0.925 | 0.944 | **0.957** |

**auc / EuroBERT-610m**

| trained \ tested | en | de | fr | es |
|---|---|---|---|---|
| en | **0.953** | 0.964 | 0.944 | 0.942 |
| de | 0.960 | **0.969** | 0.936 | 0.945 |
| fr | 0.970 | 0.969 | **0.959** | 0.964 |
| es | 0.960 | 0.959 | 0.948 | **0.962** |


## textdetox (binary, 2 classes) -- languages: en, it, de, fr, es

**macro_f1 / EuroBERT-210m**

| trained \ tested | en | it | de | fr | es |
|---|---|---|---|---|---|
| en | **0.928** | 0.590 | 0.630 | 0.661 | 0.666 |
| it | 0.722 | **0.802** | 0.570 | 0.716 | 0.730 |
| de | 0.677 | 0.591 | **0.890** | 0.702 | 0.608 |
| fr | 0.774 | 0.648 | 0.649 | **0.868** | 0.731 |
| es | 0.779 | 0.608 | 0.651 | 0.730 | **0.862** |

**macro_f1 / EuroBERT-610m**

| trained \ tested | en | it | de | fr | es |
|---|---|---|---|---|---|
| en | **0.928** | 0.594 | 0.698 | 0.726 | 0.794 |
| it | 0.776 | **0.780** | 0.635 | 0.749 | 0.761 |
| de | 0.717 | 0.619 | **0.904** | 0.712 | 0.674 |
| fr | 0.768 | 0.690 | 0.744 | **0.896** | 0.774 |
| es | 0.801 | 0.634 | 0.674 | 0.792 | **0.864** |

**auc / EuroBERT-210m**

| trained \ tested | en | it | de | fr | es |
|---|---|---|---|---|---|
| en | **0.982** | 0.627 | 0.680 | 0.738 | 0.732 |
| it | 0.798 | **0.878** | 0.615 | 0.761 | 0.782 |
| de | 0.744 | 0.618 | **0.954** | 0.764 | 0.640 |
| fr | 0.833 | 0.708 | 0.705 | **0.946** | 0.836 |
| es | 0.844 | 0.639 | 0.669 | 0.791 | **0.940** |

**auc / EuroBERT-610m**

| trained \ tested | en | it | de | fr | es |
|---|---|---|---|---|---|
| en | **0.976** | 0.652 | 0.769 | 0.818 | 0.849 |
| it | 0.849 | **0.865** | 0.676 | 0.840 | 0.845 |
| de | 0.791 | 0.666 | **0.971** | 0.783 | 0.727 |
| fr | 0.840 | 0.756 | 0.823 | **0.963** | 0.846 |
| es | 0.882 | 0.699 | 0.744 | 0.885 | **0.943** |


## Overall backbone comparison (mean over datasets)

| backbone | mean in-language (diag) | mean cross-lingual (off-diag) | mean gap |
|---|---:|---:|---:|
| 210m | 0.621 | 0.500 | 0.121 |
| 610m | 0.627 | 0.549 | 0.078 |

