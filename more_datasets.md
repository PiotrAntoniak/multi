# Out-of-the-box cross-lingual proof extended -- EuroBERT-210m mean pooling + ONE English XGBoost

Features are **mean-pooled** EuroBERT-210m with a **model-free** `perlang_z` transform (L2 + per-language train z-score). English is the sole shared model language; every target test set is scored by that one English model. A **native target-trained model** is reported alongside as a within-language reference. **ratio = target / English self, both under the SAME English model** (the transfer ratio); **comparable when ratio >= 0.9**. `global_z` (English-only stats, no target sample) is a diagnostic.

Skipped: mlsnt: ComplexDataLab/MLSNT is Chinese-only (no English)

## Combined table (headline metric per dataset)

| dataset | metric | target | EN on EN | EN on target | native on target | ratio (target/EN) | native/EN | comparable |
|---|---|---|---|---|---|---|---|---|
| xnli | `macro_f1` | de | 0.440 | 0.395 | 0.415 | 0.898 | 0.943 | NO |
| xnli | `macro_f1` | fr | 0.440 | 0.417 | 0.441 | 0.948 | 1.002 | YES |
| xnli | `macro_f1` | es | 0.440 | 0.417 | 0.441 | 0.949 | 1.003 | YES |
| pawsx | `auc` | de | 0.602 | 0.564 | 0.584 | 0.937 | 0.970 | YES |
| pawsx | `auc` | fr | 0.602 | 0.570 | 0.587 | 0.946 | 0.974 | YES |
| pawsx | `auc` | es | 0.602 | 0.553 | 0.599 | 0.918 | 0.995 | YES |
| textdetox | `auc` | it | 0.973 | 0.604 | 0.858 | 0.621 | 0.883 | NO |
| textdetox | `auc` | de | 0.973 | 0.679 | 0.946 | 0.698 | 0.973 | NO |
| textdetox | `auc` | fr | 0.973 | 0.760 | 0.949 | 0.781 | 0.975 | NO |
| textdetox | `auc` | es | 0.973 | 0.785 | 0.925 | 0.807 | 0.951 | NO |
| sib200 | `macro_f1` | de | 0.773 | 0.698 | 0.792 | 0.903 | 1.025 | YES |
| sib200 | `macro_f1` | fr | 0.773 | 0.769 | 0.771 | 0.995 | 0.998 | YES |
| sib200 | `macro_f1` | es | 0.773 | 0.693 | 0.757 | 0.896 | 0.979 | NO |

**Comparability (perlang_z headline >= 0.9 for every target):** xnli = NO; pawsx = YES; textdetox = NO; sib200 = NO

## xnli (single-label multiclass, 3 classes) -- headline `macro_f1`

spec: facebook/xnli per-language parquet; text = premise + ' </s> ' + hypothesis; label = {0:entailment,1:neutral,2:contradiction}; en train -> de/fr/es.

sizes: en tr=20000 te=5000 | de tr=20000 te=5000 | fr tr=20000 te=5000 | es tr=20000 te=5000

**config `perlang_z`**

| metric | EN self | de (EN-model) | fr (EN-model) | es (EN-model) | de (native) | fr (native) | es (native) | de/EN | fr/EN | es/EN |
|---|---|---|---|---|---|---|---|---|---|---|
| `accuracy` | 0.441 | 0.396 | 0.419 | 0.418 | 0.417 | 0.444 | 0.442 | 0.899 | 0.952 | 0.949 |
| `macro_f1` | 0.440 | 0.395 | 0.417 | 0.417 | 0.415 | 0.441 | 0.441 | 0.898 | 0.948 | 0.949 |
| `auc` | 0.624 | 0.574 | 0.599 | 0.600 | 0.604 | 0.627 | 0.625 | 0.919 | 0.960 | 0.962 |

**config `global_z`**

| metric | EN self | de (EN-model) | fr (EN-model) | es (EN-model) | de (native) | fr (native) | es (native) | de/EN | fr/EN | es/EN |
|---|---|---|---|---|---|---|---|---|---|---|
| `accuracy` | 0.441 | 0.349 | 0.387 | 0.393 | n/a | n/a | n/a | 0.791 | 0.879 | 0.893 |
| `macro_f1` | 0.440 | 0.239 | 0.352 | 0.328 | n/a | n/a | n/a | 0.544 | 0.800 | 0.746 |
| `auc` | 0.624 | 0.556 | 0.563 | 0.575 | n/a | n/a | n/a | 0.890 | 0.902 | 0.921 |


## pawsx (binary, 2 classes) -- headline `auc`

spec: google-research-datasets/paws-x dirs en,de,es,fr; text = sentence1 + ' </s> ' + sentence2; label {0,1}; en train -> de/fr/es.

sizes: en tr=20000 te=2000 | de tr=20000 te=2000 | fr tr=20000 te=2000 | es tr=20000 te=2000

**config `perlang_z`**

| metric | EN self | de (EN-model) | fr (EN-model) | es (EN-model) | de (native) | fr (native) | es (native) | de/EN | fr/EN | es/EN |
|---|---|---|---|---|---|---|---|---|---|---|
| `accuracy` | 0.573 | 0.565 | 0.567 | 0.566 | 0.575 | 0.581 | 0.585 | 0.986 | 0.990 | 0.989 |
| `macro_f1` | 0.560 | 0.529 | 0.540 | 0.530 | 0.528 | 0.544 | 0.549 | 0.945 | 0.964 | 0.947 |
| `auc` | 0.602 | 0.564 | 0.570 | 0.553 | 0.584 | 0.587 | 0.599 | 0.937 | 0.946 | 0.918 |
| `recall_at_fpr5` | 0.122 | 0.070 | 0.066 | 0.082 | 0.102 | 0.105 | 0.090 | 0.575 | 0.543 | 0.667 |
| `recall_at_fpr10` | 0.187 | 0.139 | 0.133 | 0.158 | 0.169 | 0.177 | 0.176 | 0.739 | 0.709 | 0.841 |
| `fpr_at_recall90` | 0.801 | 0.862 | 0.845 | 0.898 | 0.848 | 0.841 | 0.811 | 1.076 | 1.056 | 1.122 |

**config `global_z`**

| metric | EN self | de (EN-model) | fr (EN-model) | es (EN-model) | de (native) | fr (native) | es (native) | de/EN | fr/EN | es/EN |
|---|---|---|---|---|---|---|---|---|---|---|
| `accuracy` | 0.573 | 0.551 | 0.550 | 0.546 | n/a | n/a | n/a | 0.962 | 0.962 | 0.954 |
| `macro_f1` | 0.560 | 0.541 | 0.517 | 0.526 | n/a | n/a | n/a | 0.967 | 0.924 | 0.940 |
| `auc` | 0.602 | 0.557 | 0.550 | 0.543 | n/a | n/a | n/a | 0.926 | 0.914 | 0.903 |
| `recall_at_fpr5` | 0.122 | 0.079 | 0.069 | 0.076 | n/a | n/a | n/a | 0.648 | 0.561 | 0.622 |
| `recall_at_fpr10` | 0.187 | 0.141 | 0.141 | 0.141 | n/a | n/a | n/a | 0.751 | 0.750 | 0.753 |
| `fpr_at_recall90` | 0.801 | 0.865 | 0.868 | 0.879 | n/a | n/a | n/a | 1.081 | 1.084 | 1.098 |


## textdetox (binary, 2 classes) -- headline `auc`

spec: textdetox/multilingual_toxicity_dataset default config; SPLIT NAMES are langs (columns text/toxic); stratified 80/20 train/test per language; en -> it/de/fr/es.

sizes: en tr=4000 te=1000 | it tr=4000 te=1000 | de tr=4000 te=1000 | fr tr=4000 te=1000 | es tr=4000 te=1000

**config `perlang_z`**

| metric | EN self | it (EN-model) | de (EN-model) | fr (EN-model) | es (EN-model) | it (native) | de (native) | fr (native) | es (native) | it/EN | de/EN | fr/EN | es/EN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `accuracy` | 0.911 | 0.562 | 0.643 | 0.702 | 0.711 | 0.767 | 0.877 | 0.873 | 0.840 | 0.617 | 0.706 | 0.771 | 0.780 |
| `macro_f1` | 0.911 | 0.562 | 0.643 | 0.701 | 0.711 | 0.767 | 0.877 | 0.873 | 0.840 | 0.617 | 0.706 | 0.770 | 0.780 |
| `auc` | 0.973 | 0.604 | 0.679 | 0.760 | 0.785 | 0.858 | 0.946 | 0.949 | 0.925 | 0.621 | 0.698 | 0.781 | 0.807 |
| `recall_at_fpr5` | 0.852 | 0.154 | 0.120 | 0.292 | 0.348 | 0.554 | 0.770 | 0.764 | 0.680 | 0.181 | 0.141 | 0.343 | 0.408 |
| `recall_at_fpr10` | 0.912 | 0.216 | 0.234 | 0.384 | 0.454 | 0.640 | 0.852 | 0.862 | 0.772 | 0.237 | 0.257 | 0.421 | 0.498 |
| `fpr_at_recall90` | 0.080 | 0.810 | 0.744 | 0.684 | 0.560 | 0.466 | 0.162 | 0.168 | 0.230 | 10.125 | 9.300 | 8.550 | 7.000 |

**config `global_z`**

| metric | EN self | it (EN-model) | de (EN-model) | fr (EN-model) | es (EN-model) | it (native) | de (native) | fr (native) | es (native) | it/EN | de/EN | fr/EN | es/EN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `accuracy` | 0.911 | 0.568 | 0.582 | 0.670 | 0.674 | n/a | n/a | n/a | n/a | 0.623 | 0.639 | 0.735 | 0.740 |
| `macro_f1` | 0.911 | 0.564 | 0.549 | 0.670 | 0.669 | n/a | n/a | n/a | n/a | 0.620 | 0.602 | 0.735 | 0.734 |
| `auc` | 0.973 | 0.581 | 0.669 | 0.703 | 0.745 | n/a | n/a | n/a | n/a | 0.597 | 0.687 | 0.723 | 0.766 |
| `recall_at_fpr5` | 0.852 | 0.092 | 0.134 | 0.176 | 0.218 | n/a | n/a | n/a | n/a | 0.108 | 0.157 | 0.207 | 0.256 |
| `recall_at_fpr10` | 0.912 | 0.164 | 0.226 | 0.270 | 0.348 | n/a | n/a | n/a | n/a | 0.180 | 0.248 | 0.296 | 0.382 |
| `fpr_at_recall90` | 0.080 | 0.838 | 0.770 | 0.758 | 0.612 | n/a | n/a | n/a | n/a | 10.475 | 9.625 | 9.475 | 7.650 |


## sib200 (single-label multiclass, 7 classes) -- headline `macro_f1`

spec: Davlan/sib200 per-language TSV (cols index_id/category/text); single-label topic; en eng_Latn -> deu/fra/spa Latn; full train/test (701/204).

sizes: en tr=701 te=204 | de tr=701 te=204 | fr tr=701 te=204 | es tr=701 te=204

**config `perlang_z`**

| metric | EN self | de (EN-model) | fr (EN-model) | es (EN-model) | de (native) | fr (native) | es (native) | de/EN | fr/EN | es/EN |
|---|---|---|---|---|---|---|---|---|---|---|
| `accuracy` | 0.794 | 0.716 | 0.784 | 0.716 | 0.814 | 0.794 | 0.779 | 0.901 | 0.988 | 0.901 |
| `macro_f1` | 0.773 | 0.698 | 0.769 | 0.693 | 0.792 | 0.771 | 0.757 | 0.903 | 0.995 | 0.896 |
| `auc` | 0.969 | 0.917 | 0.946 | 0.940 | 0.956 | 0.955 | 0.956 | 0.947 | 0.977 | 0.971 |

**config `global_z`**

| metric | EN self | de (EN-model) | fr (EN-model) | es (EN-model) | de (native) | fr (native) | es (native) | de/EN | fr/EN | es/EN |
|---|---|---|---|---|---|---|---|---|---|---|
| `accuracy` | 0.794 | 0.593 | 0.569 | 0.520 | n/a | n/a | n/a | 0.747 | 0.716 | 0.654 |
| `macro_f1` | 0.773 | 0.522 | 0.519 | 0.442 | n/a | n/a | n/a | 0.675 | 0.671 | 0.572 |
| `auc` | 0.969 | 0.888 | 0.927 | 0.885 | n/a | n/a | n/a | 0.917 | 0.957 | 0.914 |

