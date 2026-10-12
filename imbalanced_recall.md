# imbalanced_recall -- English-trained vs native-trained at high recall

Recipe: EuroBERT-210m **mean** pool -> `l2_thenZ [both]` -> OVR `XGBClassifier(binary:logistic, hist, max_bin=64, device=cpu, n_jobs=2, seed=0, max_depth=8, min_child_weight=5, lr=0.1, n_estimators=400)`. **English-trained** = fit on EN train, score target test; **native** = fit on the target-language train, score target test. Comparable tolerance (headline): EN within 0.050 of native.

Lower is better only for `FPR@R90`; for every other column **higher is better**.

## Dataset: `sent`

Rare classes (total counts): business and economy 25, health 68, geography 87, safety 97.

### `business and economy`  (train_pos=18, test_pos=4)

| metric | it EN / nat | de EN / nat | fr EN / nat |
|---|---|---|---|
| AveragePrecision | 0.038 / 0.160 | 0.031 / 0.363 | 0.077 / 0.311 |
| R@FPR1% | 0.000 / 0.250 | 0.000 / 0.500 | 0.000 / 0.250 |
| R@FPR5% | 0.000 / 0.250 | 0.000 / 0.750 | 0.500 / 0.500 |
| R@FPR10% | 0.500 / 0.500 | 0.250 / 0.750 | 0.500 / 0.750 |
| R@FA5 | 0.000 / 0.250 | 0.000 / 0.500 | 0.000 / 0.250 |
| R@FA20 | 0.000 / 0.250 | 0.000 / 0.750 | 0.500 / 0.500 |
| R@FA50 | 0.500 / 0.500 | 0.250 / 0.750 | 0.500 / 0.750 |
| R@FA100 | 0.750 / 0.750 | 0.750 / 0.750 | 0.750 / 0.750 |
| FPR@R90 | 0.224 / 0.230 | 0.351 / 0.597 | 0.240 / 0.645 |
| maxF2 | 0.192 / 0.278 | 0.156 / 0.556 | 0.333 / 0.357 |
| maxF3 | 0.306 / 0.294 | 0.259 / 0.638 | 0.400 / 0.417 |

High-recall comparison (tolerate many false alarms):

| target | R@FA50 EN / nat | R@FA100 EN / nat | FPR@R90 EN / nat |
|---|---|---|---|
| it | 0.500 / 0.500 | 0.750 / 0.750 | 0.224 / 0.230 |
| de | 0.250 / 0.750 | 0.750 / 0.750 | 0.351 / 0.597 |
| fr | 0.500 / 0.750 | 0.750 / 0.750 | 0.240 / 0.645 |

### `health`  (train_pos=32, test_pos=20)

| metric | it EN / nat | de EN / nat | fr EN / nat |
|---|---|---|---|
| AveragePrecision | 0.222 / 0.275 | 0.186 / 0.232 | 0.112 / 0.207 |
| R@FPR1% | 0.100 / 0.100 | 0.100 / 0.150 | 0.050 / 0.100 |
| R@FPR5% | 0.350 / 0.500 | 0.250 / 0.450 | 0.200 / 0.350 |
| R@FPR10% | 0.400 / 0.600 | 0.350 / 0.600 | 0.300 / 0.550 |
| R@FA5 | 0.100 / 0.150 | 0.100 / 0.200 | 0.050 / 0.150 |
| R@FA20 | 0.300 / 0.500 | 0.250 / 0.450 | 0.150 / 0.350 |
| R@FA50 | 0.400 / 0.600 | 0.350 / 0.600 | 0.350 / 0.550 |
| R@FA100 | 0.650 / 0.700 | 0.500 / 0.700 | 0.500 / 0.700 |
| FPR@R90 | 0.528 / 0.445 | 0.724 / 0.540 | 0.614 / 0.489 |
| maxF2 | 0.374 / 0.465 | 0.287 / 0.461 | 0.292 / 0.412 |
| maxF3 | 0.475 / 0.524 | 0.375 / 0.556 | 0.397 / 0.519 |

High-recall comparison (tolerate many false alarms):

| target | R@FA50 EN / nat | R@FA100 EN / nat | FPR@R90 EN / nat |
|---|---|---|---|
| it | 0.400 / 0.600 | 0.650 / 0.700 | 0.528 / 0.445 |
| de | 0.350 / 0.600 | 0.500 / 0.700 | 0.724 / 0.540 |
| fr | 0.350 / 0.550 | 0.500 / 0.700 | 0.614 / 0.489 |

### `geography`  (train_pos=44, test_pos=20)

| metric | it EN / nat | de EN / nat | fr EN / nat |
|---|---|---|---|
| AveragePrecision | 0.069 / 0.115 | 0.056 / 0.075 | 0.121 / 0.096 |
| R@FPR1% | 0.000 / 0.050 | 0.000 / 0.000 | 0.100 / 0.000 |
| R@FPR5% | 0.100 / 0.150 | 0.100 / 0.100 | 0.150 / 0.150 |
| R@FPR10% | 0.200 / 0.300 | 0.100 / 0.150 | 0.200 / 0.300 |
| R@FA5 | 0.000 / 0.100 | 0.000 / 0.000 | 0.100 / 0.000 |
| R@FA20 | 0.050 / 0.150 | 0.100 / 0.100 | 0.150 / 0.150 |
| R@FA50 | 0.200 / 0.350 | 0.100 / 0.150 | 0.200 / 0.300 |
| R@FA100 | 0.350 / 0.500 | 0.350 / 0.450 | 0.300 / 0.500 |
| FPR@R90 | 0.588 / 0.423 | 0.777 / 0.664 | 0.864 / 0.590 |
| maxF2 | 0.239 / 0.308 | 0.208 / 0.287 | 0.230 / 0.281 |
| maxF3 | 0.373 / 0.466 | 0.329 / 0.388 | 0.332 / 0.422 |

High-recall comparison (tolerate many false alarms):

| target | R@FA50 EN / nat | R@FA100 EN / nat | FPR@R90 EN / nat |
|---|---|---|---|
| it | 0.200 / 0.350 | 0.350 / 0.500 | 0.588 / 0.423 |
| de | 0.100 / 0.150 | 0.350 / 0.450 | 0.777 / 0.664 |
| fr | 0.200 / 0.300 | 0.300 / 0.500 | 0.864 / 0.590 |

### `safety`  (train_pos=64, test_pos=8)

| metric | it EN / nat | de EN / nat | fr EN / nat |
|---|---|---|---|
| AveragePrecision | 0.321 / 0.177 | 0.103 / 0.187 | 0.490 / 0.243 |
| R@FPR1% | 0.250 / 0.125 | 0.000 / 0.125 | 0.500 / 0.250 |
| R@FPR5% | 0.625 / 0.375 | 0.375 / 0.500 | 0.750 / 0.750 |
| R@FPR10% | 0.750 / 0.750 | 0.375 / 0.750 | 0.750 / 0.750 |
| R@FA5 | 0.250 / 0.125 | 0.000 / 0.125 | 0.500 / 0.250 |
| R@FA20 | 0.625 / 0.375 | 0.375 / 0.500 | 0.750 / 0.750 |
| R@FA50 | 0.750 / 0.750 | 0.375 / 0.750 | 0.875 / 0.750 |
| R@FA100 | 1.000 / 1.000 | 0.625 / 1.000 | 1.000 / 0.875 |
| FPR@R90 | 0.175 / 0.151 | 0.443 / 0.173 | 0.161 / 0.620 |
| maxF2 | 0.510 / 0.390 | 0.341 / 0.435 | 0.600 / 0.600 |
| maxF3 | 0.562 / 0.516 | 0.357 / 0.511 | 0.667 / 0.667 |

High-recall comparison (tolerate many false alarms):

| target | R@FA50 EN / nat | R@FA100 EN / nat | FPR@R90 EN / nat |
|---|---|---|---|
| it | 0.750 / 0.750 | 1.000 / 1.000 | 0.175 / 0.151 |
| de | 0.375 / 0.750 | 0.625 / 1.000 | 0.443 / 0.173 |
| fr | 0.875 / 0.750 | 1.000 / 0.875 | 0.161 / 0.620 |

## Dataset: `agg`

Rare classes (11-class, business dropped): health, geography, safety, education and society.

### `health`  (train_pos=11, test_pos=4)

| metric | it EN / nat | de EN / nat | fr EN / nat |
|---|---|---|---|
| AveragePrecision | 0.578 / 0.304 | 0.338 / 0.281 | 0.224 / 0.043 |
| R@FPR1% | 0.500 / 0.250 | 0.250 / 0.250 | 0.250 / 0.000 |
| R@FPR5% | 0.500 / 0.250 | 0.250 / 0.250 | 0.250 / 0.000 |
| R@FPR10% | 0.750 / 0.250 | 0.250 / 0.250 | 0.500 / 0.000 |
| R@FA5 | 0.500 / 0.250 | 0.250 / 0.250 | 0.250 / 0.000 |
| R@FA20 | 0.750 / 0.250 | 0.750 / 0.250 | 0.500 / 0.000 |
| R@FA50 | 1.000 / 0.750 | 1.000 / 0.500 | 1.000 / 0.500 |
| R@FA100 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| FPR@R90 | 0.237 / 0.563 | 0.200 / 0.674 | 0.348 / 0.541 |
| maxF2 | 0.556 / 0.294 | 0.426 / 0.294 | 0.400 / 0.215 |
| maxF3 | 0.588 / 0.405 | 0.597 / 0.305 | 0.460 / 0.354 |

High-recall comparison (tolerate many false alarms):

| target | R@FA50 EN / nat | R@FA100 EN / nat | FPR@R90 EN / nat |
|---|---|---|---|
| it | 1.000 / 0.750 | 1.000 / 1.000 | 0.237 / 0.563 |
| de | 1.000 / 0.500 | 1.000 / 1.000 | 0.200 / 0.674 |
| fr | 1.000 / 0.500 | 1.000 / 1.000 | 0.348 / 0.541 |

### `geography`  (train_pos=14, test_pos=4)

| metric | it EN / nat | de EN / nat | fr EN / nat |
|---|---|---|---|
| AveragePrecision | 0.363 / 0.144 | 0.076 / 0.117 | 0.219 / 0.291 |
| R@FPR1% | 0.250 / 0.000 | 0.000 / 0.000 | 0.250 / 0.000 |
| R@FPR5% | 0.500 / 0.000 | 0.000 / 0.000 | 0.250 / 0.500 |
| R@FPR10% | 0.500 / 0.750 | 0.250 / 0.250 | 0.500 / 0.750 |
| R@FA5 | 0.250 / 0.000 | 0.000 / 0.000 | 0.250 / 0.500 |
| R@FA20 | 0.750 / 0.750 | 0.500 / 0.750 | 0.500 / 0.750 |
| R@FA50 | 0.750 / 1.000 | 0.500 / 1.000 | 0.750 / 1.000 |
| R@FA100 | 1.000 / 1.000 | 0.750 / 1.000 | 1.000 / 1.000 |
| FPR@R90 | 0.622 / 0.178 | 0.785 / 0.304 | 0.481 / 0.200 |
| maxF2 | 0.429 / 0.469 | 0.286 / 0.385 | 0.400 / 0.577 |
| maxF3 | 0.545 / 0.625 | 0.364 / 0.508 | 0.448 / 0.652 |

High-recall comparison (tolerate many false alarms):

| target | R@FA50 EN / nat | R@FA100 EN / nat | FPR@R90 EN / nat |
|---|---|---|---|
| it | 0.750 / 1.000 | 1.000 / 1.000 | 0.622 / 0.178 |
| de | 0.500 / 1.000 | 0.750 / 1.000 | 0.785 / 0.304 |
| fr | 0.750 / 1.000 | 1.000 / 1.000 | 0.481 / 0.200 |

### `safety`  (train_pos=18, test_pos=5)

| metric | it EN / nat | de EN / nat | fr EN / nat |
|---|---|---|---|
| AveragePrecision | 0.262 / 0.391 | 0.249 / 0.370 | 0.233 / 0.365 |
| R@FPR1% | 0.200 / 0.400 | 0.200 / 0.200 | 0.200 / 0.200 |
| R@FPR5% | 0.400 / 0.600 | 0.200 / 0.400 | 0.400 / 0.400 |
| R@FPR10% | 0.600 / 0.800 | 0.200 / 0.600 | 0.600 / 0.600 |
| R@FA5 | 0.400 / 0.600 | 0.200 / 0.400 | 0.200 / 0.400 |
| R@FA20 | 0.600 / 0.800 | 0.200 / 0.600 | 0.800 / 0.600 |
| R@FA50 | 1.000 / 0.800 | 0.600 / 0.800 | 0.800 / 0.600 |
| R@FA100 | 1.000 / 1.000 | 0.800 / 0.800 | 0.800 / 1.000 |
| FPR@R90 | 0.321 / 0.724 | 0.784 / 0.940 | 0.791 / 0.507 |
| maxF2 | 0.500 / 0.645 | 0.244 / 0.429 | 0.476 / 0.500 |
| maxF3 | 0.545 / 0.714 | 0.374 / 0.500 | 0.597 / 0.545 |

High-recall comparison (tolerate many false alarms):

| target | R@FA50 EN / nat | R@FA100 EN / nat | FPR@R90 EN / nat |
|---|---|---|---|
| it | 1.000 / 0.800 | 1.000 / 1.000 | 0.321 / 0.724 |
| de | 0.600 / 0.800 | 0.800 / 0.800 | 0.784 / 0.940 |
| fr | 0.800 / 0.600 | 0.800 / 1.000 | 0.791 / 0.507 |

### `education and society`  (train_pos=10, test_pos=7)

| metric | it EN / nat | de EN / nat | fr EN / nat |
|---|---|---|---|
| AveragePrecision | 0.050 / 0.050 | 0.050 / 0.050 | 0.050 / 0.050 |
| R@FPR1% | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| R@FPR5% | 0.143 / 0.143 | 0.143 / 0.143 | 0.143 / 0.143 |
| R@FPR10% | 0.143 / 0.143 | 0.143 / 0.143 | 0.143 / 0.143 |
| R@FA5 | 0.143 / 0.143 | 0.143 / 0.143 | 0.143 / 0.143 |
| R@FA20 | 0.143 / 0.143 | 0.143 / 0.143 | 0.143 / 0.143 |
| R@FA50 | 0.286 / 0.286 | 0.286 / 0.286 | 0.286 / 0.286 |
| R@FA100 | 0.857 / 0.857 | 0.857 / 0.857 | 0.857 / 0.857 |
| FPR@R90 | 0.924 / 0.924 | 0.924 / 0.924 | 0.924 / 0.924 |
| maxF2 | 0.240 / 0.240 | 0.240 / 0.240 | 0.240 / 0.240 |
| maxF3 | 0.375 / 0.375 | 0.375 / 0.375 | 0.375 / 0.375 |

High-recall comparison (tolerate many false alarms):

| target | R@FA50 EN / nat | R@FA100 EN / nat | FPR@R90 EN / nat |
|---|---|---|---|
| it | 0.286 / 0.286 | 0.857 / 0.857 | 0.924 / 0.924 |
| de | 0.286 / 0.286 | 0.857 / 0.857 | 0.924 / 0.924 |
| fr | 0.286 / 0.286 | 0.857 / 0.857 | 0.924 / 0.924 |

## Headline

- Cells evaluated (rare-class x target-language): **24**.
- English-trained Average Precision within 0.05 of native in **14/24** cells (58%).
- **High-recall regime** (R@FPR10pct + R@FA50 + R@FA100 all within 0.05 of native): **9/24** cells (38%).  This is the regime the task cares about -- many false alarms tolerated.
- English-trained recall (ALL operating points) within 0.05 of native in **8/24** cells (33%) -- strict, and dominated by the low-FPR points where a handful of positives makes 0.05 unattainable.

### False-alarm cost at high recall

- Mean recall at 100 false alarms: EN **0.769** vs native **0.835** (across 24 cells).
- Mean FPR needed to reach recall=0.90: EN **0.543** vs native **0.540** (across 24 cells).  A smaller number is better; the difference is the extra false-alarm rate the English model spends for the same 90% recall.

Read the `R@FA*` and `FPR@R90` columns: they state how many false alarms the detector spends to reach high recall. Under the 'missing a positive is unacceptable' rule, English-trained and native-tracked curves are compared at the SAME recall, so a small EN penalty in `FPR@R90` is the extra false-alarm cost of using the English model on target text.
