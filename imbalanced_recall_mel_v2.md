# imbalanced_recall_mel_v2 -- MultiEURLEX: English-trained vs native-trained at HIGH RECALL

> **v1 label bug (fixed in v2):** MultiEURLEX TRAIN splits are NOT parallel across languages, but the v1 `load_mel` trained the native de/fr models with the ENGLISH train labels, so the native column in `imbalanced_recall_mel.md` was near-random. v2 loads each language's OWN train labels (the TEST split is parallel, so its labels are shared).

Long, non-English EU legal documents (`nlpaueb/multi_eurlex`, en/de/fr; train/test = 11000/5000).  Recipe: EuroBERT-210m **mean** pool, 512-token truncation, batch 64 -> `l2_thenZ [both]` -> OVR `XGBClassifier(binary:logistic, hist, max_bin=64, device=cpu, n_jobs=2, seed=0, max_depth=8, min_child_weight=5, lr=0.1, n_estimators=400)`.  **English-trained** = fit on EN train, score target test; **native** = fit on the target-language train, score target test.

Rare concepts = the 8 lowest-TRAIN-frequency EuroVoc level-1 concepts present in test:

| idx | EuroVoc id | concept | train_pos | test_pos |
|---|---|---|---|---|
| 13 | 100162 | international organisations | 108 | 102 |
| 16 | 100151 | science | 180 | 57 |
| 14 | 100159 | energy | 221 | 185 |
| 9 | 100153 | employment and working conditions | 341 | 164 |
| 11 | 100145 | law | 391 | 531 |
| 10 | 100142 | politics | 408 | 335 |
| 20 | 100155 | environment | 552 | 610 |
| 8 | 100154 | transport | 582 | 600 |

Lower is better only for `FPR@R90`; every other column is **higher is better**.  Cells read `EN / native`.  Comparable tolerance (headline): EN within 0.050 of native.

## `13` international organisations  (train_pos=108, test_pos=102, EuroVoc 100162)

| metric | de EN / native | fr EN / native |
|---|---|---|
| AP | 0.094 / 0.145 | 0.104 / 0.143 |
| R@FPR1% | 0.098 / 0.147 | 0.078 / 0.127 |
| R@FPR5% | 0.265 / 0.373 | 0.363 / 0.363 |
| R@FPR10% | 0.373 / 0.549 | 0.471 / 0.549 |
| R@FA5 | 0.020 / 0.020 | 0.020 / 0.059 |
| R@FA20 | 0.049 / 0.098 | 0.049 / 0.088 |
| R@FA50 | 0.098 / 0.147 | 0.088 / 0.127 |
| R@FA100 | 0.167 / 0.216 | 0.167 / 0.176 |
| FPR@R90 | 0.508 / 0.484 | 0.422 / 0.309 |
| maxF2 | 0.221 / 0.310 | 0.273 / 0.317 |
| maxF3 | 0.316 / 0.417 | 0.370 / 0.430 |

## `16` science  (train_pos=180, test_pos=57, EuroVoc 100151)

| metric | de EN / native | fr EN / native |
|---|---|---|
| AP | 0.058 / 0.050 | 0.071 / 0.057 |
| R@FPR1% | 0.070 / 0.035 | 0.088 / 0.123 |
| R@FPR5% | 0.333 / 0.316 | 0.404 / 0.316 |
| R@FPR10% | 0.526 / 0.404 | 0.561 / 0.491 |
| R@FA5 | 0.000 / 0.018 | 0.000 / 0.000 |
| R@FA20 | 0.035 / 0.018 | 0.035 / 0.035 |
| R@FA50 | 0.088 / 0.035 | 0.088 / 0.123 |
| R@FA100 | 0.140 / 0.123 | 0.281 / 0.175 |
| FPR@R90 | 0.401 / 0.776 | 0.539 / 0.507 |
| maxF2 | 0.207 / 0.203 | 0.235 / 0.200 |
| maxF3 | 0.298 / 0.251 | 0.323 / 0.284 |

## `14` energy  (train_pos=221, test_pos=185, EuroVoc 100159)

| metric | de EN / native | fr EN / native |
|---|---|---|
| AP | 0.509 / 0.546 | 0.448 / 0.509 |
| R@FPR1% | 0.438 / 0.492 | 0.368 / 0.427 |
| R@FPR5% | 0.616 / 0.681 | 0.622 / 0.632 |
| R@FPR10% | 0.697 / 0.746 | 0.703 / 0.697 |
| R@FA5 | 0.151 / 0.178 | 0.141 / 0.195 |
| R@FA20 | 0.335 / 0.346 | 0.314 / 0.297 |
| R@FA50 | 0.449 / 0.514 | 0.378 / 0.427 |
| R@FA100 | 0.535 / 0.584 | 0.465 / 0.535 |
| FPR@R90 | 0.422 / 0.274 | 0.414 / 0.418 |
| maxF2 | 0.542 / 0.592 | 0.530 / 0.555 |
| maxF3 | 0.587 / 0.634 | 0.582 / 0.587 |

## `9` employment and working conditions  (train_pos=341, test_pos=164, EuroVoc 100153)

| metric | de EN / native | fr EN / native |
|---|---|---|
| AP | 0.214 / 0.210 | 0.137 / 0.193 |
| R@FPR1% | 0.183 / 0.067 | 0.037 / 0.091 |
| R@FPR5% | 0.384 / 0.537 | 0.305 / 0.445 |
| R@FPR10% | 0.518 / 0.671 | 0.543 / 0.701 |
| R@FA5 | 0.037 / 0.012 | 0.000 / 0.012 |
| R@FA20 | 0.110 / 0.024 | 0.006 / 0.030 |
| R@FA50 | 0.183 / 0.067 | 0.043 / 0.098 |
| R@FA100 | 0.262 / 0.195 | 0.134 / 0.165 |
| FPR@R90 | 0.375 / 0.306 | 0.362 / 0.281 |
| maxF2 | 0.383 / 0.470 | 0.371 / 0.464 |
| maxF3 | 0.475 / 0.549 | 0.489 / 0.559 |

## `11` law  (train_pos=391, test_pos=531, EuroVoc 100145)

| metric | de EN / native | fr EN / native |
|---|---|---|
| AP | 0.252 / 0.419 | 0.440 / 0.398 |
| R@FPR1% | 0.036 / 0.090 | 0.105 / 0.070 |
| R@FPR5% | 0.186 / 0.326 | 0.435 / 0.335 |
| R@FPR10% | 0.360 / 0.644 | 0.595 / 0.574 |
| R@FA5 | 0.006 / 0.015 | 0.006 / 0.015 |
| R@FA20 | 0.017 / 0.043 | 0.041 / 0.038 |
| R@FA50 | 0.040 / 0.094 | 0.124 / 0.075 |
| R@FA100 | 0.068 / 0.160 | 0.230 / 0.137 |
| FPR@R90 | 0.483 / 0.249 | 0.310 / 0.254 |
| maxF2 | 0.505 / 0.652 | 0.634 / 0.641 |
| maxF3 | 0.647 / 0.751 | 0.733 / 0.750 |

## `10` politics  (train_pos=408, test_pos=335, EuroVoc 100142)

| metric | de EN / native | fr EN / native |
|---|---|---|
| AP | 0.251 / 0.370 | 0.202 / 0.398 |
| R@FPR1% | 0.146 / 0.197 | 0.033 / 0.194 |
| R@FPR5% | 0.275 / 0.361 | 0.224 / 0.388 |
| R@FPR10% | 0.388 / 0.522 | 0.418 / 0.534 |
| R@FA5 | 0.012 / 0.060 | 0.006 / 0.134 |
| R@FA20 | 0.063 / 0.143 | 0.024 / 0.173 |
| R@FA50 | 0.149 / 0.206 | 0.039 / 0.194 |
| R@FA100 | 0.212 / 0.260 | 0.084 / 0.272 |
| FPR@R90 | 0.530 / 0.353 | 0.495 / 0.381 |
| maxF2 | 0.424 / 0.507 | 0.469 / 0.494 |
| maxF3 | 0.554 / 0.619 | 0.580 / 0.620 |

## `20` environment  (train_pos=552, test_pos=610, EuroVoc 100155)

| metric | de EN / native | fr EN / native |
|---|---|---|
| AP | 0.469 / 0.497 | 0.498 / 0.545 |
| R@FPR1% | 0.216 / 0.198 | 0.205 / 0.220 |
| R@FPR5% | 0.420 / 0.425 | 0.400 / 0.503 |
| R@FPR10% | 0.500 / 0.546 | 0.502 / 0.575 |
| R@FA5 | 0.039 / 0.056 | 0.069 / 0.039 |
| R@FA20 | 0.149 / 0.121 | 0.139 / 0.111 |
| R@FA50 | 0.234 / 0.211 | 0.225 / 0.246 |
| R@FA100 | 0.325 / 0.302 | 0.302 / 0.367 |
| FPR@R90 | 0.595 / 0.362 | 0.358 / 0.304 |
| maxF2 | 0.512 / 0.605 | 0.617 / 0.635 |
| maxF3 | 0.661 / 0.724 | 0.726 / 0.749 |

## `8` transport  (train_pos=582, test_pos=600, EuroVoc 100154)

| metric | de EN / native | fr EN / native |
|---|---|---|
| AP | 0.749 / 0.826 | 0.705 / 0.816 |
| R@FPR1% | 0.477 / 0.613 | 0.378 / 0.617 |
| R@FPR5% | 0.718 / 0.803 | 0.682 / 0.777 |
| R@FPR10% | 0.780 / 0.853 | 0.770 / 0.833 |
| R@FA5 | 0.177 / 0.277 | 0.142 / 0.375 |
| R@FA20 | 0.390 / 0.525 | 0.277 / 0.572 |
| R@FA50 | 0.490 / 0.633 | 0.412 / 0.625 |
| R@FA100 | 0.602 / 0.713 | 0.542 / 0.697 |
| FPR@R90 | 0.265 / 0.199 | 0.344 / 0.249 |
| maxF2 | 0.719 / 0.778 | 0.701 / 0.757 |
| maxF3 | 0.766 / 0.810 | 0.756 / 0.795 |

## Headline EN-vs-native recall table (high-recall regime)

| concept | target | R@FA50 EN/nat | R@FA100 EN/nat | R@FPR10% EN/nat | FPR@R90 EN/nat | AP EN/nat |
|---|---|---|---|---|---|---|
| international organisations [13] | de | 0.098 / 0.147 | 0.167 / 0.216 | 0.373 / 0.549 | 0.508 / 0.484 | 0.094 / 0.145 |
| international organisations [13] | fr | 0.088 / 0.127 | 0.167 / 0.176 | 0.471 / 0.549 | 0.422 / 0.309 | 0.104 / 0.143 |
| science [16] | de | 0.088 / 0.035 | 0.140 / 0.123 | 0.526 / 0.404 | 0.401 / 0.776 | 0.058 / 0.050 |
| science [16] | fr | 0.088 / 0.123 | 0.281 / 0.175 | 0.561 / 0.491 | 0.539 / 0.507 | 0.071 / 0.057 |
| energy [14] | de | 0.449 / 0.514 | 0.535 / 0.584 | 0.697 / 0.746 | 0.422 / 0.274 | 0.509 / 0.546 |
| energy [14] | fr | 0.378 / 0.427 | 0.465 / 0.535 | 0.703 / 0.697 | 0.414 / 0.418 | 0.448 / 0.509 |
| employment and working conditions [9] | de | 0.183 / 0.067 | 0.262 / 0.195 | 0.518 / 0.671 | 0.375 / 0.306 | 0.214 / 0.210 |
| employment and working conditions [9] | fr | 0.043 / 0.098 | 0.134 / 0.165 | 0.543 / 0.701 | 0.362 / 0.281 | 0.137 / 0.193 |
| law [11] | de | 0.040 / 0.094 | 0.068 / 0.160 | 0.360 / 0.644 | 0.483 / 0.249 | 0.252 / 0.419 |
| law [11] | fr | 0.124 / 0.075 | 0.230 / 0.137 | 0.595 / 0.574 | 0.310 / 0.254 | 0.440 / 0.398 |
| politics [10] | de | 0.149 / 0.206 | 0.212 / 0.260 | 0.388 / 0.522 | 0.530 / 0.353 | 0.251 / 0.370 |
| politics [10] | fr | 0.039 / 0.194 | 0.084 / 0.272 | 0.418 / 0.534 | 0.495 / 0.381 | 0.202 / 0.398 |
| environment [20] | de | 0.234 / 0.211 | 0.325 / 0.302 | 0.500 / 0.546 | 0.595 / 0.362 | 0.469 / 0.497 |
| environment [20] | fr | 0.225 / 0.246 | 0.302 / 0.367 | 0.502 / 0.575 | 0.358 / 0.304 | 0.498 / 0.545 |
| transport [8] | de | 0.490 / 0.633 | 0.602 / 0.713 | 0.780 / 0.853 | 0.265 / 0.199 | 0.749 / 0.826 |
| transport [8] | fr | 0.412 / 0.625 | 0.542 / 0.697 | 0.770 / 0.833 | 0.344 / 0.249 | 0.705 / 0.816 |

## Verdict

- Cells evaluated (rare concept x target-language): **16**.
- English-trained AP within 0.05 of native in **8/16** cells (50%).
- **High-recall regime** (R@FPR10% + R@FA50 + R@FA100 all within 0.05 of native): **4/16** cells (25%).  This is the regime the task cares about.

### False-alarm cost at high recall

- Mean recall at 100 false alarms: EN **0.282** vs native **0.317** (across 16 cells).
- Mean recall at 50 false alarms: EN **0.195** vs native **0.239** (across 16 cells).
- Mean FPR needed to reach recall=0.90: EN **0.426** vs native **0.357** (across 16 cells).  Smaller is better; the difference is the extra false-alarm rate the English model spends for the same 90% recall.
