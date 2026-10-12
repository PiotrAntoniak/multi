# Which dims carry language, and does the English model use them?

EuroBERT-210m **mean** embeddings (768-d). Part A: per-dim one-way ANOVA eta^2 across languages (en/it/de/fr FLORES; en/de/fr MultiEURLEX) + a language linear probe before / after zeroing the top eta^2 dims. Part B: fixed English XGBoost (`hist`, `max_bin=64`, `cpu`, `n_jobs=2`, `seed=0`, `max_depth=8`, `min_child_weight=5`, `lr=0.1`, `n_estimators=400`); per-dim mean |SHAP| (`shap.TreeExplainer`) explained on a test-row sample. SHAP-mass k50 = fraction of total SHAP landing on the dims that carry 50% of the eta^2 mass; random = k/d.

| dataset | model | d | langs | total eta^2 | max eta^2 | #>0.5 | #>0.25 | k25/k50/k80/k90/k95 | probe full | probe -k50 | probe -k95 | Spearman | top-50 overlap | SHAP-mass k50 | random | lang dims get |
|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| sent | xgb_softprob_12class | 768 | en/it/de/fr | 255.8 | 0.774 | 146 | 508 | 104/237/455/557/625 | 1.000 | 1.000 | 0.998 | -0.076 | 0/50 | 0.283 | 0.309 | LESS |
| agg | xgb_softprob_11class | 768 | en/it/de/fr | 317.1 | 0.837 | 276 | 598 | 114/252/470/571/636 | 1.000 | 1.000 | 1.000 | -0.059 | 1/50 | 0.267 | 0.328 | LESS |
| mel | xgb_21x_ovr_binary | 768 | en/de/fr | 474.4 | 0.950 | 549 | 688 | 133/281/495/588/648 | 1.000 | 1.000 | 1.000 | -0.019 | 1/50 | 0.366 | 0.366 | MORE |

(probe chance = 1/#langs; `probe -k50/-k95` zero the top-eta^2 dims covering that fraction of the language mass.)

## sent

Part A -- top-20 language dims by eta^2:

| rank | dim | eta^2 |  | rank | dim | eta^2 |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 13 | 0.7739 |  | 11 | 445 | 0.7046 |
| 2 | 383 | 0.7725 |  | 12 | 451 | 0.6879 |
| 3 | 402 | 0.7682 |  | 13 | 118 | 0.6870 |
| 4 | 436 | 0.7354 |  | 14 | 404 | 0.6845 |
| 5 | 257 | 0.7282 |  | 15 | 259 | 0.6841 |
| 6 | 602 | 0.7238 |  | 16 | 574 | 0.6823 |
| 7 | 514 | 0.7214 |  | 17 | 534 | 0.6813 |
| 8 | 511 | 0.7107 |  | 18 | 577 | 0.6761 |
| 9 | 480 | 0.7107 |  | 19 | 254 | 0.6728 |
| 10 | 153 | 0.7100 |  | 20 | 129 | 0.6697 |

Part B -- top-10 SHAP dims:

| rank | dim | mean |SHAP| |  | rank | dim | mean |SHAP| |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 370 | 0.06314 |  | 6 | 216 | 0.04450 |
| 2 | 607 | 0.05714 |  | 7 | 349 | 0.04216 |
| 3 | 418 | 0.05214 |  | 8 | 92 | 0.04171 |
| 4 | 689 | 0.04889 |  | 9 | 453 | 0.04023 |
| 5 | 764 | 0.04760 |  | 10 | 503 | 0.03959 |

SHAP mass on the 237 language dims (50%-mass): **0.283** vs random 0.309 -> language dims get **LESS** SHAP than random.

## agg

Part A -- top-20 language dims by eta^2:

| rank | dim | eta^2 |  | rank | dim | eta^2 |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 13 | 0.8373 |  | 11 | 445 | 0.7643 |
| 2 | 383 | 0.8336 |  | 12 | 480 | 0.7626 |
| 3 | 402 | 0.8161 |  | 13 | 763 | 0.7600 |
| 4 | 602 | 0.8147 |  | 14 | 259 | 0.7577 |
| 5 | 514 | 0.8132 |  | 15 | 534 | 0.7555 |
| 6 | 436 | 0.7983 |  | 16 | 704 | 0.7531 |
| 7 | 511 | 0.7972 |  | 17 | 718 | 0.7526 |
| 8 | 153 | 0.7966 |  | 18 | 574 | 0.7510 |
| 9 | 577 | 0.7672 |  | 19 | 223 | 0.7509 |
| 10 | 257 | 0.7654 |  | 20 | 451 | 0.7480 |

Part B -- top-10 SHAP dims:

| rank | dim | mean |SHAP| |  | rank | dim | mean |SHAP| |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 418 | 0.06945 |  | 6 | 216 | 0.04598 |
| 2 | 466 | 0.06639 |  | 7 | 364 | 0.04232 |
| 3 | 49 | 0.05510 |  | 8 | 363 | 0.03882 |
| 4 | 497 | 0.05171 |  | 9 | 122 | 0.03817 |
| 5 | 565 | 0.04940 |  | 10 | 734 | 0.03813 |

SHAP mass on the 252 language dims (50%-mass): **0.267** vs random 0.328 -> language dims get **LESS** SHAP than random.

## mel

Part A -- top-20 language dims by eta^2:

| rank | dim | eta^2 |  | rank | dim | eta^2 |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 53 | 0.9497 |  | 11 | 86 | 0.9344 |
| 2 | 695 | 0.9482 |  | 12 | 761 | 0.9337 |
| 3 | 102 | 0.9464 |  | 13 | 223 | 0.9337 |
| 4 | 465 | 0.9428 |  | 14 | 476 | 0.9336 |
| 5 | 437 | 0.9426 |  | 15 | 404 | 0.9334 |
| 6 | 257 | 0.9426 |  | 16 | 137 | 0.9313 |
| 7 | 553 | 0.9422 |  | 17 | 704 | 0.9313 |
| 8 | 589 | 0.9419 |  | 18 | 219 | 0.9311 |
| 9 | 763 | 0.9375 |  | 19 | 524 | 0.9310 |
| 10 | 602 | 0.9358 |  | 20 | 47 | 0.9299 |

Part B -- top-10 SHAP dims:

| rank | dim | mean |SHAP| |  | rank | dim | mean |SHAP| |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 18 | 0.09660 |  | 6 | 194 | 0.05633 |
| 2 | 405 | 0.06584 |  | 7 | 598 | 0.05629 |
| 3 | 720 | 0.06144 |  | 8 | 97 | 0.05603 |
| 4 | 199 | 0.06016 |  | 9 | 299 | 0.05272 |
| 5 | 176 | 0.05756 |  | 10 | 462 | 0.05207 |

SHAP mass on the 281 language dims (50%-mass): **0.366** vs random 0.366 -> language dims get **MORE** SHAP than random.

## Conclusion

- **sent**: language lives in a **distributed/redundant** of dims (k50=237, k95=625 of 768; probe 1.000 -> 1.000 after zeroing k50). The task model **ignored / under-used (SHAP mass <= random)** (Spearman -0.076, top-50 overlap 0/50).
- **agg**: language lives in a **distributed/redundant** of dims (k50=252, k95=636 of 768; probe 1.000 -> 1.000 after zeroing k50). The task model **ignored / under-used (SHAP mass <= random)** (Spearman -0.059, top-50 overlap 1/50).
- **mel**: language lives in a **distributed/redundant** of dims (k50=281, k95=648 of 768; probe 1.000 -> 1.000 after zeroing k50). The task model **used by the classifier (SHAP mass >= random)** (Spearman -0.019, top-50 overlap 1/50).

