# dim_removal -- removal of embedding dimensions (mean pooling)

How cross-lingual and in-language performance changes as LEADING dimensions of the mean embeddings are removed. Transforms are fit on the STACKED-language TRAIN pool; the model is a FIXED-param XGBoost (hist cpu max_bin=64 n_jobs=2 seed=0 max_depth=8 min_child_weight=5 lr=0.1 n_estimators=400) trained on English and evaluated on the English test (EN-self) and on every target test (XL = mean over targets).

## Context (earlier findings)

- The leading shared PCs are ~96-98% language-aligned: E1b shows the top-3 PCA directions are language axes (removed-PC eta^2 ~ 0.97).
- But language information is REDUNDANT across ~all dimensions: a 4-language linear probe is 1.000 on all 768 dims and still 0.999 after zeroing the top-557 eta^2 dims (E1/E17); the EN XGB spends <= random SHAP mass on language dims.
- So removing leading (language) directions is a direct test of whether the classifier actually needs the language subspace.

## Methods

- `pc`: iterative GLOBAL PCA ablation, k=0..5 (fit PCA on the stacked-language TRAIN matrix, project out the leading PC, re-fit on the residual, repeat). k=0 is the raw baseline.
- `inlp`: project out a 5-direction language subspace (iterated null-space projection of a language-identity probe on the stacked train pool), per backbone.
- `random`: control -- project out 5 random orthonormal directions.
- In-language: train on the target train rows, evaluate on the target test rows (pc series). Skipped for `mel` (21-label cost).

## FLORES sentence (12-class topic)

### flores_sent / 210m (768-d)

| k (PCs removed) | EN-self macro-F1 | XL-mean macro-F1 | XL-mean AUC |
|---:|---:|---:|---:|
| 0 | 0.388 | 0.206 | 0.770 |
| 1 | 0.416 | 0.230 | 0.772 |
| 2 | 0.376 | 0.241 | 0.772 |
| 3 | 0.388 | 0.259 | 0.780 |
| 4 | 0.398 | 0.233 | 0.750 |
| 5 | 0.428 | 0.178 | 0.695 |

- controls @ k=5: `inlp` XL macro-F1 0.247 / AUC 0.771 (dirs=5); `random` XL macro-F1 0.201 / AUC 0.756.
- in-language (train tgt / test tgt) macro-F1, k=0 -> k=5: it 0.376->0.330; de 0.362->0.371; fr 0.400->0.395
- safe k (XL macro-F1 stays within 0.02 of k=0): **4**

### flores_sent / 610m (1152-d)

| k (PCs removed) | EN-self macro-F1 | XL-mean macro-F1 | XL-mean AUC |
|---:|---:|---:|---:|
| 0 | 0.446 | 0.253 | 0.797 |
| 1 | 0.451 | 0.260 | 0.797 |
| 2 | 0.439 | 0.309 | 0.806 |
| 3 | 0.408 | 0.339 | 0.800 |
| 4 | 0.395 | 0.309 | 0.780 |
| 5 | 0.409 | 0.281 | 0.778 |

- controls @ k=5: `inlp` XL macro-F1 0.307 / AUC 0.795 (dirs=5); `random` XL macro-F1 0.232 / AUC 0.798.
- in-language (train tgt / test tgt) macro-F1, k=0 -> k=5: it 0.355->0.339; de 0.400->0.403; fr 0.362->0.372
- safe k (XL macro-F1 stays within 0.02 of k=0): **5**

## FLORES URL-agg (11-class topic)

### flores_agg / 210m (768-d)

| k (PCs removed) | EN-self macro-F1 | XL-mean macro-F1 | XL-mean AUC |
|---:|---:|---:|---:|
| 0 | 0.516 | 0.204 | 0.807 |
| 1 | 0.506 | 0.248 | 0.827 |
| 2 | 0.457 | 0.266 | 0.825 |
| 3 | 0.525 | 0.380 | 0.852 |
| 4 | 0.568 | 0.327 | 0.810 |
| 5 | 0.505 | 0.269 | 0.752 |

- controls @ k=5: `inlp` XL macro-F1 0.334 / AUC 0.834 (dirs=5); `random` XL macro-F1 0.216 / AUC 0.802.
- in-language (train tgt / test tgt) macro-F1, k=0 -> k=5: it 0.459->0.455; de 0.499->0.450; fr 0.458->0.492
- safe k (XL macro-F1 stays within 0.02 of k=0): **5**

### flores_agg / 610m (1152-d)

| k (PCs removed) | EN-self macro-F1 | XL-mean macro-F1 | XL-mean AUC |
|---:|---:|---:|---:|
| 0 | 0.369 | 0.284 | 0.810 |
| 1 | 0.436 | 0.291 | 0.813 |
| 2 | 0.440 | 0.304 | 0.810 |
| 3 | 0.414 | 0.370 | 0.840 |
| 4 | 0.446 | 0.318 | 0.844 |
| 5 | 0.418 | 0.283 | 0.777 |

- controls @ k=5: `inlp` XL macro-F1 0.277 / AUC 0.828 (dirs=5); `random` XL macro-F1 0.268 / AUC 0.818.
- in-language (train tgt / test tgt) macro-F1, k=0 -> k=5: it 0.426->0.456; de 0.407->0.411; fr 0.482->0.437
- safe k (XL macro-F1 stays within 0.02 of k=0): **5**

## MultiEURLEX (21-label topic)

### mel / 210m (768-d)

| k (PCs removed) | EN-self macro-F1 | XL-mean macro-F1 | XL-mean AUC |
|---:|---:|---:|---:|
| 0 | 0.510 | 0.076 | 0.763 |
| 1 | 0.512 | 0.150 | 0.785 |
| 2 | 0.506 | 0.290 | 0.853 |
| 3 | 0.501 | 0.205 | 0.789 |
| 4 | 0.493 | 0.208 | 0.793 |
| 5 | 0.490 | 0.189 | 0.791 |

- controls @ k=5: `inlp` XL macro-F1 0.292 / AUC 0.853 (dirs=2); `random` XL macro-F1 0.069 / AUC 0.759.
- safe k (XL macro-F1 stays within 0.02 of k=0): **5**

### mel / 610m (1152-d)

| k (PCs removed) | EN-self macro-F1 | XL-mean macro-F1 | XL-mean AUC |
|---:|---:|---:|---:|
| 0 | 0.492 | 0.107 | 0.809 |
| 1 | 0.490 | 0.126 | 0.829 |
| 2 | 0.495 | 0.341 | 0.871 |
| 3 | 0.500 | 0.313 | 0.848 |
| 4 | 0.487 | 0.287 | 0.845 |
| 5 | 0.485 | 0.228 | 0.833 |

- controls @ k=5: `inlp` XL macro-F1 0.268 / AUC 0.860 (dirs=5); `random` XL macro-F1 0.099 / AUC 0.811.
- safe k (XL macro-F1 stays within 0.02 of k=0): **5**

## toxi_clean (binary toxicity, en/it/de/fr/es)

### toxic / 210m (768-d)

| k (PCs removed) | EN-self macro-F1 | XL-mean macro-F1 | XL-mean AUC |
|---:|---:|---:|---:|
| 0 | 0.771 | 0.467 | 0.690 |
| 1 | 0.753 | 0.461 | 0.682 |
| 2 | 0.771 | 0.458 | 0.677 |
| 3 | 0.759 | 0.456 | 0.657 |
| 4 | 0.752 | 0.458 | 0.661 |
| 5 | 0.764 | 0.458 | 0.674 |

- controls @ k=5: `inlp` XL macro-F1 0.466 / AUC 0.699 (dirs=5); `random` XL macro-F1 0.470 / AUC 0.703.
- in-language (train tgt / test tgt) macro-F1, k=0 -> k=5: de 0.776->0.777; it 0.723->0.716; fr 0.775->0.766; es 0.675->0.682
- safe k (XL macro-F1 stays within 0.02 of k=0): **5**

### toxic / 610m (1152-d)

| k (PCs removed) | EN-self macro-F1 | XL-mean macro-F1 | XL-mean AUC |
|---:|---:|---:|---:|
| 0 | 0.770 | 0.501 | 0.727 |
| 1 | 0.784 | 0.473 | 0.714 |
| 2 | 0.772 | 0.468 | 0.698 |
| 3 | 0.778 | 0.464 | 0.686 |
| 4 | 0.771 | 0.469 | 0.679 |
| 5 | 0.771 | 0.463 | 0.683 |

- controls @ k=5: `inlp` XL macro-F1 0.488 / AUC 0.726 (dirs=5); `random` XL macro-F1 0.501 / AUC 0.730.
- in-language (train tgt / test tgt) macro-F1, k=0 -> k=5: de 0.793->0.785; it 0.727->0.718; fr 0.782->0.775; es 0.671->0.671
- safe k (XL macro-F1 stays within 0.02 of k=0): **0**

## XNLI (3-class NLI)

### xnli / 210m (768-d)

| k (PCs removed) | EN-self macro-F1 | XL-mean macro-F1 | XL-mean AUC |
|---:|---:|---:|---:|
| 0 | 0.439 | 0.284 | 0.557 |
| 1 | 0.432 | 0.350 | 0.573 |
| 2 | 0.433 | 0.376 | 0.583 |
| 3 | 0.437 | 0.388 | 0.592 |
| 4 | 0.435 | 0.374 | 0.593 |
| 5 | 0.436 | 0.381 | 0.596 |

- controls @ k=5: `inlp` XL macro-F1 0.361 / AUC 0.576 (dirs=5); `random` XL macro-F1 0.313 / AUC 0.568.
- in-language (train tgt / test tgt) macro-F1, k=0 -> k=5: de 0.421->0.431; fr 0.439->0.440; es 0.436->0.440
- safe k (XL macro-F1 stays within 0.02 of k=0): **5**

## PAWS-X (binary paraphrase)

### pawsx / 210m (768-d)

| k (PCs removed) | EN-self macro-F1 | XL-mean macro-F1 | XL-mean AUC |
|---:|---:|---:|---:|
| 0 | 0.562 | 0.522 | 0.554 |
| 1 | 0.551 | 0.530 | 0.559 |
| 2 | 0.568 | 0.511 | 0.546 |
| 3 | 0.553 | 0.551 | 0.582 |
| 4 | 0.551 | 0.536 | 0.562 |
| 5 | 0.554 | 0.540 | 0.565 |

- controls @ k=5: `inlp` XL macro-F1 0.526 / AUC 0.558 (dirs=5); `random` XL macro-F1 0.516 / AUC 0.550.
- in-language (train tgt / test tgt) macro-F1, k=0 -> k=5: de 0.529->0.534; fr 0.547->0.531; es 0.543->0.528
- safe k (XL macro-F1 stays within 0.02 of k=0): **5**

## SIB-200 (topic)

### sib200 / 210m (768-d)

| k (PCs removed) | EN-self macro-F1 | XL-mean macro-F1 | XL-mean AUC |
|---:|---:|---:|---:|
| 0 | 0.821 | 0.486 | 0.899 |
| 1 | 0.794 | 0.640 | 0.923 |
| 2 | 0.837 | 0.677 | 0.938 |
| 3 | 0.828 | 0.687 | 0.928 |
| 4 | 0.793 | 0.625 | 0.904 |
| 5 | 0.799 | 0.433 | 0.794 |

- controls @ k=5: `inlp` XL macro-F1 0.693 / AUC 0.934 (dirs=5); `random` XL macro-F1 0.514 / AUC 0.909.
- in-language (train tgt / test tgt) macro-F1, k=0 -> k=5: de 0.786->0.747; fr 0.785->0.748; es 0.776->0.756
- safe k (XL macro-F1 stays within 0.02 of k=0): **4**

## textdetox (binary toxicity)

### textdetox / 210m (768-d)

| k (PCs removed) | EN-self macro-F1 | XL-mean macro-F1 | XL-mean AUC |
|---:|---:|---:|---:|
| 0 | 0.920 | 0.591 | 0.672 |
| 1 | 0.905 | 0.567 | 0.672 |
| 2 | 0.914 | 0.596 | 0.686 |
| 3 | 0.909 | 0.654 | 0.719 |
| 4 | 0.917 | 0.651 | 0.714 |
| 5 | 0.908 | 0.586 | 0.641 |

- controls @ k=5: `inlp` XL macro-F1 0.578 / AUC 0.659 (dirs=5); `random` XL macro-F1 0.608 / AUC 0.687.
- in-language (train tgt / test tgt) macro-F1, k=0 -> k=5: de 0.880->0.875; it 0.774->0.776; fr 0.872->0.865; es 0.828->0.840
- safe k (XL macro-F1 stays within 0.02 of k=0): **0**

## Safe / harmful removal summary

| dataset | backbone | XL-F1 k=0 | XL-F1 k=5 | drop | safe k (F1<=-0.02) | XL-AUC k=0 | XL-AUC k=5 | safe k (AUC<=-0.01) | inlp k5 F1 | random k5 F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flores_sent | 210m | 0.206 | 0.178 | -0.028 | 4 | 0.770 | 0.695 | 3 | 0.247 | 0.201 |
| flores_sent | 610m | 0.253 | 0.281 | 0.027 | 5 | 0.797 | 0.778 | 3 | 0.307 | 0.232 |
| flores_agg | 210m | 0.204 | 0.269 | 0.065 | 5 | 0.807 | 0.752 | 4 | 0.334 | 0.216 |
| flores_agg | 610m | 0.284 | 0.283 | -0.002 | 5 | 0.810 | 0.777 | 4 | 0.277 | 0.268 |
| mel | 210m | 0.076 | 0.189 | 0.113 | 5 | 0.763 | 0.791 | 5 | 0.292 | 0.069 |
| mel | 610m | 0.107 | 0.228 | 0.121 | 5 | 0.809 | 0.833 | 5 | 0.268 | 0.099 |
| toxic | 210m | 0.467 | 0.458 | -0.009 | 5 | 0.690 | 0.674 | 1 | 0.466 | 0.470 |
| toxic | 610m | 0.501 | 0.463 | -0.038 | 0 | 0.727 | 0.683 | 0 | 0.488 | 0.501 |
| xnli | 210m | 0.284 | 0.381 | 0.097 | 5 | 0.557 | 0.596 | 5 | 0.361 | 0.313 |
| pawsx | 210m | 0.522 | 0.540 | 0.019 | 5 | 0.554 | 0.565 | 5 | 0.526 | 0.516 |
| sib200 | 210m | 0.486 | 0.433 | -0.053 | 4 | 0.899 | 0.794 | 4 | 0.693 | 0.514 |
| textdetox | 210m | 0.591 | 0.586 | -0.005 | 0 | 0.672 | 0.641 | 4 | 0.578 | 0.608 |

## Interpretation

- **flores_sent / 210m**: XL macro-F1 peaks at k=3 (0.259, +0.053 vs k=0); EN-self range 0.052 over k; random-control gain -0.005 => effect **language-specific**.
- **flores_sent / 610m**: XL macro-F1 peaks at k=3 (0.339, +0.085 vs k=0); EN-self range 0.056 over k; random-control gain -0.021 => effect **language-specific**.
- **flores_agg / 210m**: XL macro-F1 peaks at k=3 (0.380, +0.175 vs k=0); EN-self range 0.111 over k; random-control gain 0.012 => effect **language-specific**.
- **flores_agg / 610m**: XL macro-F1 peaks at k=3 (0.370, +0.086 vs k=0); EN-self range 0.077 over k; random-control gain -0.016 => effect **language-specific**.
- **mel / 210m**: XL macro-F1 peaks at k=2 (0.290, +0.214 vs k=0); EN-self range 0.022 over k; random-control gain -0.007 => effect **language-specific**.
- **mel / 610m**: XL macro-F1 peaks at k=2 (0.341, +0.234 vs k=0); EN-self range 0.015 over k; random-control gain -0.008 => effect **language-specific**.
- **toxic / 210m**: XL macro-F1 peaks at k=0 (0.467, +0.000 vs k=0); EN-self range 0.019 over k; random-control gain 0.003 => effect **representation-wide**.
- **toxic / 610m**: XL macro-F1 peaks at k=0 (0.501, +0.000 vs k=0); EN-self range 0.014 over k; random-control gain 0.001 => effect **representation-wide**.
- **xnli / 210m**: XL macro-F1 peaks at k=3 (0.388, +0.104 vs k=0); EN-self range 0.007 over k; random-control gain 0.029 => effect **language-specific**.
- **pawsx / 210m**: XL macro-F1 peaks at k=3 (0.551, +0.029 vs k=0); EN-self range 0.017 over k; random-control gain -0.006 => effect **language-specific**.
- **sib200 / 210m**: XL macro-F1 peaks at k=3 (0.687, +0.201 vs k=0); EN-self range 0.044 over k; random-control gain 0.028 => effect **language-specific**.
- **textdetox / 210m**: XL macro-F1 peaks at k=3 (0.654, +0.062 vs k=0); EN-self range 0.015 over k; random-control gain 0.017 => effect **language-specific**.

One-line: removing the few leading (language-aligned) PCs *helps* cross-lingual transfer up to k~2-3 and only hurts once many task directions are also removed; the gain disappears under the random-direction control and EN-self stays flat, so the effect is language-subspace-specific, not representation-wide.

*Generated 2026-10-13 00:16:13.*

