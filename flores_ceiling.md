# FLORES in-language ceilings -- is "0.9 x English-self" feasible?

Same protocol as `flores_calib` / `proof_mean_alllangs`: EuroBERT-210m **mean**, `perlang_z` (L2-normalize + per-language *train* z-score), one XGBoost per (dataset, language) (softprob, depth 8 / mcw 5 / lr 0.1 / n 400, hist, max_bin 64, cpu, n_jobs 2, seed 0), trained on the language's **train+val** rows and evaluated on its **test** rows. This is each language's own ceiling (best it can do with its OWN model).


## sent (12 classes, train+val=1501 / test=505)

### In-language ceilings

| lang | micro-F1 | macro-F1 | AUC |
|---|---:|---:|---:|
| en (EN self) | 0.6059 | 0.4865 | 0.9038 |
| it | 0.5624 | 0.4343 | 0.8829 |
| de | 0.5366 | 0.4355 | 0.8835 |
| fr | 0.5822 | 0.4675 | 0.8741 |

### EN-trained calibrated transfer vs ceiling (best label-free = `quantile_hardprior`)

| target | cal micro | ceiling micro | ratio micro | cal macro | ceiling macro | ratio macro |
|---|---:|---:|---:|---:|---:|---:|
| it | 0.3980 | 0.5624 | **0.708** | 0.2986 | 0.4343 | **0.687** |
| de | 0.3921 | 0.5366 | **0.731** | 0.2880 | 0.4355 | **0.661** |
| fr | 0.4198 | 0.5822 | **0.721** | 0.3476 | 0.4675 | **0.744** |

English self under the SAME (train+val) protocol: micro **0.6059**, macro **0.4865** (0.9x -> micro 0.5453, macro 0.4378). For reference, the `flores_calib` train-only EN baseline self is micro 0.5366 / macro 0.4032 (0.9x -> micro 0.4830, macro 0.3628).

### Feasibility of "0.9 x English-self" (vs each target's own ceiling)

| target | 0.9xEN-self micro | target ceil micro | infeasible? | 0.9xEN-self macro | target ceil macro | infeasible? |
|---|---:|---:|:--:|---:|---:|:--:|
| it | 0.5453 | 0.5624 | no | 0.4378 | 0.4343 | **YES** |
| de | 0.5453 | 0.5366 | **YES** | 0.4378 | 0.4355 | **YES** |
| fr | 0.5453 | 0.5822 | no | 0.4378 | 0.4675 | no |

- **it**: INFEASIBLE on macro only. (micro margin 0.9*EN-self - ceiling = -0.0170; macro = +0.0035). Using the flores_calib train-only EN self the verdict is: feasible in principle (target ceiling >= 0.9*EN-self).
- **de**: INFEASIBLE (0.9*EN-self exceeds target's own ceiling on both micro and macro). (micro margin 0.9*EN-self - ceiling = +0.0087; macro = +0.0023). Using the flores_calib train-only EN self the verdict is: feasible in principle (target ceiling >= 0.9*EN-self).
- **fr**: feasible in principle (target ceiling >= 0.9*EN-self). (micro margin 0.9*EN-self - ceiling = -0.0368; macro = -0.0297). Using the flores_calib train-only EN self the verdict is: feasible in principle (target ceiling >= 0.9*EN-self).

Note: a positive margin (0.9*EN-self > target ceiling) means the target's own best model is notionally **below the 0.9 bar**, so no EN->target transfer can reach it. A ratio `calibrated/ceiling` near 1 means the English model already matches the target's own ceiling (little headroom left).


## agg (11 classes, train+val=416 / test=139)

### In-language ceilings

| lang | micro-F1 | macro-F1 | AUC |
|---|---:|---:|---:|
| en (EN self) | 0.6187 | 0.5573 | 0.9142 |
| it | 0.6187 | 0.5588 | 0.9016 |
| de | 0.6043 | 0.5322 | 0.8949 |
| fr | 0.5324 | 0.4542 | 0.8852 |

### EN-trained calibrated transfer vs ceiling (best label-free = `prior_sinkhorn`)

| target | cal micro | ceiling micro | ratio micro | cal macro | ceiling macro | ratio macro |
|---|---:|---:|---:|---:|---:|---:|
| it | 0.5036 | 0.6187 | **0.814** | 0.3495 | 0.5588 | **0.625** |
| de | 0.5108 | 0.6043 | **0.845** | 0.3664 | 0.5322 | **0.688** |
| fr | 0.5468 | 0.5324 | **1.027** | 0.4118 | 0.4542 | **0.907** |

English self under the SAME (train+val) protocol: micro **0.6187**, macro **0.5573** (0.9x -> micro 0.5568, macro 0.5016). For reference, the `flores_calib` train-only EN baseline self is micro 0.5540 / macro 0.5014 (0.9x -> micro 0.4986, macro 0.4513).

### Feasibility of "0.9 x English-self" (vs each target's own ceiling)

| target | 0.9xEN-self micro | target ceil micro | infeasible? | 0.9xEN-self macro | target ceil macro | infeasible? |
|---|---:|---:|:--:|---:|---:|:--:|
| it | 0.5568 | 0.6187 | no | 0.5016 | 0.5588 | no |
| de | 0.5568 | 0.6043 | no | 0.5016 | 0.5322 | no |
| fr | 0.5568 | 0.5324 | **YES** | 0.5016 | 0.4542 | **YES** |

- **it**: feasible in principle (target ceiling >= 0.9*EN-self). (micro margin 0.9*EN-self - ceiling = -0.0619; macro = -0.0572). Using the flores_calib train-only EN self the verdict is: feasible in principle (target ceiling >= 0.9*EN-self).
- **de**: feasible in principle (target ceiling >= 0.9*EN-self). (micro margin 0.9*EN-self - ceiling = -0.0475; macro = -0.0307). Using the flores_calib train-only EN self the verdict is: feasible in principle (target ceiling >= 0.9*EN-self).
- **fr**: INFEASIBLE (0.9*EN-self exceeds target's own ceiling on both micro and macro). (micro margin 0.9*EN-self - ceiling = +0.0245; macro = +0.0474). Using the flores_calib train-only EN self the verdict is: feasible in principle (target ceiling >= 0.9*EN-self).

Note: a positive margin (0.9*EN-self > target ceiling) means the target's own best model is notionally **below the 0.9 bar**, so no EN->target transfer can reach it. A ratio `calibrated/ceiling` near 1 means the English model already matches the target's own ceiling (little headroom left).


## Verdict

- **sent**: infeasible-on-some-metric targets (0.9 x EN train+val self) = it, de; calibrated/ceiling micro ratios = it 0.708, de 0.731, fr 0.721.
- **agg**: infeasible-on-some-metric targets (0.9 x EN train+val self) = fr; calibrated/ceiling micro ratios = it 0.814, de 0.845, fr 1.027.

