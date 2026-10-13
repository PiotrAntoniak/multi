# toxi_pr -- precision / recall 2x2 (XGBoost, RAW mean embeddings)

XGBoost toxicity precision/recall 2x2 (EN model / IT model x EN / IT test) on **RAW** mean-pooled EuroBERT embeddings `emb_bf16/toxic/{210m,610m}/emb_<lang>_<split>.npy` (`data/toxi_clean/`, label `is_toxic`).  **No `perlang_z`** -- no L2-normalize, no per-language z-score, no alignment (matches `toxi_pr_models.py`).  Figure `toxi_pr.png`: `210m` = solid, `610m` = dotted.  Val-tuned threshold = F1-optimal on the model's own-language validation split (grid 0.00..1.00 step 0.01).  XGBoost `binary:logistic, hist, max_bin=64, cpu, n_jobs=2, seed=0, depth=8, mcw=5, lr=0.1, n=400`; a fresh EN and IT model per backbone.

## @ val-tuned threshold and @ recall=0.90 (per backbone)

| cell | backbone | P@val | R@val | F1@val | AUC | P@R90 | R@R90 | thr@R90 |
|---|---|---|---|---|---|---|---|---|
| EN model / EN test | 210m | 0.691 | 0.598 | 0.641 | 0.907 | 0.271 | 0.901 | 0.005 |
| EN model / EN test | 610m | 0.724 | 0.587 | 0.648 | 0.913 | 0.287 | 0.901 | 0.005 |
| EN model / IT test | 210m | 0.452 | 0.125 | 0.196 | 0.621 | 0.318 | 0.901 | 0.008 |
| EN model / IT test | 610m | 0.489 | 0.473 | 0.481 | 0.715 | 0.360 | 0.901 | 0.016 |
| IT model / IT test | 210m | 0.629 | 0.673 | 0.650 | 0.842 | 0.450 | 0.901 | 0.030 |
| IT model / IT test | 610m | 0.594 | 0.690 | 0.638 | 0.843 | 0.467 | 0.901 | 0.031 |

## Interpretation

- **210m**: on Italian at recall=0.90, EN-on-IT precision **0.318** vs IT-on-IT **0.450** (EN-on-IT gap **-0.131** vs native).
- **610m**: on Italian at recall=0.90, EN-on-IT precision **0.360** vs IT-on-IT **0.467** (EN-on-IT gap **-0.107** vs native).

