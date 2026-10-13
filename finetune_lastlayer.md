# Last-layer fine-tuning of EuroBERT vs frozen+XGBoost

Model `EuroBERT/EuroBERT-210m` (only `model.layers[-1]` + a linear head trained; embeddings and all lower layers frozen); masked-mean pooling of the last hidden state; AdamW(betas=(0.9,0.95), eps=1e-5, weight_decay=0.1), grad-clip 1.0, linear schedule with warmup ratio 0.1, batch 32, patience 1 epoch. **Training compute runs under bf16 autocast with fp32 master weights (standard AMP); the cross-entropy / BCE loss is computed in fp32.**

**Features are RAW**: the mean-pooled last hidden state goes straight into the head with NO per-language normalization, L2, z-score, PC/INLP projection or language-shift `D`, and every language's test split is scored on its raw features. The frozen+XGBoost baseline likewise uses raw frozen mean embeddings, so the comparison isolates the effect of fine-tuning.

LR grid [1e-05, 3.6e-05, 6e-05, 0.0001] selected on the EN validation split by validation AUC.

Each cell is test-set **AUC** (macro-F1 in the table). Frozen+XGBoost is retrained in-script on the same frozen mean embeddings (fixed config) for apples-to-apples; in-language FT is the fine-tuned ceiling.


## sent

**Chosen LR = 3.6e-05** (val AUC 0.8819, best epoch 7, steps 224). LR grid: 1e-05: val_auc=0.874/f1=0.442, 3.6e-05: val_auc=0.882/f1=0.482, 6e-05: val_auc=0.863/f1=0.389, 0.0001: val_auc=0.857/f1=0.376.

| eval | FT AUC | FT F1 | XGB AUC | XGB F1 | in-lang FT AUC | in-lang FT F1 |
|---|---:|---:|---:|---:|---:|---:|
| EN-self | 0.9156 | 0.5508 | 0.9106 | 0.4878 | - | - |
| EN->it | 0.8124 | 0.2480 | 0.7987 | 0.1843 | 0.8413 | 0.3059 |
| EN->de | 0.8139 | 0.2465 | 0.8321 | 0.2926 | 0.8565 | 0.3512 |
| EN->fr | 0.8211 | 0.2504 | 0.8295 | 0.2330 | 0.8360 | 0.3877 |

## agg

**Chosen LR = 3.6e-05** (val AUC 0.9042, best epoch 10, steps 90). LR grid: 1e-05: val_auc=0.896/f1=0.605, 3.6e-05: val_auc=0.904/f1=0.555, 6e-05: val_auc=0.903/f1=0.543, 0.0001: val_auc=0.898/f1=0.490.

| eval | FT AUC | FT F1 | XGB AUC | XGB F1 | in-lang FT AUC | in-lang FT F1 |
|---|---:|---:|---:|---:|---:|---:|
| EN-self | 0.9254 | 0.5450 | 0.9134 | 0.5555 | - | - |
| EN->it | 0.7905 | 0.2270 | 0.8530 | 0.2131 | 0.9194 | 0.5492 |
| EN->de | 0.8526 | 0.2015 | 0.8359 | 0.2779 | 0.9177 | 0.5092 |
| EN->fr | 0.8337 | 0.2689 | 0.8100 | 0.2939 | 0.9162 | 0.4806 |

## toxic

**Chosen LR = 0.0001** (val AUC 0.9262, best epoch 1, steps 400). LR grid: 1e-05: val_auc=0.888/f1=0.754, 3.6e-05: val_auc=0.909/f1=0.782, 6e-05: val_auc=0.918/f1=0.793, 0.0001: val_auc=0.926/f1=0.807.

| eval | FT AUC | FT F1 | XGB AUC | XGB F1 | in-lang FT AUC | in-lang FT F1 |
|---|---:|---:|---:|---:|---:|---:|
| EN-self | 0.9287 | 0.7971 | 0.9040 | 0.7601 | - | - |
| EN->it | 0.6425 | 0.5006 | 0.6692 | 0.4490 | - | - |
| EN->de | 0.7389 | 0.5535 | 0.7052 | 0.5106 | - | - |
| EN->fr | 0.7425 | 0.4864 | 0.7714 | 0.4457 | - | - |
| EN->es | 0.6891 | 0.5302 | 0.6293 | 0.4297 | - | - |

## Files

- figure: `finetune_lastlayer.png`
- data: `finetune_lastlayer.json`
