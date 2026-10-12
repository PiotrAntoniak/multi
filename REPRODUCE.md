# Reproduction

Everything runs on Windows PowerShell 5.1 with `python = C:\Users\pedro\anaconda3\python.exe`.
All models/datasets come from the Hugging Face cache; no GPU is required except for embedding.

## 0. Environment

```powershell
pip install -r requirements.txt      # numpy, torch, transformers, xgboost, optuna, pandas, scikit-learn, shap, datasets, matplotlib
```

Backbone `EuroBERT/EuroBERT-210m` (768-d) and `EuroBERT/EuroBERT-610m` are pulled from HF on first use.
MT models (`Helsinki-NLP/opus-mt-*`) are only needed for the *optional* translate-train experiments.

## 1. Data

- **FLORES**: `flores200_en_it_de_fr.csv` (2006 rows) and `flores_agg.csv` (562 rows) are in the repo.
  `train_flores12.py` aligns them to the historical **2009-row** embeddings via git ref
  `196bbd3:flores200_en_it_de_fr.csv` (the current CSV is that file minus 3 deleted rows).
- **MultiEURLEX**: fetched automatically from the HF parquet conversion:
  `hf://datasets/nlpaueb/multi_eurlex@refs/convert/parquet/<lang>/<split>/*.parquet` (en/de/fr).
- **SST-2 / emotion**: `sst2.csv`, `emotion.csv` are in the repo.

Large derived caches are **git-ignored** and regenerated below: `embeddings*/`, `multieurlex*/`,
`translated*/`, `*_emb/`, `*.npy`, `*.npz`.

## 2. Embeddings (mean pooling, EuroBERT-210m)

```powershell
# FLORES sentences (all four poolings; mean is what the study uses)
python extract_embeddings.py --data flores200_en_it_de_fr.csv --out-dir embeddings --no-migrate-bos

# FLORES aggregated
python extract_embeddings.py --data flores_agg.csv --out-dir embeddings_agg --no-migrate-bos

# 610m variants
python extract_embeddings.py --model-id EuroBERT/EuroBERT-610m --data flores200_en_it_de_fr.csv --out-dir embeddings_610 --no-migrate-bos
python extract_embeddings.py --model-id EuroBERT/EuroBERT-610m --data flores_agg.csv --out-dir embeddings_agg_610 --no-migrate-bos
```

MultiEURLEX mean embeddings are built by `imbalanced_recall_mel.py` / `mel_close.py` (512-token
truncation) into `multieurlex/emb_mean_<lang>_<split>.npy`.

## 3. The headline experiments

```powershell
# E18 -- out-of-the-box cross-lingual proof  -> proof_mean_alllangs.{json,md,png}
python proof_mean_alllangs.py

# E17 -- which dims carry language + their SHAP contribution -> lang_shap_dims_*.png
python lang_shap_dims.py

# E15 -- recall-first, highly unbalanced: English vs native -> imbalanced_recall.{json,md}, PR plots
python imbalanced_recall.py                 # FLORES sent + agg
python imbalanced_recall_mel.py             # MultiEURLEX (v2 fixes the per-language-label bug)
```

## 4. Supporting sweeps

```powershell
python flores_close.py     # E14b: transform / kNN / translate-train / MLP gap-closers on FLORES
python mel_close.py        # E16: 210m+XGBoost gap-closers on MultiEURLEX
python flores_calib.py     # E19: label-free threshold / prior calibration on FLORES
python flores_ceiling.py   # E19: per-language in-language ceilings + feasibility of 0.9xEN-self
python agg_pr_2models.py   # precision/recall-vs-threshold figures (210m vs 610m)
python rebuild_caches.py   # rebuild SST-2/emotion embeddings + translated text (optional)
```

Long jobs self-detach via `launch_hidden.vbs`; each script writes `<name>.log/.pid/_status.json`.

## 5. Minimal example (the actual proof, ~30 lines)

```python
import numpy as np, xgboost as xgb
from train_flores12 import prepare_sentence, load_sent_embeddings

prep = prepare_sentence()                      # aligned 2006-row FLORES + splits
X = load_sent_embeddings("mean", prep["kept"], 2009)   # {lang: (2006,768)}
Y = prep["y"]; tr, va, te = prep["train_idx"], prep["val_idx"], prep["test_idx"]

def l2norm(A):  n = np.linalg.norm(A, axis=1, keepdims=True); return A / np.clip(n, 1e-9, None)
def perlang_z(A, mu, sd):  return (l2norm(A) - mu) / sd    # mu/sd from the SAME language's train rows

# per-language train mean/std (unlabeled target statistics, same encoder)
stats = {}
for L, XL in X.items():
    Ztr = l2norm(XL[tr]); stats[L] = (Ztr.mean(0), Ztr.std(0) + 1e-6)
def tf(L, idx):  mu, sd = stats[L]; return perlang_z(X[L][idx], mu, sd)

clf = xgb.XGBClassifier(objective="multi:softprob", num_class=12, tree_method="hist",
                        max_bin=64, device="cpu", n_jobs=2, seed=0,
                        max_depth=8, min_child_weight=5, learning_rate=0.1, n_estimators=400)
clf.fit(tf("en", tr), Y[tr])                       # TRAINED ON ENGLISH ONLY
for L in ("en", "it", "de", "fr"):
    acc = (clf.predict(tf(L, te)) == Y[te]).mean()
    print(L, "accuracy", round(acc, 3))
```
