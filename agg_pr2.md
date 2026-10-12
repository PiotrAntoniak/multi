# FLORES-agg (11-class): P/R vs threshold for EuroBERT-210m vs 610m, English vs Italian (Optuna-tuned)

**Dataset:** flores_agg.csv (FLORES URL-aggregated, 11-class topic). **Model:** english XGB multi:softprob, Optuna-tuned per (backbone, method); trained on EN train+val; targets EN (in-language) and IT (cross-lingual), 11 classes, train/val/test = 276/140/139 (EN train+val = 416 rows). Backbones: 210m (768-d) LEFT, 610m (1152-d) RIGHT.

Class c is positive for a document iff `p_c >= t`; sweep t over 0.00..1.00 step 0.02. Micro P/R/F1 pooled over all (document, class) pairs. Each (backbone, method) trains one ENGLISH 11-class XGB on EN train+val and is evaluated on the **English test** (`en`, in-language) and the **Italian test** (`it`, cross-lingual transfer).

## Tuning protocol

- Optuna TPESampler(seed=0), **20 trials** per (backbone, method) cell; search space = `xgboost_topic_optuna._optuna_search_space with n_estimators capped at 500`.
- Base params: `{'objective': 'multi:softprob', 'num_class': 11, 'tree_method': 'hist', 'max_bin': 64, 'device': 'cpu', 'n_jobs': 2, 'seed': 0}`.
- Each trial trains on **EN train** (276) and scores **EN-val** (140) with objective = EN-val (140) OVR-macro AUC (sklearn roc_auc_score multi_class=ovr, average=macro); fallback to single-label macro-F1 only if AUC is non-finite.
- Best params retrained on **EN train+val** (416), then evaluated on EN test + IT test with the threshold sweep.

## Methods

- `raw`: no transform
- `pc`: iterative GLOBAL PC ablation k=2, fit on the stacked-language TRAIN pool (pc_ablation_flores.pc_ablation_dirs / apply_dirs), per backbone
- `inlp`: project out the backbone-matched FLORES language-subspace basis (210m=lang_inlp_agg.npz 3 dirs; 610m=lang_inlp_agg_610.npz)
- `pc+inlp`: pc (k=2) then project out the inlp subspace
- `l2_thenZ`: L2-normalize then per-language TRAIN z-score (stats from that language's own train_idx rows)

**610m INLP basis** (`lang_inlp_agg_610.npz`, loaded): n_removed=3, rounds=1, final probe acc=0.26553625340262316, reached chance(<=0.30)=True.

## Optuna tuning result per (backbone, method)

| backbone | method | best val AUC | val macro-F1 | metric | best params |
|---|---|---|---|---|---|
| `210m` | `raw` | **0.8935** | 0.5268 | val_ovr_macro_auc | `d6 lr0.024 n286 ss0.70 cs0.21 mcw2 rl0.422 g0.69` |
| `210m` | `pc` | **0.8905** | 0.4439 | val_ovr_macro_auc | `d7 lr0.012 n249 ss0.71 cs0.12 mcw4 rl0.775 g2.17` |
| `210m` | `inlp` | **0.8864** | 0.4282 | val_ovr_macro_auc | `d7 lr0.010 n223 ss0.71 cs0.11 mcw4 rl0.585 g1.65` |
| `210m` | `pc+inlp` | **0.8865** | 0.4301 | val_ovr_macro_auc | `d7 lr0.010 n223 ss0.71 cs0.11 mcw4 rl0.585 g1.65` |
| `210m` | `l2_thenZ` | **0.8930** | 0.4174 | val_ovr_macro_auc | `d7 lr0.012 n249 ss0.71 cs0.12 mcw4 rl0.775 g2.17` |
| `610m` | `raw` | **0.9033** | 0.4939 | val_ovr_macro_auc | `d6 lr0.024 n286 ss0.70 cs0.21 mcw2 rl0.422 g0.69` |
| `610m` | `pc` | **0.9039** | 0.5715 | val_ovr_macro_auc | `d8 lr0.078 n396 ss0.62 cs0.30 mcw2 rl0.0153 g0.59` |
| `610m` | `inlp` | **0.9058** | 0.4944 | val_ovr_macro_auc | `d6 lr0.024 n286 ss0.70 cs0.21 mcw2 rl0.422 g0.69` |
| `610m` | `pc+inlp` | **0.9044** | 0.4775 | val_ovr_macro_auc | `d7 lr0.024 n196 ss0.89 cs0.26 mcw1 rl1.53 g0.08` |
| `610m` | `l2_thenZ` | **0.8987** | 0.5827 | val_ovr_macro_auc | `d8 lr0.078 n396 ss0.62 cs0.30 mcw2 rl0.0153 g0.59` |

### Ranking by best val AUC vs by Italian test F1

- `210m` by val AUC: `raw`(0.8935), `l2_thenZ`(0.8930), `pc`(0.8905), `pc+inlp`(0.8865), `inlp`(0.8864).
  by IT test F1: `l2_thenZ`(0.5078), `raw`(0.4710), `pc+inlp`(0.4413), `inlp`(0.4377), `pc`(0.4341). Orderings DIFFER.
- `610m` by val AUC: `inlp`(0.9058), `pc+inlp`(0.9044), `pc`(0.9039), `raw`(0.9033), `l2_thenZ`(0.8987).
  by IT test F1: `l2_thenZ`(0.6103), `raw`(0.5398), `inlp`(0.5116), `pc+inlp`(0.5053), `pc`(0.4848). Orderings DIFFER.

## Per (backbone, method, language): F1-opt threshold, precision and recall

`val AUC` is the best Optuna EN-val OVR-macro AUC for that (backbone, method) cell. `F1-opt t` / `P` / `R` / `F1` are on the TEST split at that language's own F1-optimal threshold.

| backbone | method | language | val AUC | F1-opt t | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| `210m` | `raw` | en (EN->EN) | 0.8935 | 0.28 | 0.6794 | 0.6403 | 0.6593 |
| `210m` | `raw` | it (EN->IT) | 0.8935 | 0.20 | 0.4269 | 0.5252 | 0.4710 |
| `210m` | `pc` | en (EN->EN) | 0.8905 | 0.22 | 0.6720 | 0.6043 | 0.6364 |
| `210m` | `pc` | it (EN->IT) | 0.8905 | 0.22 | 0.4706 | 0.4029 | 0.4341 |
| `210m` | `inlp` | en (EN->EN) | 0.8864 | 0.20 | 0.6176 | 0.6043 | 0.6109 |
| `210m` | `inlp` | it (EN->IT) | 0.8864 | 0.22 | 0.4603 | 0.4173 | 0.4377 |
| `210m` | `pc+inlp` | en (EN->EN) | 0.8865 | 0.22 | 0.6583 | 0.5683 | 0.6100 |
| `210m` | `pc+inlp` | it (EN->IT) | 0.8865 | 0.26 | 0.6351 | 0.3381 | 0.4413 |
| `210m` | `l2_thenZ` | en (EN->EN) | 0.8930 | 0.20 | 0.6312 | 0.6403 | 0.6357 |
| `210m` | `l2_thenZ` | it (EN->IT) | 0.8930 | 0.22 | 0.5556 | 0.4676 | 0.5078 |
| `610m` | `raw` | en (EN->EN) | 0.9033 | 0.26 | 0.6519 | 0.6331 | 0.6423 |
| `610m` | `raw` | it (EN->IT) | 0.9033 | 0.20 | 0.5200 | 0.5612 | 0.5398 |
| `610m` | `pc` | en (EN->EN) | 0.9039 | 0.28 | 0.6462 | 0.6043 | 0.6245 |
| `610m` | `pc` | it (EN->IT) | 0.9039 | 0.20 | 0.4557 | 0.5180 | 0.4848 |
| `610m` | `inlp` | en (EN->EN) | 0.9058 | 0.26 | 0.6617 | 0.6331 | 0.6471 |
| `610m` | `inlp` | it (EN->IT) | 0.9058 | 0.28 | 0.5546 | 0.4748 | 0.5116 |
| `610m` | `pc+inlp` | en (EN->EN) | 0.9044 | 0.22 | 0.5570 | 0.6331 | 0.5926 |
| `610m` | `pc+inlp` | it (EN->IT) | 0.9044 | 0.24 | 0.5000 | 0.5108 | 0.5053 |
| `610m` | `l2_thenZ` | en (EN->EN) | 0.8987 | 0.26 | 0.6250 | 0.6475 | 0.6360 |
| `610m` | `l2_thenZ` | it (EN->IT) | 0.8987 | 0.26 | 0.6241 | 0.5971 | 0.6103 |

## EN vs IT gap per method, 210m vs 610m

Gap = EN peak F1 minus IT peak F1 (each at its own F1-optimal t). Positive => Italian below English. `d(gap)` = 610m gap minus 210m gap; **negative => 610m narrows the EN-IT gap**.

| method | 210m dF1 | 610m dF1 | d(gap) 610m-210m | 210m IT F1 | 610m IT F1 |
|---|---|---|---|---|---|
| `raw` | +0.1883 | +0.1025 | **-0.0857** | 0.4710 | 0.5398 |
| `pc` | +0.2023 | +0.1397 | **-0.0626** | 0.4341 | 0.4848 |
| `inlp` | +0.1732 | +0.1354 | **-0.0377** | 0.4377 | 0.5116 |
| `pc+inlp` | +0.1687 | +0.0873 | **-0.0815** | 0.4413 | 0.5053 |
| `l2_thenZ` | +0.1279 | +0.0257 | **-0.1022** | 0.5078 | 0.6103 |
| **mean over methods** | **+0.1721** | **+0.0981** | **-0.0739** | | |

## Best method per backbone

- `210m`: best Italian transfer = **`l2_thenZ`** (F1-opt t=0.22, F1=0.5078, P=0.5556 R=0.4676); best val AUC = `raw` (0.8935).
- `610m`: best Italian transfer = **`l2_thenZ`** (F1-opt t=0.26, F1=0.6103, P=0.6241 R=0.5971); best val AUC = `inlp` (0.9058).

## Figures

- `agg_pr2_grid_precision.png` -- combined grid, micro PRECISION vs threshold (rows = methods; columns = 210m left / 610m right; English + Italian).
- `agg_pr2_grid_recall.png` -- combined grid, micro RECALL vs threshold (same layout).
- `agg_pr2_grid.png` -- combined grid, F1 vs threshold (same layout).
- `agg_pr2_<method>.png` -- per-method 2x2 (rows = Precision/Recall; columns = 210m/610m; English + Italian).

## Does 610m move Italian closer to English?

- Best Italian F1: 210m `l2_thenZ`=0.5078 vs 610m `l2_thenZ`=0.6103 (delta +0.1025).
- Mean EN-IT peak-F1 gap: 210m +0.1721 -> 610m +0.0981 (change -0.0739).
- Methods where the gap narrows under 610m: 5/5.
- Interpretation: the 610m backbone raises Italian F1 and narrows the EN-IT gap (Italian moves closer to English).

Elapsed: 793.4 s.
