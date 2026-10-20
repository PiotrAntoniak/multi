# COVERAGE — experiment / script / artifact inventory (`multi/`)

Built **2026-10-19** by static inspection (no experiment re-runs). Source of truth for
"documented" = `EXPERIMENTS.md` (391 lines, 24 headings, 32 unique IDs).
Repo: `C:\Users\pedro\Documents\random_projects\multi` (git repo, HEAD `b89bcef`).
No repo file was modified. Only this report was written.

---

## 1. Summary counts

| bucket | count | definition used here |
|---|---:|---|
| **EXPERIMENTS.md headings** | 24 | `### E…` sections + the E2 bullet in "Data & utilities" |
| **unique experiment IDs** | 32 | E1, E1b, E2, E3, E4, E6, E7, E9–E15, E17–E32, E25b, E14b (gaps: **E5, E8, E16**) |
| **Python scripts present** | **69** | 62 in repo root + 7 in `tools/` |
| **DOCUMENTED scripts** | **28** | cited by name/glob in EXPERIMENTS.md and present |
| **UNDOCUMENTED scripts** | **41** | no EXPERIMENTS.md entry: **28 result/protocol-producing** + **13 infra/fetcher/stdout-only** |
| **MISSING scripts/modules** | **12** | 8 cited by EXPERIMENTS.md + 4 support modules imported/referenced by surviving code |
| **MISSING artifacts cited** | 2 groups | `lang_inlp.*` bases (`lang_inlp_agg.npz`, `lang_inlp_agg_610.npz`), `multieurlex_*` outputs |
| **ORPHAN artifact groups** | **12 (≈45 files)** | artifacts with no surviving producer, or scripts whose result artifact is absent/deleted |

Key structural finding: **three documented experiment groups have no surviving script and no
artifact** (E1/E1b, E3/E6/E7, E4). They are described in EXPERIMENTS.md as DONE, but the code
that produced them **never existed in git** and is unrecoverable from history.

Second finding: `agg_pr_2models.py` — the script that produced `agg_pr2.*` and is listed in
`REPORT.md`/`REPRODUCE.md` — **cannot run from a clean checkout**: it imports `agg_pr_methods`
(line 313, 746), which is absent.

---

## 2. Experiment → script → artifact map (the 24 EXPERIMENTS.md entries)

Legend: ✅ documented; ⚠️ documented only via a wildcard/one-line mention; ❌ not documented.
Artifact globs are relative to `multi/`.

| ID | script(s) cited | artifacts present | doc? | notes |
|---|---|---|---|---|
| **E18** | `proof_mean_alllangs.py` | `proof_mean_alllangs.{json,md,png}` | ✅ | matches 11/10 timestamps |
| **E24** | `more_datasets.py` | `more_datasets.{json,md,png}` | ✅ | |
| **E25** | `lang_matrix.py` | `lang_matrix.{json,md}` + 8 `lang_matrix_<ds>.png` (flores_sent/agg, mel, pawsx, sib200, textdetox, toxic, xnli) | ✅ | `PNG_DIR = REPO` (line 132) |
| **E20** | `toxi_proof_clean.py` | `toxi_proof_clean.{json,md,png}`; `models_toxi/` (15 `.ubj` = 5 langs × 3 seeds + `manifest.json`) | ✅ | data: `data/toxi_clean/*.parquet` + `leakage_report.md` |
| **E25b** | `alllang_vs_perlang.py` | `alllang_vs_perlang.{json,md,png}` | ✅ | |
| **E1/E1b** | `langdims_probe.py`, `lang_inlp.*` | only `lang_inlp.npz` (12 KB, orphan) | ✅ | **script missing, never in git**; no census/INLP output artifact |
| **E17** | `lang_shap_dims.py` | `lang_shap_dims.{json,md}` + `lang_shap_dims_{sent,agg,mel}.png` | ✅ | |
| **E12/E13/E26** | `pc_ablation*.py`, `dim_removal.py` | `dim_removal.{json,md}` + 8 `dim_removal_<ds>.png` | ✅ | `pc_ablation*.py` **missing**; `dim_removal.py` is the consolidated reimplementation (docstring line 13) |
| **E3/E6/E7** | `crosslingual_topic*.py`, `crosslingual_inlp.py` | **none** | ✅ | both scripts **missing, never in git** |
| **E9/E10/E11** | `multieurlex_*`, `mel_close.py` | `mel_close.{json,md,log,pid,status}`, `mel_close_plots.png`, `lang_inlp_mel.npz`, `multieurlex/` | ✅ | `multieurlex_*.py` **missing**; `mel_close.py` fits its own INLP basis because the old script is gone (line 25–26). REPORT.md/REPRODUCE.md call this **E16** |
| **E14/E14b** | `flores_close.py` | `flores_close.{json,md,log,pid,status}` | ✅ | |
| **E23** | `translate_test_eval.py` | `translate_test_eval.{json,md,png}` | ✅ | |
| **E4** | `crosslingual_sentiment.py` | **none** | ✅ | **script missing, never in git**; nearest surviving: `rebuild_caches.py` (translated sets) + `xgboost_sst2.py` / `xgboost_emotion.py` (no cross-lingual eval) |
| **E21** | `models_compare.py` | `models_compare.{json,md,png,log,pid,status}` | ✅ | |
| **E28** | `backbone_grid.py` | `backbone_grid.{json,md,png}`; **`toxi_pr_610.png`** | ✅ | `toxi_pr_610.png` is an undocumented side output (`OUT_TOXI610`, line 89) |
| **E22** | `mt_compare.py` | `mt_compare.{json,md,png}` | ✅ | |
| **E19** | `flores_calib.py`, `flores_ceiling.py` | `flores_calib.{json,md}`, `flores_ceiling.{json,md}` | ✅ | |
| **E15** | `imbalanced_recall*.py` | `imbalanced_recall.{json,md,pid,status}`; `imbalanced_recall_mel_v2.{json,md,pid,status}`; `imbalanced_pr_{sent,agg}.png`; `imbalanced_pr_mel_v2.png` | ✅ | stale `imbalanced_recall_mel_status.json` (pre-v2) is an orphan; `VERSION="v2"` (line 65) |
| **E27** | `toxi_pr*.py` (3 scripts) | `toxi_pr.json` (`toxi_pr.py`); `toxi_pr.{md,png}` + `toxi_pr_all.json` (`toxi_pr_all.py`); `toxi_pr_models.{json,md}` + 7 `toxi_pr_<clf>.png` + 3 grids + 2 smoke files (`toxi_pr_models.py`) | ⚠️ | one wildcard covers 3 distinct experiments; only the XGBoost reference is quantified in the table |
| **E30** | `synth_it.py`, `synth_toxic_it.py`; `paraphrase_prompts.py`, `paraphrase_csv.py` (text mention) | `synth_it.{json,md,png}`; `synth_toxic_it.{json,md,png}` | ✅ | toxic arm explicitly "partial (cap=160)" |
| **E31** | `gen_compare.py` | `gen_compare.{json,md,png}` | ✅ | |
| **E32** | `all_paraphrases.py` | `all_paraphrases.{json,md,png}` | ✅ | |
| **E29** | `finetune_lastlayer.py` | `finetune_lastlayer.{json,md,png}` | ✅ | |
| **E2** | `translate_datasets.py`, `rebuild_caches.py` | `translated/*.csv` (sst2/emotion × it/de/fr), `translated_emb/*.npy` | ✅ | `translate_datasets.py` **missing, never in git**; `rebuild_caches.py` performs both MT and embedding caching |
| **E16** | — (no heading) | `mel_close.*` | ❌ | referenced by `REPORT.md:109` and `REPRODUCE.md:62` as the MultiEURLEX gap-closer; EXPERIMENTS.md files it under E9/E10/E11 |
| **E5, E8** | — | — | — | ID gaps; no heading anywhere in docs |

Spot-checks of attribution (timestamps + constants):

- `toxi_pr.png` and `toxi_pr.md` are **13/10 14:03**, while `toxi_pr.json` is **12/10 10:49** →
  the figure/markdown were regenerated by `toxi_pr_all.py` (`OUT_PNG`/`OUT_MD`, lines 53–55),
  the JSON came from `toxi_pr.py`.
- `agg_pr2_*.png` are all 11/10 20:49, exactly `agg_pr_2models.py` `OUT_*` (lines 87–91) and
  `per_method_png()` (line 138, methods `raw/pc/inlp/pc+inlp/l2_thenZ`, line 97).
- `toxi_pr_610.png` is 13/10 00:35, same minute as `backbone_grid.json` → `backbone_grid.py`.
- `imbalanced_pr_mel_v2.png` is 12/10 05:08, same minute as `imbalanced_recall_mel_v2.*` →
  `imbalanced_recall_mel.py` with `VERSION="v2"`.

---

## 3. UNDOCUMENTED — scripts/artifacts with results but no EXPERIMENTS.md entry

28 result/protocol-producing scripts (below) + 13 infra/fetcher/stdout-only scripts
(the infra table has 15 rows; `table1.py` and `table1_agg.py` are listed twice because they are
protocol scripts for ID purposes).

| # | undocumented script | what it computes | evidence (paths / constants) | recommended ID |
|---:|---|---|---|---|
| 1 | `agg_pr_2models.py` | FLORES-agg per-method (raw/pc/inlp/pc+inlp/l2_thenZ) precision/recall-vs-threshold, 210m vs 610m, Optuna per cell, EN+IT | writes `agg_pr2.{json,md}` + 5 method PNGs + 3 grids (lines 87–91, 138). **Imports missing `agg_pr_methods`** (313, 746); mentions missing `agg_pr_lang.py`, `pc_ablation_flores`, `crosslingual_topic_agg` | **E33** |
| 2 | `chunk_pool_mel.py` | MultiEURLEX full-document chunk pooling (mean/max/…) vs 512-token truncation, 210m, `l2_thenZ`+XGB | `chunk_pool_mel.{json,md}` (12/10 04:29) + `chunk_pool_mel_smoke.{json,md}` (11/10 22:49); `--out-json` defaults to repo root (508–509) | **E34** |
| 3 | `it_errors.py` | EN-trained vs IT-trained XGBoost error buckets on the same Italian FLORES test rows (sent/agg) | `it_errors.{json,md}` (12/10 08:08), `OUT_JSON/OUT_MD` lines 48–49 | **E35** |
| 4 | `list_errors.py` | Generalised error lister for any (dataset, source, target); trains source- and target-language models, per-item disagreements | `list_errors.{json,md}` (12/10 08:34); caches `mt_cache_{de,it}2en.json`; `OUT_*` lines 75–76 | **E35** (same entry as #3) |
| 5 | `toxi_proof.py` | Pre-clean toxicity proof on raw `toxi-text-3M` (shipped, leaky splits), recall-first | `toxi_proof.{json,md,png,pid,status}` (12/10 08:54). Superseded by E20's leakage-corrected rerun; kept as provenance | **E36** (provenance note under E20) |
| 6 | `gguf_quant_quality.py` | Q4_K_M vs Q8_0 vs BF16 generation quality on EN→IT (greedy decode) | `gguf_quant_quality.{json,md,partial.json}` (14/10 09:05), `write_report()` 174–179 | **E38** |
| 7 | `gguf_translate.py` | GGUF-only paraphrase→translate CLI (predecessor of `paraphrase_translate.py`) | user `--out-csv`; produced `out_test_q35_gguf.csv`, `out_smoke_qwen25.csv`, `out_sep_test.csv`, `out_div_test.csv` (same CLI shape); documented in `TRANSLATE.md:86` | **E38** |
| 8 | `paraphrase_translate.py` | Dual backend (GGUF + HF) paraphrase→translate, KV-prefix cache, dedup/refill; EN→PL model comparison runs | `out_*.csv`, `cmp_{batched,filtered,seq}.csv`, `seq12.csv`, `seq_2507.csv`, `_prompt_*.txt`, `<out>.meta.json`, `kvcache_meta.csv`; `TRANSLATE.md` | **E38** |
| 9 | `pca_plots.py` | PCA of embeddings by language, raw vs +D vs no-PC1, topic coloring | `pca_plots/pca_lang_<mode>{,_noPC1}.png`, `pca_mean_lang_topic*.png` (lines 74–118) | **E41** |
| 10 | `lang_dims.py` | Per-dim η² + cumulative language-separability curve | `pca_plots/lang_dims_cumulative.png` (line 42) | **E41** |
| 11 | `shap_flores_lang.py` | SHAP of EN topic model vs η² language-dim ranking (precursor of E17) | `shap_flores_lang.{json,md}` (**deleted**, in git history); `pca_plots/shap_vs_langdims_{mean,bos}.png` survive | **E41** |
| 12 | `shap_flores_dropped.py` | SHAP when top language dims are zeroed | `shap_flores_dropped.{json,md}` (**deleted**); `pca_plots/shap_dropped_vs_kept_{mean,bos}.png` survive | **E41** |
| 13 | `table1.py` | FLORES devtest exact-translation retrieval (top-1/MRR/rank), raw vs +D, printed to stdout | no artifact (README.md:117 prints the table) | **E39** |
| 14 | `table1_agg.py` | Same retrieval protocol on FLORES-agg URLs | stdout only (line 118+) | **E39** |
| 15 | `table1_europarl_multilang.py` | Europarl de/es/fr/nl/pt→en retrieval, raw vs +D, all poolings | writes `europarl_multilang_table1.md` + `europarl_multilang_progress.json` (lines 76–77) — **both absent**; consumes `europarl_all_5k.csv` | **E39** |
| 16 | `xgboost_topic_optuna.py` | Main pre-E18 XGBoost topic sweep: 4 poolings, lenient labels, per-tag thresholds, 50 Optuna trials | outputs `xgboost_topic_*`, `xgboost_perlang*`, trial CSVs — **deleted** (see §5) | **E40** |
| 17 | `xgboost_infer.py` | Standalone inference with embedded params, optional `--shift`, `--eval` | `predictions_{transfer,perlang}_*.csv` — **none present** | **E40** |
| 18 | `xgboost_sst2.py` | Optuna XGBoost SST-2 (binary), mean/bos pooling | `xgboost_sst2_trials_{mean,bos}.csv` survive; `xgboost_sst2_results_*.{json,md}` absent | **E40** |
| 19 | `xgboost_emotion.py` | Optuna XGBoost 6-class emotion, mean/bos | `xgboost_emotion_trials_{mean,bos}.csv` survive; result json/md absent | **E40** |
| 20 | `xgboost_flores_lang.py` | EN/IT reference + top-k η² dim zeroing ablation | `.json/.md` deleted (git history) | **E40** |
| 21 | `xgboost_flores_combined.py` | EN+IT combined training; EN→IT raw vs +D | `.json/.md` deleted | **E40** |
| 22 | `xgboost_flores_matrix.py` | Full EN×IT train/eval matrix + dim-zeroing | `.json/.md` deleted | **E40** |
| 23 | `xgboost_flores_drop_optuna.py` | Per-language Optuna with top-50% language dims zeroed | status + summary only; per-lang outputs deleted | **E40** |
| 24 | `clean_toxi.py` | Builds leakage-corrected `data/toxi_clean/` (pool→dedup→80/10/10) | `data/toxi_clean/*.parquet`, `leakage_report.md` (12/10 09:47) | **E20 data** (add to E20 "data" line) |
| 25 | `fetch_europarl_langs.py` | Downloads Europarl v7 xx–en, 5k samples, combined CSV | `europarl_{xx}_en_5k.csv` (absent), `europarl_all_5k.csv` (present, 10 MB) | **E39 data** |
| 26 | `tools/compare_pl.py` | chrF of Qwen3.5-2B vs Qwen2.5-3B EN→PL against human reference | `compare_pl.{json,md}` (14/10 11:43); reads `out_*_pl.csv`, `out_timing.json` | **E38** |
| 27 | `tools/run_translate.py` | Runs `gguf_translate.py` for 2 models, records throughput | `out_qwen35_2b_pl.csv`, `out_qwen25_15b_pl.csv`, `out_timing.json` | **E38** |
| 28 | `tools/run_dual_translate.py` | Same for Qwen3.5-2B vs Qwen2.5-3B | `out_qwen35_2b_pl.csv`, `out_qwen25_3b_pl.csv`, `out_timing.json` | **E38** |

Infra / fetchers / stdout-only (15, no experiment result):

| script | purpose | artifact |
|---|---|---|
| `check_bos_diagnostics.py` | per-layer BOS/mean-pooling stats, 15 EN + 15 IT sentences | stdout only |
| `check_tokens_cos_sim.py` | input-embedding norms + pairing cosine sanity checks | stdout only |
| `table1.py`, `table1_agg.py` | retrieval protocol, console tables | stdout (counted above for ID purposes) |
| `lang_neutralize.py` | shared η² / subspace / ABTT helpers (used by `agg_pr_2models`, `flores_close`, `lang_shap_dims`) | none (library) |
| `lang_metrics.py` | shared η² helpers (used by 6 legacy scripts) | none (library) |
| `model_store.py` | `.ubj` bundle save/load + manifest | `models/<name>/manifest.json` |
| `gpu_limit.py` | cap PyTorch GPU memory (`TORCH_GPU_LIMIT_GB`) | none (library, imported by 26 scripts) |
| `train_flores12.py` | shared FLORES split/label/embedding loader; also trains+saves models | `models_registry.json` |
| `extract_embeddings.py` | materialises mean/lead/bos/eos embeddings | `embeddings*/**/emb_<lang>.npy` |
| `watch_progress.py` | tails xgboost status/trials CSVs → `progress_live.txt` | `progress_live.txt` (absent) |
| `tools/fetch_flores_en_pl.py` | downloads FLORES EN–PL devtest | `flores_en_pl.csv` |
| `tools/fetch_gguf.py`, `fetch_qwen25_gguf.py`, `fetch_qwen25_3b_gguf.py` | GGUF downloads | `models/*.gguf` |

> Note: EXPERIMENTS.md's `## Data & utilities` table covers datasets but **not** any of the
> pipeline scripts (`train_flores12`, `extract_embeddings`, `clean_toxi`, `fetch_*`,
> `model_store`, `gpu_limit`). REPRODUCE.md covers only some of them.

---

## 4. MISSING — scripts cited in EXPERIMENTS.md but absent

| missing script | cited by | git history? | nearest surviving equivalent |
|---|---|---|---|
| `langdims_probe.py` | E1/E1b (EXPERIMENTS.md:127) | **never in git** (`git log --all -- langdims_probe.py` → empty) | none for the probe census; `lang_metrics.py` (η²) + `dim_removal.py` (PC/INLP removal) reproduce parts |
| `lang_inlp.py` (+ `lang_inlp.*` bases) | E1/E1b (:127) | **never in git**; `lang_inlp.npz` exists untracked (0 commits) | `mel_close.py` INLP block (fits `lang_inlp_mel.npz`); `dim_removal.py` INLP arm; `lang_neutralize.py` |
| `pc_ablation*.py` | E12/E13/E26 (:150) | **never in git** | `dim_removal.py` (explicitly "the `pc_ablation_flores.pc_ablation_dirs/apply_dirs` cascade", line 13); missing module `pc_ablation_flores.py` |
| `crosslingual_topic*.py` | E3/E6/E7 (:169) | **never in git**; `crosslingual_topic_agg.py` also referenced inside `agg_pr_2models.py:21,121` | `proof_mean_alllangs.py`, `flores_close.py`, `dim_removal.py` |
| `crosslingual_inlp.py` | E3/E6/E7 (:169) | **never in git** | `dim_removal.py` INLP arm; `lang_neutralize.py` projection; `mel_close.py` INLP |
| `crosslingual_sentiment.py` | E4 (:211) | **never in git** | `rebuild_caches.py` (creates `translated/` + `translated_emb/`); `xgboost_sst2.py` / `xgboost_emotion.py` (EN-only, no transfer eval) |
| `translate_datasets.py` | E2 (:383) | **never in git** | `rebuild_caches.py` — same job (MT + cached embeddings), explicitly supersedes it |
| `multieurlex_*.py` | E9/E10/E11 (:177) | **never in git** | `mel_close.py`; `imbalanced_recall_mel.py`; `chunk_pool_mel.py` (chunking) |

Support modules absent but needed by **surviving** code (not cited in EXPERIMENTS.md):

| missing module | referenced by | impact |
|---|---|---|
| `agg_pr_methods.py` | `agg_pr_2models.py:313,746` (import) | full/`--recompute` runs fail; only `--reemit-from-json` works |
| `agg_pr_lang.py` | `agg_pr_2models.py:12` (docstring) | provenance only |
| `pc_ablation_flores.py` | `agg_pr_2models.py:16,441`; `dim_removal.py:13` | provenance/import; `dim_removal` has its own copy |
| `crosslingual_topic_agg.py` | `agg_pr_2models.py:21,121,185` | provenance only; it produced the missing `lang_inlp_agg.npz` |

**Git-history verdict.** `git log --all --oneline -- <path>` returns **nothing for every one of
these names**. The complete list of `.py` files ever deleted from git history is only
`pertopic_accuracy.py` and `pertopic_accuracy_transfer.py`; every other `.py` in history is
still present. The 2026-10-11 cleanup note (EXPERIMENTS.md:384) says untracked files were lost —
these scripts were **untracked and are unrecoverable from git**. `lang_inlp.npz` is the only
surviving binary from the E1/E1b line (0 commits).

---

## 5. ORPHAN artifacts (no surviving producer) and scripts with no result artifact

Orphan **results**:

| artifact | why orphan | evidence |
|---|---|---|
| `lang_inlp.npz` (12 KB) | no script in the repo writes it; the writers (`langdims_probe.py`, `lang_inlp.py`) are gone | `git log` 0 commits; no `.py` contains `"lang_inlp.npz"` (only `lang_inlp_agg*.npz`, `lang_inlp_mel.npz`) |
| `imbalanced_recall_mel_status.json` | stale status from the pre-v2 run; no matching `imbalanced_recall_mel.json/md`; current script is `VERSION="v2"` | timestamps 11/10 23:31 vs v2 outputs 12/10 05:08 |
| `toxi_pr_610.png` | side output of `backbone_grid.py`, not mentioned by E28 | `backbone_grid.py:89,862` |
| `toxi_pr_models_smoke.json`, `toxi_pr_models.json.smoke` | two smoke snapshots left in root; not referenced by any doc | same minute 14/10 14:06 |
| `chunk_pool_mel_smoke.json/.md` | smoke run wrote into repo root instead of temp | 11/10 22:49; `--smoke` path 508–509 |
| `mt_cache_de2en.json`, `mt_cache_it2en.json` | MT caches from `list_errors.py`; undocumented | `list_errors.py:94` |
| `xgboost_{sst2,emotion}_trials_{mean,bos}.csv` (4) | trial logs survive, but the owning `*_results_*.{json,md}` were deleted; only `watch_progress.py` consumes them | `xgboost_sst2.py:193,347`; `xgboost_emotion.py:205,384` |
| `pca_plots/*.png` (15) | image outputs of `pca_plots.py`, `lang_dims.py`, `shap_flores_*`; `shap_flores_*.{json,md}` deleted, so the underlying numbers are gone | all timestamps 11/10 20:37 (restored) |
| `models_registry.json` | produced by `train_flores12.py`; no EXPERIMENTS.md mention | `train_flores12.py:126` |
| translation snapshots: `cmp_{batched,filtered,seq}.csv`, `seq12.csv`, `seq_2507.csv`, `_prompt_*.txt` (3), `_sample.txt`, `_samples_filtered.txt`, `_samples_seq_bat.txt`, `_seq_examples.txt`, `_seq2507.txt` | ad-hoc dev outputs; no repo script writes these exact names (the CLI only writes user `--out-csv` + `.meta.json`) | untracked; `git status` |
| `flores_en_pl.csv`, `europarl_all_5k.csv`, `compare_pl.*` | inputs/results of the translation line, not covered by EXPERIMENTS.md | `tools/fetch_flores_en_pl.py:47`; `fetch_europarl_langs.py:49` |
| `out_*_pl.csv`, `out_timing.json` | outputs of `tools/run_translate.py` / `run_dual_translate.py`; no experiment entry | `run_translate.py:72`; `run_dual_translate.py:64` |

Scripts with **no surviving result artifact** (results deleted, stdout-only, or model-only):

| script | expected artifact | status |
|---|---|---|
| `xgboost_topic_optuna.py` | `xgboost_topic_*`, `xgboost_perlang*`, trials CSVs | deleted (all appear in `git log --diff-filter=D`) |
| `xgboost_flores_lang.py`, `xgboost_flores_combined.py`, `xgboost_flores_matrix.py` | `.json/.md` | deleted |
| `xgboost_flores_drop_optuna.py` | per-lang `.json`, summary | deleted (status only) |
| `xgboost_infer.py` | `predictions_*.csv` | none present |
| `xgboost_sst2.py`, `xgboost_emotion.py` | `*_results_*.{json,md}` | absent; only trial CSVs |
| `table1.py`, `table1_agg.py` | stdout | by design |
| `table1_europarl_multilang.py` | `europarl_multilang_table1.md`, `..._progress.json` | absent (script never completed or output removed) |
| `pca_plots.py`, `lang_dims.py`, `shap_flores_lang.py`, `shap_flores_dropped.py` | PNG (+ JSON/MD for shap) | PNGs only; shap JSON/MD deleted |
| `check_bos_diagnostics.py`, `check_tokens_cos_sim.py` | stdout | by design |
| `paraphrase_csv.py` | user-selected CSV | template only |
| `watch_progress.py` | `progress_live.txt` | absent |

---

## 6. Prioritized "to add to EXPERIMENTS.md"

Ordered by (a) results exist but no entry, (b) risk of losing provenance, (c) cheapness.

1. **E33 — `agg_pr_2models.py` → `agg_pr2.*` (11/10).** Full precision/recall sweep for
   FLORES-agg, both backbones, 5 transforms. Already advertised in REPORT/REPRODUCE. Also
   record that it is currently **non-runnable** (`agg_pr_methods` missing) so the JSON is the
   only surviving source.
2. **E34 — `chunk_pool_mel.py` → `chunk_pool_mel.*` (12/10).** Directly answers the
   "truncation vs full-document" question for MultiEURLEX and is not mentioned anywhere. Add
   the negative/smoke run note (`chunk_pool_mel_smoke.*`).
3. **E35 — `it_errors.py` + `list_errors.py` → `it_errors.*`, `list_errors.*` (12/10).**
   Only per-item error analysis in the repo; useful for the toxicity/FLORES "why" story.
   Include `mt_cache_*2en.json` as caches.
4. **E16 alias for `mel_close.py`.** REPORT.md and REPRODUCE.md already call it E16 while
   EXPERIMENTS.md files it under E9/E10/E11. Either add `### E16 — MultiEURLEX gap-closers`
   or change the two other docs; record the missing `multieurlex_*.py`.
5. **E36 — `toxi_proof.py` (pre-clean) as provenance under E20.** State explicitly that
   `toxi_proof.*` is the leaky-split predecessor and must not be quoted; E20 supersedes it.
6. **E37 — `toxi_pr_models.py`.** 7 non-XGBoost classifiers × 2 backbones × 2 cells, with the
   percentile-axis fix. E27's wildcard hides it and its table only shows XGBoost numbers.
   Also document `toxi_pr_all.py` as the regenerator of `toxi_pr.png/.md`.
7. **E38 — translation/generation line:** `gguf_quant_quality.py`, `gguf_translate.py`,
   `paraphrase_translate.py`, `tools/run_translate.py`, `tools/run_dual_translate.py`,
   `tools/compare_pl.py` (+ `TRANSLATE.md`). One entry, artifacts: `gguf_quant_quality.*`,
   `out_*_pl.csv`, `out_timing.json`, `compare_pl.*`, `cmp_*.csv`.
8. **E39 — retrieval protocol:** `table1.py`, `table1_agg.py`,
   `table1_europarl_multilang.py` (+ `fetch_europarl_langs.py`,
   `tools/fetch_flores_en_pl.py`). Currently only README.md documents it; the Europarl
   markdown output is missing, so the result is effectively unrecorded.
9. **E40 — legacy XGBoost suite:** `xgboost_topic_optuna.py`, `xgboost_infer.py`,
   `xgboost_sst2.py`, `xgboost_emotion.py`, `xgboost_flores_{lang,combined,matrix,drop_optuna}.py`.
   Results were deleted; add a "superseded / results not retained" paragraph so the scripts
   are not mistaken for live evidence, and preserve the 4 trial CSVs' meaning.
10. **E41 — dimensionality diagnostics:** `pca_plots.py`, `lang_dims.py`,
    `shap_flores_lang.py`, `shap_flores_dropped.py`. These are the precursors of E17;
    the SHAP JSON/MD are gone, so document that only PNGs survive.
11. **Add the missing-module/provenance block** (from §4) to the cleanup note at
    EXPERIMENTS.md:384: list `langdims_probe.py`, `lang_inlp.py`, `pc_ablation*.py`,
    `crosslingual_topic*.py`, `crosslingual_inlp.py`, `crosslingual_sentiment.py`,
    `translate_datasets.py`, `multieurlex_*.py`, `agg_pr_methods.py`, `agg_pr_lang.py`,
    `pc_ablation_flores.py`, `crosslingual_topic_agg.py` — none ever tracked, all
    unrecoverable; `lang_inlp.npz` is the sole surviving binary.
12. **Close the ID gaps.** E5, E8 and E16 are referenced nowhere in EXPERIMENTS.md. State
    whether they were dropped, renamed, or are free for reuse before assigning E33+.
