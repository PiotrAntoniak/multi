# EXPERIMENTS — Cross-lingual study (single tracking file)

**Date:** 2026-10-20
**Repo:** `C:\Users\pedro\Documents\random_projects\multi` (audited against git HEAD `b89bcef`)
**Status:** audited consolidated record. This file replaces the pre-audit `EXPERIMENTS.md`.
**Terminology:** any reference below to `EXPERIMENTS.md`, "the old table" or "the old entry" means the
pre-audit version at `b89bcef`, not this document.

**Scope.** This document consolidates the audited record of the `multi/` cross-lingual program: one
English-trained classifier (mostly XGBoost on mean-pooled EuroBERT-210m / 610m features) applied to
Italian, German, French, Spanish and back, across FLORES topic (sentence and URL-aggregate),
MultiEURLEX, XNLI, PAWS-X, SIB-200, textdetox, SST-2 / emotion and toxicity, plus the diagnostics,
classifier, MT, calibration, retrieval, generation and fine-tuning work around it. It holds **43 IDs**
(E1–E42, including E1b, E14b, E25b and the E16 alias) assembled from 34 audited entry files and
`experiments_audit/COVERAGE.md`. Every number is the verified number from its part; no experiment was
re-run during assembly. Line-level evidence, command transcripts and discrepancy lists stay in
`experiments_audit/*.audit.md` and are not repeated here.

---

## Pass bar: 0.975

The project bar is **0.975**: a target (or method) result must reach at least 97.5% of its reference,

```
ratio = numerator / denominator >= 0.975
```

**The 0.90 bar is obsolete.** Any statement of the form "comparable at 0.9", "3/3 at the 0.9 bar" or
"passes at 0.9" is historical. It is not quoted as a pass anywhere in this document. Where an entry
previously passed at 0.90 and fails at 0.975, that is stated explicitly (E18, E23, E24, E25, E36).

**Ratios are computed on unrounded values.** A value displayed at 2 dp (e.g. E20 fr `0.90`) can be
below the bar (0.8959). Never read a rounded table cell as a pass.

### Denominator per experiment

| experiment(s) | numerator | denominator | metric |
|---|---|---|---|
| E18 | target language | English self | AUC (sent, agg); micro-F1 with quantile-matched thresholds (mel) |
| E19 | target language, calibrated | English self, **two variants**: train+val and train-only | single-label micro-F1 / macro-F1 |
| E20 | target test, EN-trained | English self; and EN-trained vs **native** | AUC |
| E21 | EN→IT | EN-self of the **same** model | macro-F1; OVR-macro AUC |
| E23 | MT-to-English; untranslated | English self; in-language model | macro-F1; OVR-macro AUC |
| E24 | target language | English self | macro-F1 (XNLI, SIB-200); AUC (PAWS-X, textdetox) |
| E25 | off-diagonal cell mean | diagonal (in-language) cell mean | macro-F1 |
| E25b | all-langs model | per-language model | macro-F1 |
| E27 / E37 | EN model on IT test | IT model on IT test | precision at recall >= 0.90 |
| E36 | target; EN-trained | English self; native | AUC |
| E15 | EN FPR at recall 0.90 | native FPR at recall 0.90; pass bound `<= 1/0.975 = 1.025641` | FPR@R90 (lower is better) |
| E29 | fine-tuned | frozen, same cell and metric; also target/EN-self | AUC, macro-F1 |
| E30 | synthetic IT / mixed | real-IT, size-matched on FLORES | macro-F1 (FLORES); F1(pos) (toxic) |
| E31 | one generator | `real_it` | macro-F1 |
| E32 | union of generators | `real_it`; best single generator | macro-F1 |
| E38 | quantized model | BF16 | chrF, chrF++, BLEU |
| E42 | A(IT)/B(IT), A/C, D/B, D/C, D/A(EN), D′(IT)/B, D′(IT)/C, B_EN/A_EN | see the E42 entry | OVR-macro AUC |
| E9 / E10 / E11 | target language | English self; in-language ceiling | micro-F1 |
| E14 / E14b | method | in-language reference | OVR micro-F1 (primary); macro-F1 (secondary) |
| E4 | target language | EN-self | macro-F1 |
| E22 | multilingual MT | bilingual MT | BLEU (chrF secondary) — a system-to-system ratio, not target/English |
| E33 | IT peak F1 | EN peak F1 (derived by the audit, not stored) | micro-F1 |
| E34 | target language | English self | micro-F1, macro-F1, AUC |

**Bar is N/A** (no numerator/denominator pair exists) for: **E1/E1b** (probe accuracy),
**E2** (data preparation), **E3/E6/E7** (absolute macro-F1 only, no EN-self stored),
**E12/E13/E26** (absolute XL-F1; derived ratios are context only), **E17** (Spearman / SHAP mass),
**E28** (absolute XL-mean F1), **E35** (descriptive error buckets), **E39** (rank metrics),
**E40** (superseded suite), **E41** (diagnostics). "N/A" is not a pass.

---

## How to read this document

Every entry starts with a `**Script to reproduce:**` line directly under its heading, giving the
script, the command (or the reason no command exists), and the reproduction status.
The pre-audit file sometimes carried that line above the heading; here it has been moved under it.
Duplicate headings and duplicate script lines were removed at assembly.

**Reproduction status legend**

| status | meaning |
|---|---|
| **REPRODUCIBLE** | At least one cell was re-run exactly at stored precision from repo code, or every stored cell was re-derived from the committed artifact. |
| **PARTIAL** | Artifact numbers were verified or a subset re-ran, but at least one of: a cited script is missing; the run is not bit-reproducible; a cache outside the repo is required; a stage was not re-executed. |
| **NOT AVAILABLE (UNK)** | No producing script and no result artifact survive. The numbers are documentation-only and cannot be checked. `UNK` is the status token used by `COVERAGE.md`. |

Other conventions:

- `->` marks a direction or a change; `>=` marks a comparison.
- `†` in a table marks an omitted or non-reproducible cell, as noted in that entry.
- A "ceiling" or "in-language reference" is a target-trained model, **not** an upper bound. Several
  methods exceed it, which is why E14 and E19 both warn against the word "ceiling".
- Metrics are not interchangeable across entries: E18 mel is micro-F1 with quantile-matched
  thresholds; E25 mel is macro-F1 at 0.5; E24 PAWS-X is AUC with train-only. Do not compare those
  numbers directly.
- `perlang_z` and `l2_thenZ` use **unlabeled target train rows**. An "English-only" model under
  those transforms is not strictly target-free.

---

## Coverage index

| ID | experiment | script to reproduce | status | 0.975 result |
|---|---|---|---|---|
| E1 | Language-dimension census (4-way probe, η²) | NOT AVAILABLE: `langdims_probe.py` | NOT AVAILABLE (UNK) | N/A; probe 1.000 full, 0.999 after zeroing top-557 η² dims |
| E1b | INLP language-subspace removal | NOT AVAILABLE: `lang_inlp.py` | NOT AVAILABLE (UNK) | N/A; INLP claim (~30 dirs) unverified |
| E2 | Translated SST-2 / emotion eval sets | `rebuild_caches.py` | PARTIAL | N/A (data prep); supports E4 |
| E3 | FLORES topic cross-lingual (+ neutralization) | NOT AVAILABLE: `crosslingual_topic*.py` | NOT AVAILABLE (UNK) | N/A; claims 0.234 -> 0.334 unverified |
| E4 | Sentiment / emotion cross-lingual | NOT AVAILABLE: `crosslingual_sentiment.py` | NOT AVAILABLE (UNK) | 0/12 cells; best SST-2 fr centroid 0.819 |
| E6 | FLORES topic, INLP arm | NOT AVAILABLE: `crosslingual_inlp.py` | NOT AVAILABLE (UNK) | N/A; no separate claim |
| E7 | FLORES topic, second variant | NOT AVAILABLE: `crosslingual_topic*.py` | NOT AVAILABLE (UNK) | N/A; no separate claim |
| E9 | MultiEURLEX long-doc, 210m + 610m | `mel_close.py` | PARTIAL | 610m 0.996 / 0.993 vs native ceiling (numerator unverified); 0.946 / 0.940 vs EN self; 210m fails |
| E10 | MultiEURLEX, threshold calibration | `mel_close.py` | PARTIAL | 210m l2_thenZ+qm: 0.922 / 0.923 vs EN self — fail |
| E11 | MultiEURLEX, 610m arm | NOT AVAILABLE: `multieurlex_*.py` | NOT AVAILABLE (UNK) | 610m 0.711 / 0.707 text-only; only passes vs unverified ceiling |
| E12 | Leading-PC removal (global PCA k=0..5) | `dim_removal.py` | PARTIAL | N/A; best headline XL/EN-self 0.895 (FLORES-agg 610m) |
| E13 | PC-ablation cascade (original) | NOT AVAILABLE: `pc_ablation*.py` | NOT AVAILABLE (UNK) | N/A; no separate artifact survives |
| E14 | FLORES gap-closers, primary micro-F1 | `flores_close.py` | REPRODUCIBLE | 5 methods 3/3 on both datasets (micro); no pure EN transform passes |
| E14b | FLORES gap-closers, macro-F1 view | `flores_close.py` | REPRODUCIBLE | kNN / translate-train / `mlp_align` 6/6 target cells |
| E15 | Recall-first (FPR at 90% recall) | `imbalanced_recall.py` / `imbalanced_recall_mel.py` | REPRODUCIBLE | FLORES PASS 1.005 (12/24 cells); MultiEURLEX FAIL 1.196 (2/16) |
| E16 | Alias for the MultiEURLEX gap-closers | `mel_close.py` | PARTIAL | Same as E9/E10/E11; `multieurlex_*.py` missing |
| E17 | Language dims vs SHAP contribution | `lang_shap_dims.py` | REPRODUCIBLE | N/A; Spearman −0.076 / −0.059 / −0.019 |
| E18 | Headline out-of-the-box transfer | `proof_mean_alllangs.py` | REPRODUCIBLE | 0/8 target cells; best FLORES-agg IT 0.967 |
| E19 | Calibration and ceiling feasibility | `flores_calib.py` + `flores_ceiling.py` | PARTIAL | sent 6/6 infeasible; agg 3/6 infeasible vs 0.975 bar |
| E20 | Toxicity, clean split | `toxi_proof_clean.py` | REPRODUCIBLE | 0/4 on both bars; best fr EN/native 0.896 |
| E21 | Non-XGBoost classifiers | `models_compare.py` | PARTIAL | F1 3/34, AUC 5/34; no non-degenerate both-metric pass |
| E22 | Bilingual vs multilingual MT | `mt_compare.py` | PARTIAL | 0/3 BLEU (0.778 / 0.719 / 0.756); chrF 0/3 |
| E23 | Translate-test | `translate_test_eval.py` | REPRODUCIBLE | 4/6 macro-F1, 6/6 AUC vs EN; 6/6 both vs in-language |
| E24 | XNLI, PAWS-X, textdetox, SIB-200 | `more_datasets.py` | PARTIAL | 1/13 cells (SIB-200 fr 0.995); 0/4 datasets |
| E25 | Full language × language matrices | `lang_matrix.py` | REPRODUCIBLE | 2/16 rows (PAWS-X 1.024 / 0.994, near chance); gap shrinks on 8/8 by absolute value |
| E25b | All-languages vs per-language training | `alllang_vs_perlang.py` | REPRODUCIBLE | 6/6 cells; FLORES 1.032–1.227, toxicity 0.979 / 0.981 |
| E26 | Leading-dimension removal (third ID) | `dim_removal.py` | PARTIAL | N/A; shares E12's artifact; no separate claim |
| E27 | Toxicity precision/recall, XGBoost | `toxi_pr.py` / `toxi_pr_all.py` | PARTIAL | 0/2 backbones; 0.708 / 0.771 (JSON); reverse 0.720 / 0.789 |
| E28 | Backbone × dataset × method grid | `backbone_grid.py` | PARTIAL | N/A (absolute); mel gain largest, toxic none |
| E29 | Fine-tune last layer + linear head | `finetune_lastlayer.py` | PARTIAL | FT/frozen 20/26; target/EN-self 0/20 |
| E30 | Synthetic Italian from English | `synth_it.py` / `synth_toxic_it.py` | REPRODUCIBLE (FLORES) | FLORES synth 0.989 / 0.983 and mixed 1.026 / 0.986 pass; toxic 0/8 |
| E31 | Generator comparison | `gen_compare.py` | PARTIAL | 2/3 generators (0.994, 0.980; Qwen3.5-2B 0.956); toxic NOT DONE |
| E32 | Union of all generators' paraphrases | `all_paraphrases.py` | REPRODUCIBLE (FLORES) | 1.145 vs real-IT, 1.151 vs best single; 3× volume confound; toxic 2/3 |
| E33 | FLORES-agg P/R vs threshold, 210m vs 610m | NOT AVAILABLE for full rerun: `agg_pr_2models.py` | PARTIAL (re-emit only) | N/A; derived IT/EN tops at 0.9595, 0/10 cells |
| E34 | Full-document chunk pooling vs truncation | `chunk_pool_mel.py` | PARTIAL | 0/24 cells; best `concat` FR AUC 0.9638 |
| E35 | EN- vs IT-trained error buckets | `it_errors.py` / `list_errors.py` | REPRODUCIBLE | N/A; accuracy ratio EN/IT 0.857 (sent), 0.753 (agg) |
| E36 | Toxicity on the ORIGINAL leaky splits | `toxi_proof.py` | PARTIAL (provenance only) | target/EN 1/4 (es 0.986); EN/native 1/4 (de 1.062); both are artifacts, do not quote |
| E37 | Seven classifiers at the toxicity bar | `toxi_pr_models.py` | PARTIAL | 13/14 fail; GaussianNB 610m 1.0675 nominal only |
| E38 | GGUF quantization + EN→PL paraphrase→translate | `gguf_quant_quality.py` | PARTIAL | 6/6 quant cells (min 0.9904); EN→PL N/A |
| E39 | Exact-translation retrieval | `table1.py` / `table1_agg.py` / `table1_europarl_multilang.py` | PARTIAL | N/A; FLORES mean +D top-1 83.9–90.0% |
| E40 | Legacy pre-E18 XGBoost suite | NOT AVAILABLE for the tables; archived at `aa3f6f6` | NOT AVAILABLE (UNK), superseded | N/A |
| E41 | PCA / η² / SHAP diagnostics | `pca_plots.py`, `lang_dims.py`, `shap_flores_*.py` | PARTIAL | N/A |
| E42 | Raw EN vs EN+IT, IT control, translate-train, translate-test | `en_vs_enit_raw.py`, `en_vs_enit_dclean.py`, `en_vs_enit_ttest.py` | REPRODUCIBLE | A(IT)/B 0.9414 fail; E/B 1.0057, E/C 0.9994; F/E 1.0227 (F best), F/E macro 1.2042 |

**ID gaps.** E5 and E8 have no heading in `EXPERIMENTS.md`, `REPORT.md` or `REPRODUCE.md`; they are
free IDs, not lost experiments. E16 has no heading in `EXPERIMENTS.md` but is used by `REPORT.md:109`
and `REPRODUCE.md:62` for the MultiEURLEX gap-closers; it is kept here as an alias.

---

# PART A — Headline transfer

The seven entries that carry the main claim: one English model, model-free transforms, 0.975 bar.
Order: the two out-of-the-box proofs (E18, then the four extra datasets in E24), the translate-test
escape hatch (E23), the full matrices (E25, E25b), the clean toxicity counter-example (E20), and the
raw-feature / translate-train control (E42).

## E18 — Out-of-the-box cross-lingual proof — DONE (verified 2026-10-19)

**Script to reproduce:** `proof_mean_alllangs.py` — `python proof_mean_alllangs.py --dataset flores`
and `python proof_mean_alllangs.py --dataset mel --launch`. Reproduction status: **REPRODUCIBLE**
(two cells re-ran exactly).

**Question.** Can ONE English-trained XGBoost, fed only mean-pooled EuroBERT-210m features and
model-free transforms, reach >= 0.975 × the English self score in every target language?

**Script / outputs.**

- `proof_mean_alllangs.py` -> `proof_mean_alllangs.json` (numbers), `.md` (full tables), `.png`.
- `--dataset flores` computes sent + agg; `--dataset mel --launch` computes MultiEURLEX detached.
- `--render` rebuilds md/png from the JSON only.
- Resumable: datasets already in the JSON are skipped. **Delete the JSON to recompute.**

**Inputs.**

| dataset | embeddings | labels / split | model |
|---|---|---|---|
| sent | `embeddings/mean/emb_{en,it,de,fr}.npy`, 2009x768, sliced to 2006 kept rows | `flores200_en_it_de_fr.csv`; 12 classes; 997/504/505; parallel rows | one EN model |
| agg | `embeddings_agg/mean/emb_{en,it,de,fr}.npy`, 562x768 | `flores_agg.csv`; 11 classes; 276/140/139 | one EN model |
| mel | `multieurlex/emb_mean_{en,de,fr}_{train,test}.npy`, 11000/5000x768 | `multieurlex/{en,de,fr}_{train,test}.jsonl`, `multi_hot` (21); re-split 10000/1000/5000, seed 0 | one EN model |

MultiEURLEX test rows are parallel; all 5000 `multi_hot` vectors match across en/de/fr. Train rows are
NOT parallel across languages. The script uses each target language's train rows only for feature
statistics, never labels. The official `*_validation.jsonl` is not used.

**Feature ladder (same mean embeddings throughout).**

| config | operation | target data |
|---|---|---|
| `raw` | none | no |
| `l2` | row L2-normalize | no |
| `global_center` | subtract EN-train mean | no |
| `global_z` | z-score with EN-train mean/std | no |
| `perlang_center` | subtract each language's own train mean | unlabeled target train rows |
| `perlang_z` | L2, then each language's own train z-score | unlabeled target train rows |

`global_center` and `global_z` are exact no-ops for XGBoost: per-feature affine transforms do not
change tree splits. The JSON confirms metrics identical to `raw` on all three datasets. Treat them as
one control, not two results.

**Protocol.** Fixed XGBoost (depth 8, 400 trees, lr 0.1, hist, max_bin 64, seed 0). Train on English
train rows. Score every language's test rows with that same model. "Comparable" = every
target/English ratio >= **0.975** on the dataset's headline metric. Headline: AUC (sent, agg);
micro-F1 with quantile-matched target thresholds (mel). Argmax-F1 ratios are also reported and never
reach 0.975 (best ~0.82).

**Results.**

| dataset | metric | config | EN-self | targets | min ratio | >=0.975 |
|---|---|---|---|---:|---|---:|:--:|
| sent | AUC | `raw` | 0.856 | it .759 / de .763 / fr .768 | 0.886 | no |
| sent | AUC | `perlang_center` | 0.856 | .778 / .770 / .778 | 0.899 | no |
| sent | AUC | `perlang_z` | 0.849 | .782 / .786 / .778 | 0.917 | no |
| agg | AUC | `raw` | 0.895 | .824 / .807 / .795 | 0.888 | no |
| agg | AUC | `perlang_center` | 0.895 | .859 / .815 / .836 | 0.911 | no |
| agg | AUC | `perlang_z` | 0.888 | .859 / .843 / .835 | 0.940 | no |
| mel | micro-F1-qm | `raw` | 0.717 | de .535 / fr .560 | 0.746 | no |
| mel | micro-F1-qm | `perlang_center` | 0.717 | .658 / .659 | 0.918 | no |
| mel | micro-F1-qm | `perlang_z` | 0.718 | .669 / .667 | 0.928 | no |

Full per-config tables: `proof_mean_alllangs.md`.

**Verdict.** At the **0.975** bar, **0/8 target cells pass**. The closest cell is FLORES-agg Italian
at **0.967** (`perlang_z`), still 0.8 pp short. Per-language unlabeled train statistics are
**necessary** for the improvement (no English-stats-only config exceeds 0.899), but they are **not
sufficient** for 97.5% parity on these datasets. The old 0.90 bar passed at 8/8 on AUC; that result
does not survive at 0.975.

**Caveats.**

- Single seed, no confidence intervals. FLORES test rows: 505 (sent), 139 (agg).
- "Comparable" is metric-specific. Argmax / global-threshold F1 does not reach 0.975; see E19 for the
  label-noise / ceiling argument.
- Reproduced on 2026-10-19 for `sent` (`raw`, `perlang_z`) at stored precision (4 dp).
- Resumable JSON: stale results persist unless the JSON is deleted.

---

## E23 — Translate-test — DONE (verified 2026-10-19)

**Script to reproduce:** `translate_test_eval.py` — `python translate_test_eval.py --launch` (or
`--backbones 210m --targets it,de,fr`). Reproduction status: **REPRODUCIBLE**.

**Question.** Does machine-translating the FLORES it/de/fr **test** text into English recover the
English-level score of ONE English-trained XGBoost — and how does that compare with (a) training
in-language on the target's own train+val and (b) untranslated transfer — on the **same 505
documents**?

**Script / outputs.**

- `translate_test_eval.py` -> `translate_test_eval.json` (all numbers), `.md` (tables), `.png`
  (macro-F1 bars).
- `--launch` runs both backbones detached and hidden; `--backbones 210m --targets it,de,fr` narrows it.
- MT hypotheses and translated-English embeddings are cached under
  `%TEMP%\opencode\translate_test_eval\`; the 610m run reuses the 210m MT cache, so both backbones
  see identical translated text.
- Outputs are overwritten on rerun and models are refit. Delete the JSON to recompute; delete the
  cache to re-run MT.

**Inputs.**

| item | value |
|---|---|
| dataset | FLORES `sent` (12-class); `flores200_en_it_de_fr.csv`, 2006 rows; train = `dev` 997, val = 504, test = 505 (seed-0 permutation of the 1009 `devtest` rows, parallel across languages) |
| embeddings | EuroBERT mean: `embeddings/mean/emb_{en,it,de,fr}.npy` (2009x768) and `embeddings_610/mean/...` (2009x1152), sliced to 2006 rows with `kept` |
| MT model | `Helsinki-NLP/opus-mt-{it,de,fr}-en`; greedy (`num_beams=1`), fp16 on CUDA, batch 64, max_length 512 |
| model | one `XGBClassifier` per backbone: `multi:softprob`, 12 classes, hist, max_bin 64, cpu, `n_jobs=2`, seed 0, depth 8, `min_child_weight=5`, lr 0.1, 400 trees; trained on EN **train+val** (1501) |
| ceiling | native model trained on the target's own **train+val** (1501), own-train `perlang_z`; same XGBoost params |
| seed | split RNG 0; XGBoost seed 0; MT greedy |

**Protocol.** Features are `perlang_z` in **English space**: L2-normalise, then z-score with the
**English-train** mean/std. The same EN-train stats are applied to (i) the English test, (ii) the
MT-translated target test, and (iii) the original target test; only the text changes, never the
feature space. The EN model is trained on EN train+val. For each target, the target **test** text is
machine-translated to English, embedded with EuroBERT mean, transformed with EN-train stats, and
scored by the same EN model. The untranslated control applies the same model and stats to the
original target embeddings. The in-language ceiling is a target-trained model (own-train z-score,
train+val) evaluated on the target test.

**Results — direct comparison on the same 505 documents (macro-F1).**

| backbone | target | EN test | MT-to-English | untranslated | in-language ceiling |
|---|---|---:|---:|---:|---:|
| 210m | it | 0.486 | **0.466** | 0.231 | 0.434 |
| 210m | de | 0.486 | **0.466** | 0.299 | 0.435 |
| 210m | fr | 0.486 | **0.476** | 0.288 | 0.467 |
| 610m | it | 0.495 | **0.519** | 0.276 | 0.428 |
| 610m | de | 0.495 | **0.494** | 0.311 | 0.466 |
| 610m | fr | 0.495 | **0.515** | 0.340 | 0.437 |

**Results — OVR-macro AUC (same 505 documents).**

| backbone | target | EN test | MT-to-English | untranslated | in-language ceiling |
|---|---|---:|---:|---:|---:|
| 210m | it | 0.904 | 0.897 | 0.799 | 0.883 |
| 210m | de | 0.904 | 0.901 | 0.775 | 0.883 |
| 210m | fr | 0.904 | 0.896 | 0.816 | 0.874 |
| 610m | it | 0.910 | 0.911 | 0.849 | 0.898 |
| 610m | de | 0.910 | 0.907 | 0.864 | 0.894 |
| 610m | fr | 0.910 | 0.914 | 0.851 | 0.876 |

**0.975 bar.** Ratio = target score / reference score, computed on the unrounded JSON values.

| backbone | metric | MT/EN | pass | MT/ceiling | pass |
|---|---|---:|---:|---:|---:|
| 210m | macro-F1 | it 0.957 / de 0.958 / fr 0.978 | **1/3** | 1.072 / 1.070 / 1.018 | **3/3** |
| 210m | OVR-macro AUC | 0.992 / 0.997 / 0.991 | **3/3** | 1.016 / 1.020 / 1.025 | **3/3** |
| 610m | macro-F1 | 1.050 / 0.998 / 1.042 | **3/3** | 1.215 / 1.061 / 1.180 | **3/3** |
| 610m | OVR-macro AUC | 1.002 / 0.997 / 1.004 | **3/3** | 1.014 / 1.014 / 1.042 | **3/3** |
| **both** | **macro-F1** | | **4/6** | | **6/6** |
| **both** | **AUC** | | **6/6** | | **6/6** |

**Verdict.** Translate-test recovers most of the English score: **4/6 macro-F1** cells and **6/6 AUC**
cells pass at 0.975 against the English model, and **6/6 macro + 6/6 AUC** pass against the
in-language reference. The two macro-F1 misses are both at 210m: it 0.957 and de 0.958, each 1.8 pp
below the bar; 210m fr (0.978) and all three 610m targets pass. MT is far above untranslated transfer
on every target (macro-F1 +0.167 to +0.244; AUC +0.043 to +0.126). It also **exceeds the in-language
ceiling** in all six macro-F1 and all six AUC cells in this pipeline.

**Caveats.**

- FLORES target columns are themselves human translations of the English source. E23 translates those
  target texts back into English, so this is a **round-trip MT-to-English** test, not a test on
  naturally occurring target text. The `en` column is the human-written English source for the same
  documents and is the EN-test reference; E23 never uses it as the target input, so MT is distinct
  from the original human translation. It is a reference, not a hard upper bound: 610m MT exceeds it
  on it/fr.
- The "ceiling" is a same-model, same-hyperparameter in-language reference on 1501 target train+val
  rows, not a tuned language ceiling. MT > ceiling means the English-source text is easier for this
  classifier than the target-side human translations; it is not evidence that MT beats
  native-language modelling in general.
- The untranslated column deliberately uses **EN-train** feature stats, so it is a controlled
  baseline, not the repo's best transfer number. With per-language target z-score, E21's XGBoost
  reports EN→IT macro-F1 0.366 (210m) / 0.461 (610m), well above the 0.231 / 0.276 here.
- Single seed; 505 test rows; rare classes are small (class 0 has 4 test rows, class 8 has 8), so
  macro-F1 differences of ~0.01 are within split noise. No confidence intervals.
- The shipped `.md` verdict uses a **0.9** bar ("3/3 EN-level"); the PNG omits the ceiling. At 0.975
  the correct counts are 4/6 macro-F1 and 6/6 AUC.
- MT / embedding caches live in `%TEMP%` and are now shared with `en_vs_enit_translate.py` (E42); a
  clean checkout must re-run MT, and a CPU-only run uses fp32 instead of fp16, which can change
  hypotheses.

---

## E24 — More datasets: XNLI, PAWS-X, textdetox, SIB-200 — DONE (numbers verified 2026-10-19; interpretation qualified)

**Script to reproduce:** `more_datasets.py` — `python more_datasets.py --dataset
xnli,pawsx,textdetox,sib200` (or `--launch` for a detached run). Reproduction status: **PARTIAL** —
all numbers reproduce the shipped JSON, but the dataset caches and HF revisions live outside the repo,
so a clean checkout cannot rebuild them without downloading the datasets.

**Question.** Does ONE English-trained XGBoost on mean-pooled EuroBERT-210m features, with the
model-free `perlang_z` transform, stay within **0.975×** English self on four additional tasks: 3-class
NLI (XNLI), binary paraphrase (PAWS-X), topic (SIB-200), and binary toxicity (textdetox)?

**Script / outputs.**

- `more_datasets.py` -> `more_datasets.json` (all numbers), `.md` (tables), `.png` (ratio bars).
- `python more_datasets.py --dataset all --launch` (detached), or
  `--dataset xnli,pawsx,textdetox,sib200` (foreground equivalent). The artifact log records a detached
  run 2026-10-12 21:16:21 -> 22:05:51; the exact argv is not stored.
- `--render` rebuilds md/png from the JSON only.
- Resumable per dataset: a dataset already present in the JSON is skipped; remove the JSON to force a
  recompute.
- `--smoke` is not a tiny check for SIB-200: `_load_sib200` ignores the smoke caps and embeds the full
  701/204 rows.

**Inputs.**

| dataset | embeddings | labels / split | provenance / notes |
|---|---|---|---|
| XNLI | `emb_xnli_{en,de,fr,es}_{train,test}_*.npy`, 20,000/5,000 × 768 | 3 classes 0=entailment, 1=neutral, 2=contradiction; official train 392,702/lang, test 5,010/lang; script caps 20,000/5,000, seed 0, same indices per language | train = translated MNLI; test = translated XNLI; labels identical across languages (parallel) |
| PAWS-X | `emb_pawsx_{en,de,fr,es}_{train,test}_*.npy`, 20,000/2,000 × 768 | binary 0/1 (1=paraphrase); train 49,401/lang, test 2,000/lang; caps 20,000/2,000 | train = machine-translated; test = human-translated; labels aligned 0.994–1.000 across languages |
| textdetox | `emb_textdetox_{en,it,de,fr,es}_{train,test}_*.npy`, 4,000/1,000 × 768 | binary 0/1 (1=toxic); 5,000/lang; stratified 80/20 per language, seed 0 | no official split; independent per-language split (no parallel-translation structure observed) |
| SIB-200 | `emb_sib200_{en,de,fr,es}_{train,test}_*.npy`, 701/204 × 768 | 7 topic classes; 701 train / 99 dev / 204 test; dev unused | Flores-200 parallel: same `index_id` and labels across languages |
| model | EuroBERT/EuroBERT-210m, mean-pooled last hidden state, `MAX_LEN=512`, fp32 | XGBoost: depth 8, 400 trees, lr 0.1, hist, `max_bin=64`, `min_child_weight=5`, `n_jobs=2`, `device=cpu`, seed 0 | one English model per dataset |

The cache revisions used by the logged run were XNLI `b8dd5d7`, PAWS-X `4cd8187`, textdetox `0190754`,
SIB-200 `38977a6`. The script does **not** pin these revisions.

**Feature configs.**

| config | operation | target data |
|---|---|---|
| `perlang_z` | L2-normalize, then z-score with each language's own train mean/std | unlabeled target train rows |
| `global_z` | L2-normalize, then z-score with English-train mean/std | none (English-only stats; **not** the same as E18's no-L2 `global_z`) |

`perlang_z` never sees target labels. For XNLI / PAWS-X / SIB-200 the "unlabeled target train" is a
translated/parallel version of the English train set (20k rows for XNLI/PAWS-X, 701 for SIB-200), not a
small sample of native text. E24's `global_z` includes the L2 step (`fit_z` at
`more_datasets.py:168-177`); it is therefore not a raw baseline, and E24 reports no `raw`/`l2` row.

**Protocol.** Fixed XGBoost (above), one English model trained on English train, then scored on English
test and every target test. `ratio = target / English self`, both under the same English model and the
same metric; comparable = every target ratio >= **0.975**. Headline: macro-F1 (argmax) for XNLI /
SIB-200; AUC for PAWS-X / textdetox. A target-trained model (same encoder/config, labelled target
train) is reported as a reference. Single seed; no thresholds tuned (binary accuracy/F1 use 0.5).

**Results (`perlang_z`; 3 dp).**

| dataset | metric | target | EN-self | EN on target | target-trained | ratio | target-trained/EN | >=0.975 |
|---|---|---|---|---|---:|---:|---:|---:|:--:|
| XNLI | macro-F1 | de | 0.440 | 0.395 | 0.415 | 0.898 | 0.943 | no |
| XNLI | macro-F1 | fr | 0.440 | 0.417 | 0.441 | 0.948 | 1.002 | no |
| XNLI | macro-F1 | es | 0.440 | 0.417 | 0.441 | 0.949 | 1.003 | no |
| PAWS-X | AUC | de | 0.602 | 0.564 | 0.584 | 0.937 | 0.970 | no |
| PAWS-X | AUC | fr | 0.602 | 0.570 | 0.587 | 0.946 | 0.974 | no |
| PAWS-X | AUC | es | 0.602 | 0.553 | 0.599 | 0.918 | 0.995 | no |
| SIB-200 | macro-F1 | de | 0.773 | 0.698 | 0.792 | 0.903 | 1.025 | no |
| SIB-200 | macro-F1 | fr | 0.773 | 0.769 | 0.771 | **0.995** | 0.998 | **yes** |
| SIB-200 | macro-F1 | es | 0.773 | 0.693 | 0.757 | 0.896 | 0.979 | no |
| textdetox | AUC | it | 0.973 | 0.604 | 0.858 | 0.621 | 0.883 | no |
| textdetox | AUC | de | 0.973 | 0.679 | 0.946 | 0.698 | 0.973 | no |
| textdetox | AUC | fr | 0.973 | 0.760 | 0.949 | 0.781 | 0.975 | no |
| textdetox | AUC | es | 0.973 | 0.785 | 0.925 | 0.807 | 0.951 | no |

\* PAWS-X is near chance: AUC 0.5, EN-self 0.602, target-trained 0.584–0.599. The ratio is not
evidence of usable paraphrase transfer. "target-trained" is **not native** for XNLI / PAWS-X / SIB-200:
those train splits are translated / parallel.

**Config contrast (min target ratio).**

| dataset | metric | `perlang_z` min | `global_z` min | `perlang_z` all >=0.975 | `global_z` all >=0.975 |
|---|---|---:|---:|:--:|:--:|
| XNLI | macro-F1 | 0.898 | 0.544 | no | no |
| PAWS-X | AUC | 0.918 | 0.903 | no | no |
| SIB-200 | macro-F1 | 0.896 | 0.572 | no | no |
| textdetox | AUC | 0.621 | 0.597 | no | no |

At 0.975, `global_z` does **not** clear PAWS-X either (min 0.903), so E24 no longer weakens E18's
"per-language stats" claim at this bar. `perlang_z` also fails all four datasets at dataset level.

**Verdict.** All 13 target cells and every number in E24 reproduce the artifact at 3 dp. At the
**0.975** bar, **1/13 cells pass: SIB-200 fr (0.995)**. **0/4 datasets pass** at dataset level. XNLI
fr/es are 0.948/0.949; PAWS-X is 0.918–0.946; SIB-200 de/es are 0.903/0.896; textdetox is 0.62–0.81.
PAWS-X no longer passes at 0.975 (it passed only at 0.90, and it is a degenerate pass because both
models are near chance: EN-self AUC 0.602). XNLI is weak in absolute terms (EN-self macro-F1 0.440,
target-trained 0.415–0.441). SIB-200 is the only dataset where the English model is healthy and one
target transfers (EN-self 0.773; SIB-200 fr 0.769; target-trained 0.771). Textdetox is the clean
failure (EN-self AUC 0.973; EN-on-target 0.604–0.785; target-trained 0.858–0.949). The datasets are
also confounded with provenance: XNLI / PAWS-X / SIB-200 are translated/parallel and textdetox has no
observed parallel structure, so task and data source are not separated.

**Caveats.**

- Single seed, no confidence intervals. At 0.975 the nearest misses are XNLI fr/es (0.948/0.949) and
  PAWS-X fr (0.946); all are well below the bar. SIB-200 es rises only to 0.901 on **accuracy**
  (headline macro-F1 0.896), still far below 0.975.
- PAWS-X is near chance (AUC 0.602 vs 0.5); XNLI is weak in absolute terms (macro-F1 0.440; accuracy
  0.441 vs 0.333). Their ratios compare weak scores, not a working cross-lingual system. SIB-200 (204
  test rows) and textdetox (1,000) are small.
- textdetox has exact train/test text leakage from the random split: duplicate test texts in train =
  en 32, it 6, de 125, fr 35, es 0 (per 1,000 test). This inflates English self (slightly) and the
  target-trained de reference (substantially); it does not directly leak into the English model's
  target test (no cross-language identical texts).
- Binary accuracy/macro-F1 are at a fixed 0.5 and are calibration-sensitive; the headline AUC is
  threshold-free. `fpr_at_recall90` uses test labels to pick the operating point (oracle metric).
- E24 has no `raw`/`l2` baseline; its `global_z` is L2 + English z, not E18's no-L2 `global_z`. The
  PAWS-X English-only pass is therefore not decomposable into L2 vs z effects.
- Cache key is `dataset/language/split/count` only (no text hash, model, or `MAX_LEN`); HF dataset
  revisions are not pinned. The cache and datasets live outside the repo, so this artifact is not
  reproducible from the repo alone.
- XNLI test has 5,010 rows but the script caps at 5,000 (10 randomly dropped); SIB-200 dev is unused;
  textdetox has no official split. The exact launch argv is not recorded.
- Coverage: 4 datasets × 3–4 targets = 13 cells; 2 configs; 1 backbone; 1 seed; 5 languages. At the
  old 0.90 bar it was 7/13 (PAWS-X 3/3 but degenerate, XNLI 2/3, SIB-200 2/3, textdetox 0/4); that
  bar is not used.

---

## E25 — Full language × language matrices — DONE (verified 2026-10-19)

**Script to reproduce:** `lang_matrix.py` — `python lang_matrix.py --launch` (full 66-row grid,
~4353 s) or `python lang_matrix.py --render`. Reproduction status: **REPRODUCIBLE** for the stored
cells; results are reused from per-cell checkpoints, so delete the cell directory and JSON for a clean
recompute. The 0.975 bar is applied to the off-diagonal / diagonal parity ratio.

**Question.** For each multilingual dataset and each EuroBERT backbone (210m, 610m), how much does
mean-pooled transfer fall when the classifier is trained on language L and tested on L′? Is the
in-language (diagonal) versus cross-lingual (off-diagonal) gap smaller at 610m, and is PAWS-X at
parity?

**Script / outputs.**

- `lang_matrix.py` -> `lang_matrix.json` (all cells and summaries), `lang_matrix.md` (tables),
  `lang_matrix_<dataset>.png` (2-panel 210m/610m matrix per dataset).
- `--launch` -> 66 rows (8 datasets × 2 backbones × languages), 2 workers, 4352.8 s, 0 failed.
- `--render` rebuilds md/png from the JSON only. `--smoke` is a tiny isolated plumbing run under
  `%TEMP%`. `--no-reuse` forbids the `emb_bf16/` source caches.
- Embeddings: `lang_matrix_emb/<dataset>/<backbone>/<lang>_{train,val,test}.npy`.
- Rows are resumable via cell checkpoints in `%TEMP%\opencode\lang_matrix_cells`. **Delete the cell
  directory and JSON to recompute**; a present cell is reused without a meta hash check.

**Inputs.**

| dataset | task / primary metric | languages | train / val / test | labels / split |
|---|---|---|---|---|
| flores_sent | 12-class topic, macro-F1 | en, it, de, fr | 997 / 504 / 505 | `flores200_en_it_de_fr.csv`; `dev`=train, `devtest` split 504/505 with seed 0; parallel rows |
| flores_agg | 11-class URL topic, macro-F1 | en, it, de, fr | 276 / 140 / 139 | URL split from the sentence CSV; `business and economy` (7 rows) dropped |
| mel | 21-label multilabel, macro-F1 | en, de, fr | 11000 / 1000 / 5000 | official `multieurlex/{lang}_{train,validation,test}.jsonl` |
| toxic | binary, macro-F1 | en, it, de, fr, es | 20000 cap / 5000 cap / full | `data/toxi_clean/*.parquet` (E20 80/10/10); actual n per language below |
| xnli | 3-class NLI, macro-F1 | en, de, fr, es | 20000 / 2490 / 5010 | HF `facebook/xnli`; non-EN train is MT of MNLI |
| pawsx | binary paraphrase, macro-F1 | en, de, fr, es | 20000 / 2000 / 2000 | HF `google-research-datasets/paws-x` |
| sib200 | 7-class topic, macro-F1 | en, de, fr, es | 701 / 99 / 204 | `Davlan/sib200` TSV (`eng_Latn`, `deu_Latn`, `fra_Latn`, `spa_Latn`) |
| textdetox | binary toxicity, macro-F1 | en, it, de, fr, es | 4000 / 500 / 500 | no official split; 80/10/10 per language, per class, seed 0 |

Toxic actual rows (train/val/test): en 20000/3000/3000, it 18472/2310/2310, de 9836/1230/1230,
fr 20000/2627/2627, es 20000/2919/2919.

Embeddings are EuroBERT masked mean over the plain tokenization (bf16 forward -> fp32 storage),
max_length 512, batch 64. Width 768 (210m) / 1152 (610m). `emb_bf16/` is reused for flores_sent,
flores_agg, mel train/test and toxic; xnli, pawsx, sib200, textdetox and mel val are embedded fresh.
Reused arrays were checked byte-identical to their source slices, and every array matches its
`meta.json` row count.

**Protocol.** One XGBoost per (dataset, backbone, train language): `hist`, `max_bin=64`, CPU,
`n_jobs=2`, `seed=0`, depth 8, `min_child_weight=5`, lr 0.1, 400 trees. Train on the train language's
**train+val** rows; evaluate on every language's held-out test rows.

Features use `perlang_z`: L2-normalize, then subtract/divide by that language's own train mean/std.
Each language self-normalizes at train and at test, so the evaluated language contributes only its own
unlabeled train texts; no target labels and no test statistics are used.

- **Diagonal** = mean macro-F1 over the N cells where train language = test language.
- **Off-diagonal** = mean macro-F1 over all N(N−1) **ordered** cross-language cells (all pairs, both
  directions; not just English→target).
- **Gap** = diag − off (signed).

Metric per dataset: **macro-F1** for all eight (multiclass argmax; binary at 0.5; mel = one binary
model per label, macro-F1 over 21 labels at 0.5). AUC is secondary. Single seed 0; no CIs.

**Results.** All values are macro-F1 means; `gap` is computed from unrounded cells, so displayed
diag/off may subtract to ±0.001. All 16 rows and all 556 matrix cells match `lang_matrix.json`/`.md`.

| dataset | backbone | in-language (diagonal) | cross-lingual (off-diag) | gap | 610 vs 210 gap |
|---|---|---:|---:|---:|---|
| flores_sent | 210m | 0.461 | 0.377 | 0.084 | — |
| flores_sent | 610m | 0.486 | 0.447 | 0.039 | smaller |
| flores_agg | 210m | 0.544 | 0.423 | 0.121 | — |
| flores_agg | 610m | 0.510 | 0.464 | 0.046 | smaller |
| mel | 210m | 0.588 | 0.330 | 0.258 | — |
| mel | 610m | 0.577 | 0.417 | 0.160 | smaller |
| toxic | 210m | 0.739 | 0.567 | 0.171 | — |
| toxic | 610m | 0.750 | 0.598 | 0.152 | smaller |
| xnli | 210m | 0.452 | 0.408 | 0.044 | — |
| xnli | 610m | 0.460 | 0.430 | 0.030 | smaller |
| **pawsx** | 210m | 0.520 | **0.532** | **−0.013** | closed |
| **pawsx** | 610m | 0.561 | 0.558 | 0.003 | closed |
| sib200 | 210m | 0.792 | 0.691 | 0.101 | — |
| sib200 | 610m | 0.800 | 0.763 | 0.037 | smaller |
| textdetox | 210m | 0.870 | 0.672 | 0.198 | — |
| textdetox | 610m | 0.874 | 0.717 | 0.158 | smaller |

Gap sign check (4 dp; |gap| is the absolute value):

| dataset | gap 210m | gap 610m | signed change | \|gap\| 210m -> 610m |
|---|---:|---:|---:|---:|
| flores_sent | +0.0843 | +0.0389 | −0.0454 | 0.0843 -> 0.0389 |
| flores_agg | +0.1209 | +0.0458 | −0.0751 | 0.1209 -> 0.0458 |
| mel | +0.2575 | +0.1598 | −0.0977 | 0.2575 -> 0.1598 |
| toxic | +0.1714 | +0.1520 | −0.0194 | 0.1714 -> 0.1520 |
| xnli | +0.0440 | +0.0299 | −0.0141 | 0.0440 -> 0.0299 |
| **pawsx** | **−0.0125** | **+0.0032** | **+0.0157** | **0.0125 -> 0.0032** |
| sib200 | +0.1013 | +0.0371 | −0.0642 | 0.1013 -> 0.0371 |
| textdetox | +0.1983 | +0.1578 | −0.0405 | 0.1983 -> 0.1578 |

Overall means over datasets: 210m diag 0.621 / off 0.500 / gap 0.121; 610m 0.627 / 0.549 / 0.078.

**0.975 parity check.** Parity = off-diagonal / diagonal macro-F1 >= 0.975. Only **PAWS-X** passes:
**1.024** (210m) and **0.994** (610m) — 2/16 dataset × backbone rows. The other 14 ratios are
0.562–0.954 (mel 210m 0.562; flores_sent 610m 0.920; xnli 610m 0.935; sib200 610m 0.954). PAWS-X
passes at near-chance absolute scores (EN-self AUC 0.530/0.628), so this is not evidence of a working
cross-lingual system.

**Verdict.**

- 610m reduces the **absolute** gap on all 8 datasets. The **signed** gap shrinks on 7 of 8. PAWS-X
  moves from −0.0125 to +0.0032: |gap| falls 4×, but the signed gap **rises** by 0.0157 and crosses
  zero. "610m shrinks the gap on every dataset" is therefore **true only for |gap|**; it is **false
  for the signed gap shown in the table**.
- On flores_agg and mel the 610m diagonal is lower (0.544 -> 0.510; 0.588 -> 0.577). Part of those gap
  reductions is lower in-language quality, not better transfer. On flores_agg the drop is
  en 0.564 -> 0.466 and de 0.590 -> 0.476 while fr rises 0.464 -> 0.559 (139 test rows; noisy).
- PAWS-X "parity" is numerically true but weak: EN self AUC is 0.530 (210m) / 0.628 (610m); macro-F1
  is 0.510 / 0.575, barely above the balanced-binary baseline. E24 reports EN-self AUC 0.602 with
  train-only; E25's train+val protocol lowers it.
- Largest all-pairs gaps: mel (0.258 / 0.160) and textdetox (0.198 / 0.158). The **English-centric**
  largest gap is toxic (macro-F1 0.342 / 0.341; AUC 0.231 / 0.209), not mel/textdetox.
- The off-diagonal mean is **all pairs**, not English→target. Directionality is material: toxic 210m
  en→others macro-F1 0.422 vs others→en 0.642; textdetox 0.637 vs 0.738. The all-pairs average
  therefore overstates English→target transfer on the toxicity datasets.

**Caveats.**

- **textdetox duplicate leakage.** The 80/10/10 split is per row, and textdetox repeats texts within a
  language. Test rows whose exact text also occurs in that language's train split: en 3.4%, it 1.0%,
  de 12.4%, fr 6.0%, es 0.0%. On a train-only reproduction, macro-F1 on the non-duplicated test subset
  drops by 0.017 (de), 0.008 (fr), 0.003 (en), 0.002 (it), 0.000 (es). The textdetox diagonal (and
  therefore its gap) is inflated.
- **PAWS-X train+val sensitivity.** Adding the 2,000 validation rows to the 20k train rows lowers EN
  test AUC from 0.581 to 0.528 (250-tree reproduction); E25's 400-tree 0.530 vs E24's 0.602 is the
  same effect. PAWS-X validation shares 1,339 sentences with test (train shares 0), so this is a
  dataset-construction effect, not a leak. The PAWS-X row is not comparable to E24.
- **FLORES test size.** For flores_sent/agg, test is half of the `devtest` split (505/139), not the
  full standard split. The `lang_matrix.md` sentence "test is the full standard split" is false for
  these two datasets.
- Single seed, no CIs. Test n is as low as 139 (flores_agg), 204 (sib200), 500 (textdetox), 505
  (flores_sent). The 0.003 PAWS-X gap and the 0.014 xnli gap **change** are within noise.
- Most test sets are parallel translations of shared items (FLORES, MEL, XNLI, SIB200; PAWS-X labels
  are 99.4–100% aligned), so the matrix measures translation invariance on the same items, not
  generalization to new target-language items. textdetox is **not** verified as cross-language
  aligned: full label sequences differ for it/de/es and exact cross-language text overlap is 0.
- Metric mismatch across entries: E18 mel is micro-F1 with quantile-matched thresholds; E25 mel is
  macro-F1 at 0.5. E24 PAWS-X is AUC with train-only. Do not compare those numbers directly.
- No per-cell predictions are stored; CIs and alternative thresholds require a re-run.

---

## E25b — All-languages vs per-language training — DONE (verified 2026-10-19)

**Script to reproduce:** `alllang_vs_perlang.py` — `python alllang_vs_perlang.py` (skips pairs already
in the JSON; delete the JSON to recompute) or four shards + `--merge` as in the stored run.
Reproduction status: **REPRODUCIBLE** (all table cells re-checked against the JSON; md/png re-render
byte-identically).

**Question.** Does ONE XGBoost trained on the pooled train rows of every language beat one XGBoost per
language, on FLORES topic (12-class `sent`, 11-class `agg`) and on binary toxicity?

**Script / outputs.**

- `alllang_vs_perlang.py` -> `alllang_vs_perlang.json` (numbers), `.md` (per-language tables), `.png`
  (3 panels: accuracy / macro-F1 / AUC).
- `--render` rebuilds md/png from the JSON only; `--launch` runs detached hidden; `--smoke` is a
  plumbing check; `--tag NAME` writes a shard under `%TEMP%\opencode\alllang_vs_perlang\` and `--merge`
  folds the shards into the repo files.
- The stored run used four shards (`flores_210m`, `flores_610m`, `toxic_210m`, `toxic_610m`) plus
  `--merge` (`meta.merged_from`), 2026-10-13.
- Embeddings are read only from `emb_bf16/`.

**Inputs.**

| dataset | embeddings | labels / split | languages |
|---|---|---|---|
| FLORES sent | `emb_bf16/flores_sent/{210m,610m}/mean/emb_{lg}.npy`, 2006 x 768/1152 | `flores200_en_it_de_fr.csv`; 12 classes; 997 / 504 / 505 | en/it/de/fr; **parallel rows**, same label per row |
| FLORES agg | `emb_bf16/flores_agg/{210m,610m}/mean/emb_{lg}.npy`, 562 x 768/1152 | `flores_agg.csv`; 11 classes (`business and economy` dropped); 276 / 140 / 139 | en/it/de/fr; **parallel rows** |
| toxicity | `emb_bf16/toxic/{210m,610m}/emb_{lg}_{train,validation,test}.npy` | `data/toxi_clean/*.parquet`; binary `is_toxic`; train 24,000/18,472/9,836/21,011/23,349; test 3,000/2,310/1,230/2,627/2,919 (val = test sizes) | en/it/de/fr/es; deduped per language, 0 within-language cross-split overlap |

FLORES rows are translations of the same document, so the pooled FLORES model sees the same 997 label
vectors four times, not four times as many documents. Toxicity rows are independent texts within a
language.

**Features.** `perlang_z` on every row of both regimes: L2-normalize, then z-score with that row's own
language's **TRAIN** mean/std. The transform is model-free and never uses test rows or labels. It is
applied before pooling, so the all-langs matrix is a stack of per-language normalized rows. Embeddings
are masked-mean EuroBERT (bf16 forward, fp32 storage).

**Protocol.** Fixed XGBoost: `hist`, `max_bin=64`, `device=cpu`, `n_jobs=2`, `seed=0`, depth 8,
`min_child_weight=5`, lr 0.1, 400 trees. `multi:softprob` with 12/11 classes for FLORES;
`binary:logistic` for toxicity. No Optuna, no early stopping, no validation calibration.

- `all-langs`: ONE model on the pooled TRAIN rows of every language (3,988 / 1,104 / 96,668).
- `per-lang`: one model per language on its own TRAIN rows.
- Both are scored on every language's held-out TEST split.

Headline is macro-F1. Toxicity uses a fixed 0.5 threshold. `delta = all-langs - per-lang`; the verdict
deadzone is ±0.002. Macro averages are the plain mean over the dataset's languages.

**Results.**

| dataset | backbone | metric | per-lang | all-langs | delta (all-per) | verdict |
|---|---|---:|---:|---:|---:|---|
| FLORES sent | 210m | macro-F1 | 0.388 | 0.400 | +0.012 | all-langs better (3/4) |
| FLORES sent | 610m | macro-F1 | 0.379 | 0.414 | +0.035 | all-langs better (3/4) |
| FLORES agg | 210m | macro-F1 | 0.506 | 0.543 | +0.037 | all-langs better (3/4) |
| FLORES agg | 610m | macro-F1 | 0.446 | 0.548 | **+0.101** | all-langs better (4/4) |
| toxicity | 210m | macro-F1 | 0.746 | 0.730 | -0.016 | per-lang better (4/5; fr ties at 3 dp) |
| toxicity | 610m | macro-F1 | 0.752 | 0.738 | -0.014 | per-lang better (5/5) |

Italian-only check (IT test rows; `delta = all-langs - per-lang`):

| dataset | backbone | IT per-lang | IT all-langs | delta macro-F1 | IT AUC delta |
|---|---|---:|---:|---:|---:|
| FLORES sent | 210m | 0.378 | 0.354 | -0.023 | +0.016 |
| FLORES sent | 610m | 0.317 | 0.377 | **+0.060** | +0.016 |
| FLORES agg | 210m | 0.540 | 0.498 | -0.042 | +0.016 |
| FLORES agg | 610m | 0.482 | 0.516 | **+0.034** | **+0.032** |
| toxicity | 210m | 0.709 | 0.695 | -0.014 | -0.016 |
| toxicity | 610m | 0.722 | 0.711 | -0.011 | -0.006 |

The AUC delta column is the **Italian-only** OVR-macro AUC delta, not the macro over languages.

**Verdict.** At the **0.975** bar, all-langs/per-lang is >= 0.975 in **6/6** cells: FLORES 1.032/1.093
(sent) and 1.073/1.227 (agg); toxicity 0.979/0.981 (just above the bar). On the macro average, pooling
helps FLORES in all four dataset × backbone cells and hurts toxicity on both backbones. The FLORES gain
is bigger at 610m (+0.035 sent, +0.101 agg) than at 210m (+0.012, +0.037). All-langs wins 3/4, 3/4,
3/4, 4/4 of the per-language comparisons on FLORES and 1/5, 0/5 on toxicity. For Italian, **macro-F1**
favors all-langs only on FLORES at 610m (+0.060 sent, +0.034 agg); at 210m Italian macro-F1 drops
(-0.023, -0.042) even though Italian AUC rises (+0.016). AUC improves on all four FLORES cells and
drops on toxicity.

**Caveats.**

- Single seed (0), no CIs. Test n is 505 (sent), 139 (agg), and 3,000/2,310/1,230/2,627/2,919
  (toxicity). The agg cells and the +0.101 / +0.060 Italian gains are the noisiest.
- All-langs gets more rows than per-lang: 4× on FLORES and 4×–10× on toxicity. There is no
  matched-row or equal-capacity control, and FLORES "more rows" means more translations of the same
  documents, so "language sharing" is confounded with translation augmentation. The toxicity loss is
  the stronger result because it survives 4×–10× more pooled data.
- Toxicity uses a fixed 0.5 threshold with no validation calibration. Per-language positive rates run
  from 11.7% (en) to 36.5% (fr), while the pooled model sees a mixed prior. The macro-F1 gap is
  partly a prior/threshold effect; AUC also drops (by -0.0004 to -0.017), so a small real ranking loss
  remains.
- Pooled toxicity training is language-imbalanced (en 24,000 vs de 9,836 rows); small-language
  dilution is a plausible part of the de/it losses. No per-language sample weights were used.
- The Italian result is metric-dependent at 210m (macro-F1 down, AUC up).
- Dedup in `data/toxi_clean` is per language, not cross-language: one text occurs in es train and fr
  test (1 of 96,668 pooled train rows), so pooled leakage is not exactly zero. Negligible.
- The JSON is resumable; `meta.finished` is the first shard's finish (07:06:52), not the merge time
  (07:14:19). Every table cell was re-checked against the JSON; md/png re-render byte-identically.

---

## E20 — Toxicity, clean split — DONE (verified 2026-10-19)

**Script to reproduce:** `toxi_proof_clean.py` — `python toxi_proof_clean.py --launch` (data rebuild:
`python clean_toxi.py --langs en,it,de,fr,es --out data/toxi_clean --seed 0 --total-cap 30000`).
Reproduction status: **REPRODUCIBLE** (all 15 saved models reproduce the stored per-seed AUCs).

**Question.** Can ONE English-trained XGBoost, fed only mean-pooled EuroBERT-210m features, reach
>= **0.975** × the **native** (in-language) AUC on binary toxicity after leakage correction?

**Script / outputs.**

- `toxi_proof_clean.py` -> `toxi_proof_clean.json` (numbers), `.md` (full table), `.png`.
- Data builder: `clean_toxi.py` -> `data/toxi_clean/<lang>_{train,validation,test}.parquet` +
  `leakage_report.md`.
- Models: `models_toxi/{en,it,de,fr,es}_seed{0,1,2}.ubj` + `models_toxi/manifest.json` (15 models).
- `--smoke` writes TEMP-only scratch outputs; `--launch` runs detached.
- The script retrains all 15 models on every run. Embeddings are cached in
  `%TEMP%\opencode\toxi_clean_cache\emb_<lang>_<split>.npy`; delete that cache to recompute them.

**Inputs.** `toxi-text-3M` (FredZhang7), pooled over all shipped splits and rebuilt by `clean_toxi.py`:

| lang | train | validation | test | test pos rate |
|---|---:|---:|---:|---:|
| en | 24,000 | 3,000 | 3,000 | 0.121 |
| it | 18,472 | 2,310 | 2,310 | 0.294 |
| de | 9,836 | 1,230 | 1,230 | 0.310 |
| fr | 21,011 | 2,627 | 2,627 | 0.369 |
| es | 23,349 | 2,919 | 2,919 | 0.313 |

Total 120,840 rows. Build: pool train+validation+test, deduplicate **exact `text`** per language, cap
the per-language total at 30,000 **before** splitting, then re-split 80/10/10 with seed 0 (ceil on the
10% parts). Only EN reaches the cap: 2,534,095 deduplicated rows -> 30,000-row random sample (1.2%).
The other pools are 12,296–29,187 rows. The build asserts **0 exact-text cross-split overlap**; the
check was repeated on the shipped parquet (0 overlap, 0 exact duplicates inside each split).

**Protocol.**

- EuroBERT-210m, **masked mean** over the last hidden state, fp32, max length 512, batch 64, GPU.
- `perlang_z`: L2-normalize, then z-score with each language's **own train** mean/std. The English
  model also uses the target's train statistics at test time; these are unlabeled target rows.
- XGBoost: `binary:logistic`, `hist`, `max_bin=64`, depth 8, `min_child_weight=5`, lr 0.1, 400 trees,
  `n_jobs=2`, CPU.
- **EN model**: train on EN train, test on each target test. **Native model**: train on the target's
  own train, test on the target's own test. "Native" means in-language, not a second architecture.
- Thresholds: F1-optimal on the model's own-language validation split. AUC is threshold-free.
- **Seeds**: the XGBoost config has no row/column subsampling and is seed-deterministic. To obtain
  variability, each of seeds 0/1/2 trains on a bootstrap resample (size = n_train, with replacement)
  of its train rows. Validation and test rows are fixed. All 15 models are saved.
- Bars: target/EN AUC >= **0.975** and EN/native AUC >= **0.975**.

**Results.** Mean ± sd over the three bootstrap seeds.

| target | EN-trained AUC | native AUC | EN/native | target/EN |
|---|---:|---:|---:|---:|
| it | 0.649 ± 0.011 | 0.834 ± 0.002 | 0.778 ± 0.015 | 0.720 ± 0.010 |
| de | 0.678 ± 0.013 | 0.869 ± 0.002 | 0.780 ± 0.016 | 0.752 ± 0.016 |
| fr | 0.776 ± 0.008 | 0.866 ± 0.000 | **0.896 ± 0.010** | 0.861 ± 0.013 |
| es | 0.639 ± 0.005 | 0.797 ± 0.007 | 0.802 ± 0.009 | 0.709 ± 0.009 |

EN self reference: AUC **0.901 ± 0.004**, AP 0.689 ± 0.007. Mean AP: EN 0.508 vs native 0.715.
Mean FPR@R90 (lower is better): EN 0.716 vs native 0.427.

**Verdict.** **Not comparable on toxicity at 0.975: 0/4** targets on both bars. The English model
loses 0.09–0.19 AUC against the in-language model. The closest target is fr: EN/native **0.8959** and
target/EN **0.861** — both far below 0.975. The verdict survives removal of residual near-duplicate
test rows (max AUC shift 0.013).

**Caveats.**

- **Ratio denominator.** The E20 table in `EXPERIMENTS.md` uses EN/native. E18 and the master result
  use target/EN. Both are 0/4, so the verdict does not change, but the two ratios differ.
- **Rounding trap.** fr EN/native is 0.8959, shown as 0.90 at two decimals. It is **below** the 0.975
  bar; do not read the 2-dp table as a pass.
- **"3 seeds" means bootstrap resamples**, not three independent data splits or initializations. The
  sd reflects bootstrap/training-sample noise, not split uncertainty.
- **`perlang_z` uses unlabeled target train rows.** The English model is not strictly target-free.
- **Exact-text dedup only.** After lowercase/punctuation/whitespace normalization, 39/70/91/37 test
  rows (it/de/fr/es) still match a train row. These are 1.3–5.7% of test; AUC changes by at most
  0.013, and the verdict is unchanged.
- **Train sizes are not matched.** EN has 24,000 rows; de native has 9,836, it 18,472, fr 21,011,
  es 23,349. EN has the largest training set, so its observed deficit is conservative.
- **Threshold view.** The EN threshold is tuned on EN val and applied to target test. Target F1@val-thr
  is therefore low (0.119–0.224). AUC and AP are the fair cross-lingual metrics.
- 15/15 saved models reproduce the stored per-seed AUCs exactly (<= 1e-6).

---

## E42 — Raw-feature EN vs EN+IT, IT control, translate-train and translate-test, FLORES sentence — DONE (verified 2026-10-20)

**Script to reproduce:** `en_vs_enit_raw.py` — `--smoke`, `--launch` (A/B), `--add-c` (IT control),
`--add-d` (B-on-EN + MT translate-train). Clean translate-train: `en_vs_enit_dclean.py --launch`.
Raw translate-test: `en_vs_enit_ttest.py --launch`. Reproduction status: **REPRODUCIBLE** (A/B/C/D are
preserved bit-exactly through the merges; E/F are separate sidecars).

**Question.** With **only raw mean-pooled EuroBERT-210m features** — no L2, no z-score, no
per-language z, no PCA/INLP — how much of an in-language Italian model's test AUC does an English-only
model retain, does adding language data cost English, and which is better: translating the **training**
text (translate-train) or translating the **test** text (translate-test)?

**Script / outputs.**

- `en_vs_enit_raw.py` -> `en_vs_enit_raw.json` (all numbers), `.md` (tables), `.png` (7-cell metrics +
  ratio panel at the 0.975 bar).
- Backups: `en_vs_enit_raw.ab_premerge.json` (A/B before C), `en_vs_enit_raw.pre_d.json` (A/B/C
  before D).
- `--smoke` writes `en_vs_enit_raw_smoke.*` (2 trials/arm) and never touches the canonical files.
- Provenance: `%TEMP%\opencode\en_vs_enit_raw\` (`en_vs_enit_raw.log`, `.pid`, `_status.json`,
  `run.cmd`, `launch.log`, `mt_cache_verify.json`). Runs: A/B PID 7368 (586 s), C PID 21812 (94 s),
  D PID 11392 (394 s).

**Inputs.**

| item | value |
|---|---|
| labels / split | `flores200_en_it_de_fr.csv` via `train_flores12.prepare_sentence()`; 12 classes; **997 / 504 / 505** (seed 0) |
| embeddings | `train_flores12.load_sent_embeddings("mean", kept, 2009)`; `embeddings/mean/emb_{en,it}.npy`, 2009x768 float32, sliced to the 2006 kept rows |
| MT (arm D) | `Helsinki-NLP/opus-mt-en-it`; the repo's `flores_close` cache `%TEMP%\opencode\flores_close_cache\tt_emb_sent_it.npy` (1501x768), **verified** against a fresh fp16 GPU translate |
| dataset | FLORES **sentence only** — `flores_agg` is never loaded |
| model | `xgboost.XGBClassifier`, `multi:softprob`, `num_class=12`, `hist`, `max_bin=64`, `device=cpu`, `n_jobs=2`, seed 0 |

FLORES devtest rows are parallel across languages, so every arm shares one label vector; arms differ
only in which language's train rows (and which labels) they see. **Test text is never translated**:
D predicts on the raw Italian and raw English test embeddings.

**Arms.**

| arm | train rows | tuned on (objective = OVR-macro AUC) | test |
|---|---:|---|---|
| A | 997 (EN) | EN val | IT test (primary), EN test (self reference) |
| B | 1994 (EN+IT) | IT val | IT test, EN test |
| C | 997 (IT) | IT val | IT test (in-language control) |
| D | 3002 (EN + MT-IT, train+val) | IT val⁺ | real IT test, EN test |
| E | 1994 (997 EN + 997 MT-IT train) | IT val (unseen) | real IT test, EN test |
| F | 997 (EN) | EN val | **MT-IT test** (translate-test), EN test, real IT test |

⁺ D's hyperparameters are selected on real IT val using a **1994-row train-only** fit (so no val row
enters the tuning signal), then refit on 3002 rows. The 1994-row refit is reported as D′. E is the
clean translate-train arm: 997 EN + 997 MT-IT, fresh tuning on the unseen 504-row IT val.

Six independent Optuna TPE studies, seed 0, **10 trials each**. Search: max_depth 3–10,
learning_rate 0.03–0.3 (log), n_estimators 100–800, min_child_weight 1–10, subsample 0.6–1.0,
colsample_bytree 0.4–1.0, reg_lambda 0–5.

**Results (test).**

| cell | accuracy | micro-F1 | macro-F1 | OVR-macro AUC |
|---|---:|---:|---:|---:|
| A: EN train -> EN test (self ref) | 0.5446 | 0.5446 | 0.4149 | **0.8739** |
| A: EN train -> IT test (primary) | 0.3901 | 0.3901 | 0.1799 | **0.7880** |
| B: EN+IT train -> IT test | 0.5168 | 0.5168 | 0.3856 | **0.8370** |
| B: EN+IT train -> EN test | 0.5564 | 0.5564 | 0.4399 | **0.8729** |
| C: IT train -> IT test (control) | 0.5109 | 0.5109 | 0.3741 | **0.8423** |
| D: EN + MT-IT train -> IT test | 0.5644 | 0.5644 | 0.4096 | **0.8860** |
| D: EN + MT-IT train -> EN test | 0.6238 | 0.6238 | 0.5409 | **0.9161** |
| E: EN + MT-IT train (1994) -> IT test | 0.5109 | 0.5109 | 0.3547 | **0.8418** |
| E: EN + MT-IT train (1994) -> EN test | 0.5406 | 0.5406 | 0.4243 | **0.8827** |
| F: EN train -> MT-IT test (translate-test) | 0.5386 | 0.5386 | 0.4272 | **0.8609** |

**1. English side of B — does adding Italian hurt English?** No. B_EN/A_EN = 0.8729 / 0.8739 =
**0.9989**, above the 0.975 bar (AUC cost 0.10 pp; macro-F1 actually rises 0.4149 -> 0.4399). Adding
Italian labels to an English model leaves English essentially intact.

**2. Translate-train with MT EN→IT (arm D).** 1501 EN train+val rows translated (greedy, fp16, batch
64, max_length 512), embedded with EuroBERT-210m mean, trained on EN + MT-IT (3002 rows).

| question | ratio | value | >= 0.975 | reading |
|---|---|---:|---:|---|
| MT-IT train vs real-IT train | D(IT) / B(IT) | **1.0586** | yes | +5.9%, but D trains on 3002 vs 1994 rows |
| MT-IT train vs Italian ceiling | D(IT) / C(IT) | **1.0519** | yes | +5.2%, with 3× C's rows |
| English retention of translate-train | D(EN) / A(EN) | **1.0483** | yes | +4.8%, again 3× A's rows |

The three ratios clear 0.975 but are **confounded with training-set size**, so they overstate the value
of MT. At matched 1994 rows (D′, same tuned params): IT AUC **0.8418**, EN AUC **0.8827**,
**D′(IT)/B = 1.0057**, **D′(IT)/C = 0.9994**, D′(EN)/A_EN = 1.0101. That is the fair comparison:
**MT-Italian substitutes for real Italian almost exactly at equal rows**, and the apparent D win is
mostly the extra 1008 rows.

**2b. Clean translate-train (E) and raw translate-test (F).** E retunes on the unseen 504-row IT val
(no val row in training). F is A's model evaluated on MT-Italian-to-English test embeddings; F and A
share one model, so F/A is a pure test-representation swap.

| arm | strategy | test input | train rows | MT cost | IT AUC | IT macro-F1 |
|---|---|---|---:|---|---:|---:|
| A | none | real Italian | 997 EN | none | 0.7880 | 0.1799 |
| E | translate-train | real Italian | 1994 (EN + MT-IT) | 997 rows, once | 0.8418 | 0.3547 |
| F | **translate-test** | MT Italian -> English | 997 EN | 505 items, every inference | **0.8609** | **0.4272** |
| C | IT-only reference | real Italian | 997 IT | none | 0.8423 | 0.3741 |

| comparison | ratio | >= 0.975 | reading |
|---|---:|---:|---|
| F / E | **1.0227** | yes | translate-test leads by 1.9 pp AUC (1-2 SE) |
| F / A | **1.0926** | yes | translating the test recovers most of the gap |
| E / A | **1.0683** | yes | translate-train helps without inference MT |
| E / B | **1.0057** | yes | MT-Italian equals real-Italian training at matched rows |
| E / C | **0.9994** | yes | parity with the native Italian model |
| F macro-F1 / E macro-F1 | **1.2042** | yes | translate-test wins decisions by 7.3 pp |

E is numerically identical to D′ (the clean retune selected the same params), so the 1994-row result
is not a tuning artifact. F's AUC edge is narrow; its macro-F1 edge is the decisive difference. Cost
is the reverse: translate-test pays MT on every test item, translate-train pays it once on 997 training
rows and serves with no MT in the loop.

**3. Existing arms (unchanged).** A(IT)/B(IT) = **0.9414** -> FAIL at 0.975. C/A(EN) = 0.9639 (Italian
harder). A/C = 0.9355 (FAIL), B/C = 0.9937 (no resolved effect). Train+val refit: A/B = 0.9358,
A/C = 0.9242, B/C = 0.9876 — same verdicts.

**Verdict.** Raw features only, the EN-only model reaches **0.7880** AUC on Italian against **0.8423**
for an Italian-trained model (A/C = **0.9355**, FAIL). Italian is intrinsically harder than English
(C/A = 0.9639), and adding English to Italian gives parity, not a gain (B/C = 0.9937). Adding Italian
to English costs nothing in English (B_EN/A_EN = 0.9989).

Both translation strategies reach the native level, by different routes:

- **Translate-train (E)** is at parity with native Italian (E/C = **0.9994**) and with real-Italian
  training (E/B = **1.0057**) at equal rows, and it needs **no MT at inference**.
- **Translate-test (F)** is the quality leader — AUC **0.8609** (F/E = **1.0227**) and macro-F1
  **0.4272** (F/E = **1.2042**) — but it pays **MT on every test item**.
- D's 3002-row result (0.8860, D/C = 1.0519) is **not** a fair comparison: it trains on 1.5× B / 3× A
  rows. Do not quote D/C = 1.0519 as “MT beats real Italian”. The clean 1994-row translate-train
  number is **E = 0.8418**.

**Practical choice:** if decision quality matters and inference-time MT is acceptable, use
**translate-test**. If you need a fixed model with no MT in the serving loop, use **translate-train**;
it matches native Italian without target labels.

**Caveats.**

- Single seed (0); 505-row test. Differences under ~0.02 AUC are not resolved. Every ratio inherits
  that noise.
- All arms share the same 12-class label vector (parallel FLORES rows).
- 10 Optuna trials per arm: enough to beat defaults, not to exhaust the space.
- Main A/B/C numbers refit best params on train rows only; D reports the 3002-row fit (as specified)
  plus the 1994-row D′ variant.
- Arm C is a control, not a transfer cell; C/A measures corpus difficulty, not transfer.
- D trains on synthetic Italian; MT output is fluent but not human, so D is an upper bound on what
  translated training data can buy.
- D's val AUC is **in-sample** (val rows are in its 3002 training rows) and is stored as
  `it_val_in_sample`, not as a held-out metric. D's tuning uses a 1994-row fit to keep val out of the
  tuning signal; the 3002-row fit is the reported model, and its test AUC is inflated by row count,
  not by test leakage.
- E is numerically identical to D′ (same selected params) but was retuned on the unseen 504-row val;
  it is the clean translate-train number.
- F uses MT on the test text by definition; F/A is a pure test-representation swap because F and A
  share one model. F's AUC edge is 1.9 pp (~1-2 SE); its macro-F1 edge is decisive.
- B_EN uses B's stored best params (no re-tuning on English).

---

# PART B — Why mean pooling, and what the dimensions do

The mechanism half of the programme. If mean-pooled transfer works, the language signal should be
redundant (E1/E1b), the classifier should not preferentially use it (E17, E41), and removing leading
directions should either help a little (topic/multilabel: E12/E13/E26, E28) or not at all (toxicity).

## E1/E1b — Language-dimension census + INLP — DONE (partially verified 2026-10-19)

**Script to reproduce:** NOT AVAILABLE — `langdims_probe.py` and `lang_inlp.py` never existed in git
and no complete artifact survives. **(UNK)**

Both paths are absent from the working tree and from every commit on every branch. `git log --all --
<path>` returns nothing, and a search of every tree reachable from `git rev-list --all` finds no such
path. Commit `21ab888` (2026-10-12) added the entry's table and citations, but not the scripts or their
`lang_inlp.*` outputs.

**Question.** Is language identity carried by a small set of embedding dimensions, or is it redundant
across nearly all dimensions? Does a small INLP subspace remove it?

**Script / outputs.**

- Missing: `langdims_probe.py`, `lang_inlp.py`.
- Missing outputs: `lang_inlp.npz`, `lang_inlp_mean.npz`, `lang_inlp_agg.npz`,
  `lang_inlp_agg_610.npz`; no `lang_inlp.json` or `lang_inlp.md` exists.
- Surviving equivalent: `lang_shap_dims.py` -> `lang_shap_dims.json` / `.md`. It runs the same 4-way
  probe on the same FLORES mean embeddings, with the same η² dimension ranking.
- Surviving primitives: `lang_neutralize.py` (`eta2_per_dim`, `top_dims`, `zero_dims`, `project_out`).
- Surviving INLP artifact: `lang_inlp_mel.npz`, one `basis` array (2,768) float64. It is untracked and
  ignored by `*.npz` in `.gitignore`. It is a MultiEURLEX basis, not the FLORES mean basis.
- Surviving INLP code: `backbone_grid.py` (`_fit_inlp`, 30-direction cap), `mel_close.py` (`fit_inlp`,
  30-direction cap), `agg_pr_2models.py` (accuracy stop at 0.30), `dim_removal.py` (k=5).
- `dim_removal.py:404-408` repeats the E1 values in its prose. That is a restatement, not new evidence.

**Inputs.**

| item | value |
|---|---|
| embeddings | `embeddings/mean/emb_{en,it,de,fr}.npy`, 2009x768 raw, sliced to the 2006 kept FLORES rows (rows 997-999 deleted), EuroBERT-210m mean pooling |
| languages | en, it, de, fr (4 classes; nominal chance 0.250) |
| η² | one-way ANOVA per dimension over the 4 languages |
| probe | `StandardScaler` + multinomial `LogisticRegression(C=1.0, max_iter=300)` |
| split | stratified 70/30, 2006 kept rows per language; seed 5 reproduces the table triple |
| INLP | iterated null-space projection of a language-identity logistic probe; 30-direction cap |

**Claims.**

| # | claim (E1/E1b) | status | evidence |
|---|---|---|---|
| 1 | 4-language probe, all 768 dims = 1.000 | VERIFIED | `lang_shap_dims.json` sent `probe.full = 1.0`; re-run 1.00000 (seed 5) |
| 2 | zero top-557 η² dims (90% mass) -> 0.999 | VERIFIED | k=557 reproduced exactly; stratified probe gives 0.99917 -> 0.999 (seed 5) |
| 3 | zero all 768 dims (chance) -> 0.250 | VERIFIED | constant predictor on the same split gives exactly 0.25000 (602 test rows per class) |
| 4 | INLP ~30 directions to chance (mean pooling) | UNVERIFIED | no FLORES language basis or accuracy curve survives |

**Partial reproduction.** Claims 1–3 reproduce with surviving code, but not with the cited script. The
run used `lang_neutralize.eta2_per_dim` + `top_dims`, then a stratified 70/30 `train_test_split` with
the probe above, on the 2006 kept rows per language. In a seed sweep 0–5, full accuracy is 0.999–1.000
and top-557 accuracy is 0.999–1.000; seed 5 reproduces the table's 1.000 / 0.999 / 0.250 exactly at
3 dp. Claim 4 is not reproducible: no FLORES language-subspace basis, no per-round probe accuracy, and
no stopping log survive. Do not run a new INLP study to fill the gap.

**Results (reproduced 2026-10-19).**

| check | value |
|---|---:|
| k covering 90% of η² mass | 557 |
| total η² (2006 kept rows) | 255.8357 |
| probe, all 768 dims (seed 5) | 1.000 |
| probe, top-557 zeroed (seed 5) | 0.999 (0.999–1.000 over seeds 0–5) |
| probe, all 768 zeroed | 0.250 |
| stored sent probe: full / -237 / -625 | 1.000 / 1.000 / 0.9975 |
| surviving mel INLP basis | 2 directions |
| mel probe after those 2 directions | 0.546 (chance 0.333) |

**Verdict.** PARTIAL. The central finding survives: language is redundant across nearly all
dimensions. Zeroing the 557 dimensions that carry 90% of the η² mass leaves the probe at 0.999. No
small axis-aligned dimension set carries the language. The INLP half of the entry is not verified. The
only surviving measured chance run uses 3 directions (`agg_pr2.md`, FLORES-agg 610m); the 30 in
`backbone_grid.json` is a configuration cap, not a measured stopping point.

**0.975 bar: N/A.** This experiment has no English reference ratio. Its metrics are 4-way probe
accuracy and INLP direction count.

**Caveats.**

- The cited scripts were never committed; this entry cannot be reproduced from repository history.
- The third decimal depends on the split. A capped, non-stratified probe (`lang_shap_dims.py`, cap
  2000, seed 0) gives 1.000 after zeroing the top 557 dims and 0.241 after zeroing all dims. The
  stratified split gives the table's 0.999 and 0.250. The difference is a few test rows.
- `lang_inlp_mel.npz` is the only INLP artifact. It is untracked and ignored, so it can disappear
  without a git trace. Its 2 directions do not neutralize language on a held-out 3-way probe
  (0.546 vs 0.333).
- INLP code in the repository uses different stop rules and caps: `backbone_grid.py` caps at 30 with no
  accuracy test, `agg_pr_2models.py` stops at 0.30, `mel_close.py` caps at 30 but cached 2 directions,
  and `dim_removal.py` uses 5. Direction counts are not comparable across these runs.
- Single seed, no confidence intervals. The 0.999 vs 1.000 boundary is a few test rows.

---

## E17 — Language dims vs their SHAP contribution — DONE (verified 2026-10-19)

**Script to reproduce:** `lang_shap_dims.py` — `python lang_shap_dims.py` (all three, ~411 s) or
`python lang_shap_dims.py --datasets sent,agg` (~23 s). Reproduction status: **REPRODUCIBLE**
(sent/agg re-ran exactly on 2026-10-19); mel artifact-verified only and would need 384 s. Note: a
`--datasets` subset run overwrites the full JSON; back it up first.

**Question.** Do the embedding dimensions that carry language (highest one-way ANOVA η² across
languages) also drive the English topic classifier (highest mean |SHAP|)?

**Script / outputs.**

- `lang_shap_dims.py` -> `lang_shap_dims.json` (numbers), `.md` (tables), one 3-panel plot per dataset
  `lang_shap_dims_{sent,agg,mel}.png`.
- `--datasets sent,agg,mel` selects; `--smoke` is a plumbing check; `--launch` starts a hidden detached
  run.
- Full run 410.6 s: sent 18 s, agg 5 s, mel 384 s. The script is **not resumable**: every run
  recomputes the selected datasets and overwrites the JSON/MD/PNGs (the docstring calls them
  "create-only", but the files are opened with `"w"`). A `--datasets` subset run replaces the JSON
  with that subset only.

**Inputs.**

| dataset | embeddings | labels / split | model |
|---|---|---|---|
| sent | `embeddings/mean/emb_{en,it,de,fr}.npy`, 2009x768, sliced to the 2006 kept rows (git rows 997-999 are gone) | `flores200_en_it_de_fr.csv`; 12 classes; 997 train / 504 val / 505 test, seed 0 | one EN XGBoost, `multi:softprob` |
| agg | `embeddings_agg/mean/emb_{en,it,de,fr}.npy`, 562x768 | `flores_agg.csv`; 11 classes (class 0 dropped); 276 / 140 / 139 | one EN XGBoost, `multi:softprob` |
| mel | `multieurlex/emb_mean_{en,de,fr}_{train,test}.npy`, 11000/5000x768 | `multieurlex/{en,de,fr}_{train,test}.jsonl`, `multi_hot` (21); 11000 EN train, 5000 test | 21 × EN XGBoost, `binary:logistic` |

FLORES rows are parallel across languages. MultiEURLEX test rows are parallel (the `multi_hot` matrices
are identical for en/de/fr); train rows are not. Only EN train labels are used. The EN train rows are
the E18 rows for sent/agg; for mel all 11000 rows are used (E18: 10000 train + 1000 val).

**Part A — language dims.** Per-dim one-way ANOVA η² = SS_between / SS_total over the stacked language
rows (2006/language for FLORES; 16000/language for mel, train+test). `k_f` = number of top dims
covering fraction `f` of the total η² mass. Probe = multinomial logistic regression (StandardScaler,
30% held-out) on up to 2000 (FLORES) / 3000 (mel) rows per language, before and after zeroing the
top-k dims.

**Part B — SHAP.** Fixed English XGBoost (`hist`, `max_bin=64`, `cpu`, `n_jobs=2`, `seed=0`,
`max_depth=8`, `min_child_weight=5`, `lr=0.1`, 400 trees). Train on EN train rows only. Per-dim mean
|SHAP| (`shap.TreeExplainer`), averaged over explained rows and all class outputs (mel: averaged over
the 21 concept models). Explained rows: all 505 (sent) / 139 (agg) test rows; 1000 of 5000 mel test
rows (seed 0). Compare Spearman(η², mean|SHAP|) over 768 dims, top-50 overlap, SHAP mass on the `k50`
η² dims / total SHAP, and the random expectation `k50/768`.

**Part A results.**

| dataset | total η² | max η² | #>0.5 | #>0.25 | k25/k50/k80/k90/k95 | probe full | probe −k50 | probe −k95 |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| sent | 255.8 | 0.774 | 146 | 508 | 104/237/455/557/625 | 1.000 | 1.000 | 0.998 |
| agg | 317.1 | 0.837 | 276 | 598 | 114/252/470/571/636 | 1.000 | 1.000 | 1.000 |
| mel | 474.4 | 0.950 | 549 | 688 | 133/281/495/588/648 | 1.000 | 1.000 | 1.000 |

**Part B results.**

| dataset | Spearman(η², SHAP) | top-50 overlap | SHAP mass on k50 dims | random k50/d | Δ | z vs random |
|---|---:|---:|---:|---:|---:|---:|
| sent | −0.076 | 0/50 | 0.283 | 0.309 | −0.026 | −1.4 (p .08) |
| agg | −0.059 | 1/50 | 0.267 | 0.328 | −0.061 | −2.0 (p .02) |
| mel | −0.019 | 1/50 | 0.366 | 0.366 | +0.00005 | ≈0 (abs z < 0.02) |

**Verdict.** Language is distributed over 237–281 of 768 dims and redundant: the language probe stays
at 1.000 after zeroing those dims (E1 result). The English classifier does **not preferentially use**
the language dims. Their SHAP mass is at the random `k/d` expectation for mel and below it for
sent/agg, Spearman is ≈0 (−0.08 to −0.02), and the top-50 sets do not overlap. This is consistent with
mean-pooled transfer not depending on the language-carrying dims. It does **not** prove that the
classifier ignores them: a model that used all dims uniformly would also give exactly `k/d`.

**Caveats.**

- Single seed; no CI. The "random" baseline is the expectation `k/d`, not a sampled null. Its null sd
  at k50 is 1.8 pp (sent) / 3.0 pp (agg), so the sent deficit is within noise; the agg deficit is ~2σ
  nominal, not corrected for the 9 tests (3 datasets × 3 fractions).
- η² is computed on all rows including test (mel: train+test). For sent/agg, train-only η² gives the
  same k50 (±1) and a 45–49/50 top-50 match; mel not checked.
- Multiclass SHAP weights classes equally; mel weights the 21 concepts equally regardless of
  prevalence (108–6121 positives).
- Spearman p-values are anti-conservative: the 768 dims are correlated.
- Top-50 overlap has chance expectation 3.26 (sd 1.69); 0/50 is p=.03, 1/50 is p=.12.
- Reproduced on 2026-10-19 for sent/agg at stored precision (6 dp). Mel was not recomputed (384 s);
  its numbers are internally consistent and match the run log.

---

## E12/E13/E26 — Leading-dimension removal (global PCA / INLP / random) — DONE (verified 2026-10-19)

**Script to reproduce:** `dim_removal.py` — `python dim_removal.py` (exact flags from the docstring).
Reproduction status: **PARTIAL**; `pc_ablation*.py` is **NOT AVAILABLE (UNK)**.

Flags from the docstring (`dim_removal.py:40-46`): `python dim_removal.py` (foreground full run),
`python dim_removal.py --launch` (detached hidden full run), `python dim_removal.py --smoke
--no-extras` (tiny plumbing check), `python dim_removal.py --render-only` (rebuild md/png from the
JSON).

**Question.** Do the leading directions of the mean embeddings carry language identity, and does
projecting them out (iterative global PCA, k=0..5; INLP; random control) improve an English-trained
XGBoost on topic, multi-label and toxicity tasks, at 210m and 610m?

**Three IDs, one artifact set.** `pc_ablation*.py` is missing. The docstring (line 13) says
`dim_removal.py` is the consolidated reimplementation of the `pc_ablation_flores.pc_ablation_dirs /
apply_dirs` cascade. No separate E13 or E26 artifact or claim survives; this section is the only
evidence for all three IDs.

**Script / outputs.**

- `dim_removal.py` -> `dim_removal.json` (config + `aggregate` + 129 units), `dim_removal.md` (all
  per-k tables), and 8 `dim_removal_<dataset>.png` (`flores_sent`, `flores_agg`, `mel`, `toxic`,
  `xnli`, `pawsx`, `sib200`, `textdetox`).
- `--render-only` rebuilds md/png from the JSON; the run is resumable via
  `%TEMP%\opencode\dim_removal\cells`.
- **Caveat:** every run writes the same repo-root outputs. A `--smoke` check or a `--datasets mel`
  subset run overwrites the full `dim_removal.json/.md` and all 8 PNGs.

**Inputs.** All primary arrays verified present with the shapes below; all label files are present.

| dataset | embeddings (mean) | labels / split | model |
|---|---|---|---|
| `flores_sent` | `emb_bf16/flores_sent/{210m,610m}/mean/emb_{en,it,de,fr}.npy`, 2006x768 / 2006x1152 | `flores200_en_it_de_fr.csv`; 12 classes; 997/504/505; parallel rows | one EN model |
| `flores_agg` | `emb_bf16/flores_agg/{210m,610m}/mean/emb_{en,it,de,fr}.npy`, 562x768 / 562x1152 | `flores_agg.csv`; 11 classes; 276/140/139 | one EN model |
| `mel` | `emb_bf16/mel/{bb}/emb_mean_{en,de,fr}_{train,test}.npy`, 11000/5000 × 768/1152 | `multieurlex/{en,de,fr}_{train,test}.jsonl`, `multi_hot` (21); train re-split 10000/1000, seed 0; parallel 5000-row test | one EN model |
| `toxic` | `emb_bf16/toxic/{bb}/emb_{en,it,de,fr,es}_{train,validation,test}.npy` | `data/toxi_clean/*.parquet`; binary `is_toxic`; EN 24000/3000/3000; targets it 18472/2310, de 9836/1230, fr 21011/2627, es 23349/2919 | one EN model |
| `xnli`, `pawsx`, `sib200`, `textdetox` | 210m only, `%TEMP%\opencode\more_datasets\emb_cache` (60 `.npy`) | `more_datasets.py` loaders; HF dataset cache also required | one EN model |

FLORES-agg test has 139 rows and classes with as few as 4 examples; the per-cell F1 is high-variance.
The four "extra" datasets are not repo-tracked: they need the temp embedding cache and the HF datasets
cache.

**Method.**

- `pc`: iterative **global shared** PCA, k=0..5. Fit PCA on the stacked-language **train** matrix,
  project out the leading component, re-fit on the residual, repeat. The target-language train rows are
  used only as unlabeled features; only EN labels train the model. Verified: the cascade is nested
  (`V5[:3] == V3` exactly), the directions are orthonormal (max err 1.6e-15), the projector is
  idempotent (3e-8), and k=5 removes 26.6% of the total energy on FLORES-agg 210m.
- `inlp`: iterated null-space projection of a language-identity logistic probe
  (`LogisticRegression(max_iter=100, C=1.0)`) on the stacked train pool, capped at k=5. It can return
  fewer than 5 directions (mel 210m returns 2).
- `random`: control, 5 Gaussian-QR orthonormal directions, seed 0. Stored at k=5 only.
- Model: fixed XGBoost `hist`, `max_bin=64`, `n_jobs=2`, `seed=0`, `max_depth=8`,
  `min_child_weight=5`, `lr=0.1`, `n_estimators=400`; multiclass, binary, or 21 one-vs-rest binary
  models for mel. Trained on EN train; evaluated EN test (EN-self) and every target test.
- Metrics: macro-F1 and macro-AUC; **XL-mean** = unweighted mean of the per-target macro-F1.
- In-language: train target -> test target, PC series; skipped for mel (21-label cost).
- **0.975 bar: N/A.** No gap-closure ratio is defined in `dim_removal.md`; derived ratios are context
  only.

**Results.** JSON/MD values at 3 dp. Rows 1–6 are exactly the rows of `EXPERIMENTS.md:153-160`.
`EN-self@best` is the EN-self at the selected best k. `inlp@k5` values marked `*` are stale
pre-revision checkpoints; `flores_sent/210m` re-ran to 0.236, not 0.247.

| dataset | bb | XL-F1 k=0 | best k | best XL-F1 | XL-F1 k=5 | EN-self k=0 | EN-self@best | inlp@k5 | random@k5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mel | 210m | 0.076 | 2 | **0.290** | 0.189 | 0.510 | 0.506 | 0.292 | 0.069 |
| mel | 610m | 0.107 | 2 | **0.341** | 0.228 | 0.492 | 0.495 | 0.268 | 0.099 |
| flores_agg | 210m | 0.204 | 3 | **0.380** | 0.269 | 0.516 | 0.525 | 0.334* | 0.216 |
| flores_sent | 210m | 0.206 | 3 | 0.259 | 0.178 | 0.388 | 0.388 | 0.247* | 0.201 |
| toxic | 210m | 0.467 | 0 | **0.467** | 0.458 | 0.771 | 0.771 | 0.466 | 0.470 |
| toxic | 610m | 0.501 | 0 | **0.501** | 0.463 | 0.770 | 0.770 | 0.488 | 0.501 |
| flores_agg | 610m | 0.284 | 3 | 0.370 | 0.283 | 0.369 | 0.414 | 0.277* | 0.268 |
| flores_sent | 610m | 0.253 | 3 | 0.339 | 0.281 | 0.446 | 0.408 | 0.307* | 0.232 |
| xnli | 210m | 0.284 | 3 | 0.388 | 0.381 | 0.439 | 0.437 | 0.361 | 0.313 |
| pawsx | 210m | 0.522 | 3 | 0.551 | 0.540 | 0.562 | 0.553 | 0.526 | 0.516 |
| sib200 | 210m | 0.486 | 3 | 0.687 | 0.433 | 0.821 | 0.828 | 0.693 | 0.514 |
| textdetox | 210m | 0.591 | 3 | 0.654 | 0.586 | 0.920 | 0.909 | 0.578 | 0.608 |

All 216 per-k cells in the 12 MD tables match the JSON at 3 dp, and the 12-row summary table matches.
The 8 PNGs are valid 1625x624 RGBA files from the same run.

**Verdict.**

- **6/6 EXPERIMENTS rows verified.** The leading-PC removal helps the multi-label / topic cells: mel
  +0.214 (210m) / +0.234 (610m), FLORES-agg 210m +0.175, FLORES-sent +0.053 / +0.085. Peak k is 2 for
  mel (3 languages -> 2 discriminative directions) and 3 for the 4-language FLORES tasks, consistent
  with the language-subspace rank.
- **Toxicity (`toxi_clean`) does not benefit:** XL-F1 is best at k=0; k=5 loses 0.009 (210m) and 0.038
  (610m). But `textdetox` (also binary toxicity) gains +0.063, so "neutral or harmful for toxicity" is
  a `toxi_clean` statement, not a task-level law. Likewise, xnli (+0.104) and pawsx (+0.029) gain, so
  the effect is not topic-specific.
- **Specificity is real at the peak, weaker at small k.** A fresh random k=3 projection on FLORES-agg
  210m gives XL-F1 0.241 (vs PC 0.380); a fresh random k=2 gives 0.257 (vs PC 0.266). The k=2 gain is
  therefore mostly generic, while the k=3 PC peak is +0.139 above the random-k=3 control.
- **0.975 bar: N/A.** For context, the best XL/EN-self macro-F1 ratio in the JSON is 0.997 (PAWS-X
  210m), but PAWS-X is already near parity at k=0 (0.929). Among the headline topic / multi-label cells
  the best ratio is 0.895 (FLORES-agg 610m, k=3), still below 0.975. Derived PC gap closure vs the
  in-language k=0 mean (auditor calculation, not in the artifact): FLORES-sent 30.7% / 71.7%
  (210m/610m), FLORES-agg 65.5% / 55.8%, toxi_clean 0% / 0%; mel has no in-language arm.
- **EN-self is not uniformly flat.** It is flat for mel (range 0.015–0.022) and toxic (0.014–0.019),
  but FLORES-agg 210m spans 0.457–0.568 (0.111) and FLORES-sent 210m spans 0.376–0.428 (0.052). At the
  selected best k the change is small (<= 0.045).

**Caveats.**

- **Best k is selected on the test metric** (argmax over six k values). "Best XL-F1" is therefore an
  oracle-selected maximum; `safe k` is the more honest number. Single seed, no CIs.
- **Stale controls:** 20 of 129 units were reused from pre-revision checkpoints. The FLORES INLP
  control at k=5 does not reproduce with the current code; `flores_sent/210m` is 0.236 fresh vs 0.247
  stored.
- **Fixed, under-tuned baseline:** E28's tuned raw XGB is 0.294 on FLORES-agg 210m vs this
  experiment's k=0 0.204. Part of the small-k gain may be regularization by projection, not language
  removal. The random-k=2 result supports that reading.
- **No threshold calibration:** binary and 21-label macro-F1 use a fixed 0.5 threshold. AUC shows the
  same topic gains, but the F1 magnitudes are threshold-dependent.
- **PCA is fitted on an unbalanced stack** for toxic (EN 24000 rows vs DE 9836), so languages are
  weighted by corpus size; the "global shared PCA" is not language-balanced.
- The "leading 2–3 PCs are 96–98% language-aligned" claim is cited from E1b, which has no surviving
  script/artifact. The audit probe supports the functional claim: language η² on the projected stack
  collapses at k=3 (0.212 -> 0.001 FLORES-agg; 0.158 -> 0.001 FLORES-sent), and the language probe
  drops from 1.000 to about 0.44 (chance 0.25), so removal is substantial but not complete.
- The four extras require a temp embedding cache and HF caches; a clean machine cannot reproduce them
  from the repo alone.

---

## E28 — Backbone × dataset × method grid — DONE (verified 2026-10-19)

**Script to reproduce:** `backbone_grid.py` — `python backbone_grid.py --launch` (all 48 cells;
`--datasets/--backbones/--methods/--trials` narrow it). Reproduction status: **PARTIAL** —
mean/pc/D reproduce exactly, but the stored INLP cells are not bit-reproducible (up to +0.022 XL
macro-F1 drift). The 0.975 bar is **N/A** here: this entry reports absolute XL-mean F1, not a
target/reference ratio.

**Question.** Which feature treatment makes one English-trained XGBoost transfer across languages, and
does the answer depend on the backbone (EuroBERT-210m vs 610m), the dataset, and the target language?

**Script / outputs.**

- `backbone_grid.py` -> `backbone_grid.json` (all 48 cells), `.md` (per-metric tables), `.png` (XL-mean
  macro-F1 bars). The same script also rebuilds `toxi_pr_610.png`.
- `--launch` runs all 6 datasets × 2 backbones × 4 methods detached. `--datasets`, `--backbones`,
  `--methods`, `--trials`, `--workers`, `--smoke` narrow the grid.
- Resumable: per-cell checkpoints in `%TEMP%\opencode\backbone_grid_cells` are reused. Delete a cell
  JSON to recompute that cell. There is no `--render` mode.
- Run: 2026-10-12 19:11 -> 2026-10-13 00:35, 3 workers, 10 trials/cell, no failed cells.

**Inputs.**

| dataset | embeddings (`emb_bf16/<bb>/...`) | labels / split | model |
|---|---|---|---|
| flores_sent | `flores_sent/<bb>/mean/emb_{en,it,de,fr}.npy`, 2006×768/1152 | `flores200_en_it_de_fr.csv`; 12 classes; 997/504/505 (dev 997; devtest 1009 split 504/505, seed 0); parallel rows | one EN model |
| flores_agg | `flores_agg/<bb>/mean/emb_{en,it,de,fr}.npy`, 562×768/1152 | `flores_agg.csv`; 11 classes (class 0 excluded); 276/140/139; 7 rows unused | one EN model |
| mel | `mel/<bb>/emb_mean_{en,de,fr}_{train,test}.npy`, 11000/5000×768/1152 | `multieurlex/{lang}_{train,test}.jsonl`, `multi_hot` (21); train 11000 -> 10000/1000 (seed 0), test 5000; test rows parallel | one EN model |
| sst2 | `sst2/<bb>/mean_{train,val,test}.npy`, 54577/6822/6822×768/1152 | `sst2.csv`, binary; 80/10/10 re-split of GLUE train+validation | EN-only |
| emotion | `emotion/<bb>/mean_{train,val,test}.npy`, 16000/2000/2000×768/1152 | `emotion.csv`, 6 classes | EN-only |
| toxic | `toxic/<bb>/emb_{lang}_{train,validation,test}.npy` | `data/toxi_clean/*.parquet`, binary; per-language 80/10/10 after exact-text dedup; EN 24000/3000/3000; target train sizes 18472/9836/21011/23349 | one EN model |

Backbones: `EuroBERT/EuroBERT-210m` (768-d) and `EuroBERT/EuroBERT-610m` (1152-d). All embeddings are
masked mean pooling of the last hidden state over non-pad tokens, with 512-token truncation; the
forward is bf16, the stored arrays fp32. No CLS/BOS pooling.

**Protocol.**

- `mean`: raw mean embedding.
- `pc`: global PCA k=2 fitted on the stacked-language train pool; remove the two leading components
  from every split.
- `inlp`: iterative logistic language-identity probe on the stacked train pool; project out up to 30
  directions. Actual directions: 30 for flores_sent/flores_agg/toxic and mel/610m; 2 for mel/210m;
  0 for sst2/emotion (single language -> no-op).
- `D`: add `mean_en_train - mean_target_train` to target features; EN features unchanged; no-op when
  there are no targets.
- XGBoost: `hist`, `max_bin=64`, `n_jobs=2`, `seed=0`. 10 Optuna trials per cell, `TPESampler(seed=0)`,
  objective = val AUC (binary AUC; mean concept AUC for multilabel; macro OVR for multiclass). Tune on
  EN val, refit on EN train, score EN test and every target test.
- XL-mean = unweighted mean over non-English targets: FLORES it/de/fr; mel de/fr; toxic it/de/fr/es.
  sst2/emotion have no targets.
- Seeds: 0 for the FLORES split, mel split, toxic split, PCA, Optuna, and XGBoost. No target labels are
  used. `pc`/`inlp`/`D` use **unlabeled** target train features only; `mean` is the only target-free
  method.
- **Optuna caveat:** `TPESampler` defaults to `n_startup_trials=10`. With `n_trials=10` every draw is a
  startup random draw, and the same 10 configurations are used in every cell. Only 4 distinct
  best-parameter sets occur across the 48 cells. This is a fixed 10-config random search, not adaptive
  TPE.

**Results — XL-mean macro-F1.** Bold = best stored method in the row. `n/a` = no non-English target.

| dataset | backbone | mean | pc | inlp | D |
|---|---:|---:|---:|---:|---:|
| flores_sent | 210m | 0.199 | 0.207 | 0.206 | **0.209** |
| flores_sent | 610m | 0.241 | 0.265 | **0.307**† | 0.274 |
| flores_agg | 210m | **0.294** | 0.282 | 0.292 | 0.293 |
| flores_agg | 610m | 0.302 | **0.340** | 0.301 | 0.296 |
| mel | 210m | 0.059 | 0.293 | **0.299**† | 0.298 |
| mel | 610m | 0.115 | 0.338 | **0.349**† | 0.329 |
| sst2 | 210m | n/a | n/a | n/a | n/a |
| sst2 | 610m | n/a | n/a | n/a | n/a |
| emotion | 210m | n/a | n/a | n/a | n/a |
| emotion | 610m | n/a | n/a | n/a | n/a |
| toxic | 210m | 0.470 | 0.450 | **0.482**† | 0.447 |
| toxic | 610m | **0.500** | 0.465 | 0.494 | 0.452 |

† INLP is the stored best; it is not bit-reproducible. A gap below ~0.02 between `inlp` and `pc`/`D` is
not robust.

**Results — EN-self macro-F1.**

| dataset | backbone | mean | pc | inlp | D |
|---|---:|---:|---:|---:|---:|
| flores_sent | 210m | 0.375 | 0.378 | **0.386** | 0.375 |
| flores_sent | 610m | **0.403** | 0.403 | 0.391 | 0.403 |
| flores_agg | 210m | **0.493** | 0.457 | 0.492 | 0.493 |
| flores_agg | 610m | 0.406 | 0.403 | **0.429** | 0.406 |
| mel | 210m | 0.509 | 0.521 | **0.523** | 0.509 |
| mel | 610m | 0.501 | **0.517** | 0.495 | 0.501 |
| sst2 | 210m | 0.805 | **0.809** | 0.805 | 0.805 |
| sst2 | 610m | **0.765** | 0.762 | **0.765** | **0.765** |
| emotion | 210m | 0.365 | **0.374** | 0.365 | 0.365 |
| emotion | 610m | 0.358 | **0.399** | 0.358 | 0.358 |
| toxic | 210m | **0.770** | 0.750 | 0.767 | **0.770** |
| toxic | 610m | 0.784 | 0.774 | **0.788** | 0.784 |

**Results — XL-mean AUC** (threshold-free; sst2/emotion have no XL cells).

| dataset | backbone | mean | pc | inlp | D |
|---|---:|---:|---:|---:|---:|
| flores_sent | 210m | 0.811 | 0.808 | 0.817 | **0.821** |
| flores_sent | 610m | 0.829 | 0.827 | **0.837** | 0.830 |
| flores_agg | 210m | 0.881 | 0.886 | **0.887** | 0.874 |
| flores_agg | 610m | 0.885 | **0.891** | 0.875 | 0.883 |
| mel | 210m | 0.807 | **0.866** | **0.866** | 0.864 |
| mel | 610m | 0.839 | 0.880 | **0.881** | **0.881** |
| toxic | 210m | **0.706** | 0.691 | 0.698 | 0.685 |
| toxic | 610m | **0.730** | 0.704 | 0.705 | 0.728 |

Full per-metric tables are in `backbone_grid.md`; the JSON also contains per-target scores.

**Verdict.**

- The grid is **48 cells**, not 20: 6 datasets × 2 backbones × 4 methods. 32 cells have XL targets; 16
  are EN-only. The earlier `EXPERIMENTS.md` E28 table listed only five dataset/backbone rows (20
  cells), and the omitted 210m rows change the story.
- **mel:** raw mean collapses on targets (0.059 at 210m, 0.115 at 610m). All three target-adaptive
  methods recover most of it (0.293–0.299; 0.329–0.349). `D` alone (a mean shift) reaches 0.298/0.329,
  i.e. 99%/91% of the best gain; `pc`/`inlp` add at most 0.001/0.020 over `D`. Macro-F1 gains are much
  larger than AUC gains (0.807 -> 0.866; 0.839 -> 0.881) and XL mel subset accuracy stays <= 0.094, so
  this is mostly threshold / calibration recovery, not exact multi-label accuracy.
- **flores_sent:** only 610m benefits materially: mean 0.241 -> inlp 0.307 (D 0.274); 210m is flat
  (0.199 -> 0.209). AUC moves little (0.811 -> 0.821; 0.829 -> 0.837).
- **flores_agg:** 610m `pc` 0.340 vs mean 0.302; at 210m no method beats mean (0.294; `pc` 0.282). `D`
  does not help (0.296/0.293).
- **toxic:** 610m raw mean 0.500 is best; `pc`/`inlp`/`D` give 0.465/0.494/0.452. At 210m `inlp` 0.482
  vs mean 0.470 (+0.012, within INLP noise). AUC is 0.730 for 610m mean vs 0.704–0.728 for the others.
  The fixed 0.5 threshold and the EN/target prior gap make toxic macro-F1 calibration-sensitive.
- **sst2/emotion (EN-only):** `inlp` (0 dirs) and `D` (no targets) are exact no-ops, so 8 of the 16
  EN-only cells duplicate `mean`. Only `pc` changes the result (e.g. emotion/610m 0.358 -> 0.399). They
  are not method comparisons.
- **Bottom line:** the robust effect is that a target-adaptive feature transform (even a simple mean
  shift) is necessary for multi-label topic transfer on mel. The evidence does not isolate
  "language-subspace removal" as the mechanism, and it does not support a uniform gain on single-label
  topic, sentiment, emotion, or toxicity. The 210m and 610m backbones disagree, and the specific `inlp`
  ranking is within its reproducibility noise.

**Caveats.**

- **Optuna:** 10 trials, all startup random draws; no adaptive TPE; only 4 distinct best-parameter sets
  across 48 cells. Single seed, no CIs.
- **INLP is not bit-reproducible.** Recomputing four inlp cells with the stored best params changed XL
  macro-F1 by -0.001 to +0.022 (flores_agg/210m 0.292 -> 0.314; flores_sent/210m 0.206 -> 0.216;
  flores_agg/610m 0.301 -> 0.306; flores_sent/610m 0.307 -> 0.306). Mean, pc, and D reproduce exactly.
  The INLP basis is not saved. The likely cause is environment drift: NumPy was updated on 2026-10-14,
  after the 13/10 run. The run did not record library versions or a code hash.
- **Metrics:** binary / multilabel scores use a fixed 0.5 threshold; no target threshold tuning.
  AUC gains are much smaller than macro-F1 gains (mel 610m +0.042 AUC vs +0.234 macro-F1; flores_sent
  610m +0.008 vs +0.066). Report AUC and, for toxic, PR-AUC before calling a method "better".
- **Target adaptation:** `pc`/`inlp`/`D` fit on unlabeled target train features; only `mean` is
  target-free. The gain conflates language-subspace removal with target-domain adaptation. E18's
  `perlang_center`/`perlang_z` controls are the relevant comparison; E28 has no EN-only PCA/INLP
  control.
- **mel is long-doc but truncated:** all texts are cut at 512 tokens; mean pooling sees only the first
  512 tokens. Treat "long-doc mel" as "first-512-token mel".
- **mel train rows are not parallel across languages.** Test rows are: all 5000 `multi_hot` vectors
  match across en/de/fr. Train `multi_hot` match is only 169/11000 (en-de), 183/11000 (en-fr),
  180/11000 (de-fr). `_load_mel` assigns EN labels to de/fr train and EN val indices to de/fr arrays;
  those values are unused by the current metrics, but the code is wrong if target-language tuning or
  target validation is added.
- **toxic:** per-language 80/10/10 after exact-text dedup; cross-split exact-text overlap is 0.
  Cross-language translation near-duplicates are not removed. Target test sizes differ (it 2310, de
  1230, fr 2627, es 2919) and prevalence differs (EN 12.1%; it 29.4%, de 31.0%, fr 36.9%, es 31.3%).
  The XL mean is unweighted, so de gets the same weight as es.
- **sst2/emotion:** sst2 is a 80/10/10 re-split of GLUE train+validation (68,221 rows); 61 exact texts
  overlap train/test (and 58 train/val). emotion has 11 train/test duplicates. These are EN-only and
  not central to E28, but they are mild leakage.
- **Embedding provenance:** 210m arrays were built by a preceding process at 18:54–19:09 and reused
  (`embeddings[210m]: nothing to build`); 610m arrays were built in the logged run. Both match the
  committed 210m/610m fp32 caches (row-wise cosine ~0.998–0.9997). The 210m build is not in the
  retained log.
- **Artifacts:** JSON/MD/PNG are from 2026-10-13 00:35:12; the log ends `ALL DONE` at 19,397 s and the
  status is `done`. `backbone_grid.json` writes NaN for EN-only XL cells; strict JSON parsers reject
  that. `toxi_pr_610.png` uses a separate fixed XGBoost config and per-language z-scoring, not the
  grid's tuned cells.

---

## E41 — Dimensionality diagnostics: PCA, η², SHAP — DONE, evidence PARTIAL (audited 2026-10-20)

**Script to reproduce:** `pca_plots.py` / `lang_dims.py` / `shap_flores_lang.py` /
`shap_flores_dropped.py` — `python pca_plots.py`; `python lang_dims.py`;
`python shap_flores_lang.py`; `python shap_flores_dropped.py` (from `multi\`, no arguments; each
script writes its own outputs). Reproduction status: **PARTIAL** (PNGs survive; the SHAP JSON/MD were
deleted).

**Status.** This entry was not in the pre-audit file. `README.md:35-40` lists the four scripts.
`COVERAGE.md:96-99` assigns them E41. The SHAP `.json`/`.md` were never tracked (`.gitignore:55-56`)
and are gone. The 15 PNGs under `pca_plots/` and the four scripts survive; the four PNGs spot-checked
match `HEAD` byte for byte.

**Question.** Three questions share these scripts:

1. Do the four FLORES languages form separate clusters, and does a mean shift to English remove them?
2. How many embedding dimensions carry the language?
3. Does the English topic model spend its SHAP budget on those dimensions?

**Script / outputs.**

| script | purpose | artifact present? | can it be re-run? |
|---|---|---|---|
| `pca_plots.py` | 2-D PCA panels, 4 pooling modes: `raw` (PCA fit on all 4 languages pooled) vs `+D` (each language mean-shifted onto English, PCA fit on English); all-but-the-top (k=1) variants; mean-mode language-vs-topic view | **10/10 PNGs**, tracked | **NO as written** — dies at `X.load_data("lenient")` (`pca_plots.py:93`, `AssertionError` in `xgboost_topic_optuna.py:230`, CSV now 2006 rows) *after* rewriting 8 of the 10 PNGs |
| `lang_dims.py` | per-dim one-way ANOVA η² over en/it/de/fr + cumulative mass curve for `mean` and `bos` | **1/1 PNG**, tracked | **YES** — no args, no CLI; seconds; numbers reproduced exactly (it never touches the CSV) |
| `shap_flores_lang.py` | 8 lenient-tag binary XGBoost models (EN dev+devtest, 1503 rows); SHAP on the 506 EN test rows; Spearman(η², mean\|SHAP\|), top-50 overlap, SHAP mass on the k25/k50 η² dims | **PNGs 2/2**; `shap_flores_lang.json` + `.md` **MISSING** | **NO as written** — fails on its first statement (`X.load_data`, `:36`) |
| `shap_flores_dropped.py` | same models; attribution summed over the top-k η² dims ("dropped") vs the rest ("kept"), for EN model on EN/IT test inputs and IT model on IT test inputs; `share_dropped` vs `k/768` | **PNGs 2/2**; `.json` + `.md` **MISSING** | **NO as written** — same `X.load_data` failure (`:51`); 32 binary models once fixed |

**Blocker for three of the four scripts.** `flores200_en_it_de_fr.csv` now has 2006 data rows (commit
`00d4823` removed three), while `embeddings/*` still has 2009 rows and `xgboost_topic_optuna.load_data`
still asserts `len(df) == 2009` (`:230`). `lang_dims.py` is unaffected because it reads the `.npy` files
directly. Fix either way: load the 2009-row CSV through
`train_flores12.GIT_REF = "196bbd3:flores200_en_it_de_fr.csv"`, or slice the arrays to the 2006 kept
rows (`train_flores12.load_sent_embeddings`).

No script takes a flag: no `--launch`, no resume, no PID/status file, no seed option. A crash loses the
run. Every run overwrites its outputs.

**Inputs.**

| use | data |
|---|---|
| PCA / η² | `embeddings/{mean,bos,eos,lead}/emb_{en,it,de,fr}.npy`, 2009x768 each; all 2009 rows, train+test, unsupervised |
| SHAP | `flores200_en_it_de_fr.csv` (2006 data rows today; the loader wants 2009), `lenient` target = 8 binary tags (`xgboost_topic_optuna.TAGS`); split 997 dev / 506 devtest / 506 test, seed 0; train on dev+devtest, explain test |
| model | `xgboost_topic_optuna.BASELINE_PARAMS` = `max_depth=6, lr=0.1, n_estimators=300`, `hist`, `max_bin=64`, `n_jobs=2`, seed 0, `binary:logistic` (8 independent tag models) |

**What the PNGs show.**

- `pca_lang_mean.png`: `raw` gives four tight, disjoint language blobs. `+D` gives one overlapping
  cloud. The `+D` overlap is forced: the shift sets every language's centroid onto the English centroid
  (measured residual 5e-5), so that panel cannot show anything about language removal.
- `pca_lang_bos.png`: `raw` separation is much weaker than `mean` (ratio 2.44 vs 27.0; de/it/fr partly
  overlap). Its `+D` panel also leaves a small detached cluster of outlier rows on the far right — a
  mean-only shift does not move atypical sentences, and it does not change the per-language spread
  (English stays the widest cloud).
- `pca_lang_mean_noPC1.png`: after removing the mean and the top PC, the four blobs are still
  separate. All-but-the-top (k=1) does not remove language.
- `pca_mean_lang_topic{,_noPC1}.png`: topics are mixed inside every language blob. No topic cluster is
  aligned with a language.
- `lang_dims_cumulative.png` (legend, exact): `mean` 50% = **237** dims, 90% = **557**; `bos` 50% =
  **211**, 90% = **545**.
- `shap_vs_langdims_{mean,bos}.png` (titles, exact): Spearman **−0.081** (`mean`), **−0.033** (`bos`).
- `shap_dropped_vs_kept_{mean,bos}.png`: three model/input groups, two panels (k25, k50), bars for
  "dropped" and "kept".

**Checked numbers (independent recompute, 2009 rows/language, `lang_metrics.eta2`).**

| mode | total η² | max η² | #>0.5 | k25 | k50 | k80 | k90 | k95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean | 255.84 | 0.774 | 145 | 104 | 237 | 455 | 557 | 625 |
| bos | 139.96 | 0.558 | 9 | 87 | 211 | 434 | 545 | 618 |
| eos | 289.16 | 0.800 | 214 | 110 | 246 | 467 | 571 | 640 |
| lead | 93.04 | 0.435 | 0 | 80 | 199 | 417 | 528 | 606 |

Only `mean` and `bos` are plotted. The plotted k values match this table exactly. `eos` carries the
most language signal and `lead` the least; neither is in the figure.

PCA structure (the script's own procedure; between-language / within-language sum-of-squares ratio in
the 2-D coordinates):

| panel | PC1 + PC2 explained | language separation ratio |
|---|---|---|
| `mean` raw (pooled fit) | 14.9% + 13.6% | 27.0 |
| `mean` +D (EN fit) | 7.4% + 4.1% | 0.000 (forced) |
| `mean` all-but-top k=1, raw | — | 29.7 |
| `bos` raw | 11.5% + 7.4% | 2.44 |
| `bos` +D | 18.0% + 8.2% | 0.000 (forced) |
| `bos` all-but-top k=1, raw | — | 1.87 |

**SHAP numbers.** The PNG titles give Spearman exactly. The bar heights give the shares to about
±0.01 only.

| mode | Spearman(η², \|SHAP\|) | k25 dims | SHAP mass on k25 | random 25/100 | k50 dims | SHAP mass on k50 | random 50/100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean | −0.081 | 104 | ≈0.119 | 0.1354 | 237 | ≈0.291 | 0.3086 |
| bos | −0.033 | 87 | ≈0.111 | 0.1133 | 211 | ≈0.280 | 0.2747 |

Dropped-vs-kept shares (≈, read from bars): all six `mean` cells sit below `k/768` (≈0.12–0.13 at
k=104, ≈0.28–0.32 at k=237). In `bos`, the EN-model / IT-input cell at k=211 is the only one above
random (≈0.31 vs 0.275). Nothing here is more than 2 pp from `k/d`.

**Deleted numbers — UNVERIFIED.** The lost `shap_flores_lang.json`/`.md` and
`shap_flores_dropped.json`/`.md` also held: top-50 overlap counts, exact `mass25`/`mass50`, the per-mode
top-5 language dims with their SHAP ranks, the top-10 SHAP dims, and per-mode runtimes. None of these
can be checked from the PNGs. They need a re-run.

**Relation to E17 / E18 / E25 — do not double-count.**

- **E17 is the successor for the SHAP result.** These scripts carried a private `eta2` copy;
  `lang_metrics.py` now holds it verbatim. E17 uses a second copy of the same formula,
  `lang_neutralize.eta2_per_dim` (float64). `lang_dims.py`'s `mean` curve **is** E17's Part A: the
  recompute matches E17's `sent` row exactly when the three dropped rows are removed (255.84 / 0.774 /
  146 / 508, k = 104/237/455/557/625). The first 237 η² dims are the same set on 2009 rows and on that
  2006-row slice (overlap 237/237). The SHAP pair asks E17's question on a **different model**: 8 lenient
  binary tags at depth 6 / 300 trees instead of E17's 12/11-class `multi:softprob` at depth 8 / mcw 5 /
  400 trees. It is a precursor, not a replication.
- **E18 supplies the accuracy numbers; E41 supplies none.** The `+D` panel is the picture form of E18's
  `perlang_center`: `X_l + mean(X_en − X_l) = X_l − mean(X_l) + const`. E18 computes the target mean
  from unlabeled train rows; `pca_plots.py` uses all 2009 rows, test included. E41 is a mechanism
  picture for E18, not a result of it.
- **E25 is unrelated except in inputs.** E25 uses `perlang_z` on 8 datasets and reports transfer gaps.
  E41 has no classifier metric and says nothing about any gap.

**0.975 bar — N/A.** These are diagnostics. No script computes a target/English ratio, an F1 or an
AUC; the SHAP scripts never score the models they explain. No E41 artifact can support or refute a
0.975 claim.

**Verdict.**

- The four languages are cleanly separated in raw mean space (ratio 27.0) and stay separated after the
  top PC is removed (29.7). Language is spread across the vector: ~237 of 768 dims (mean) / 211 (bos)
  hold 50% of the η² mass. This agrees with E1/E17's redundancy result.
- The `+D` panels are degenerate. Mean-shifting equalises the language centroids by construction, so
  the overlap is an arithmetic identity, not a finding about residual language information.
- The task model's attribution on the language dims is ≈ `k/768`: −1.6 pp (mean, k25), −1.8 pp (mean,
  k50), −0.2 pp (bos, k25), +0.5 pp (bos, k50). All are inside the 1.8–3.0 pp null noise E17 measured.
  The honest reading is "no preferential use", not "does not use".

**Caveats.**

- **Three of the four scripts are not re-runnable today.** `00d4823` (2026-10-05 21:05) cut the CSV
  from 2009 to 2006 rows; the SHAP PNGs were committed six hours earlier (`0d45288`, 09:56) and
  `xgboost_topic_optuna.load_data` was never updated. The PNGs are genuine artifacts of the 2009-row
  data. A re-run needs the row alignment fixed first.
- The SHAP JSON/MD were never committed; the underlying per-dim vectors and rank tables are lost. Only
  the plotted Spearman values and approximate bar heights survive.
- η² uses all 2009 rows (train+test). The dims used to group the SHAP cells are therefore chosen with
  test rows in them. No model sees them, but the group boundary is not train-only.
- `xgboost` is unpinned in `requirements.txt` (installed 3.2.0). The PNGs date from 2026-10-05. A
  re-run today may not reproduce them exactly.
- Single seed, no CIs. The "random" baseline is the expectation `k/d`, not a sampled null.
- The SHAP shares are shares of a non-negative importance mass, so `k/d` is the correct null, but the
  two scripts average absolute SHAP over 8 tags with no prevalence weighting.
- `pca_plots.py`'s docstring says "script + outputs are gitignored". That is stale: both are tracked.

---

# PART C — Task-level and data-level results

The entries that test whether the headline result is a property of the method or of the data: the
missing-script FLORES work (E3/E6/E7), the long-document MultiEURLEX line and its E16 alias
(E9/E10/E11), the gap-closers (E14/E14b), sentiment (E4), the translated caches (E2), and the three
new task/data entries (E33 agg P/R, E34 chunk pooling, E35 error buckets).

## E3/E6/E7 — FLORES topic cross-lingual — UNVERIFIED (no script, no artifact)

**Script to reproduce:** NOT AVAILABLE — `crosslingual_topic*.py` and `crosslingual_inlp.py` never
existed in git; no result artifact survives. **(UNK)**

None of the cited filenames was ever tracked: `git log --all --oneline -- crosslingual_topic.py`,
`..._topic_agg.py`, `..._topic_sent.py` and `crosslingual_inlp.py` all return nothing;
`git log --all --full-history -- "*crosslingual*"` is empty; `git ls-files` contains no `crosslingual*`
file. No JSON/MD/PNG/log/PID for this entry exists in the working tree, in the 98-commit history, or
under `%TEMP%\opencode`.

**Question.** Does an English-trained FLORES topic classifier transfer to it/de/fr, and does language
neutralization (`inlp` / `subspace` / `zero90`) close part of the gap?

**Only record.** `EXPERIMENTS.md:169-175` — a five-line table introduced in commit `21ab888`
(2026-10-12). That commit added 49 files; none is a `crosslingual_*` script and none is an E3/E6/E7
artifact. `REPORT.md` and `REPRODUCE.md` at `21ab888` never mention E3/E6/E7. The entry is
documentation-only, not a report generated from a stored run. E6 and E7 have no separate claim or
artifact in any document; they are additional IDs for the same missing line of work.

**Inputs.** Unknown. The surrounding study and commit `21ab888` describe mean-pooled EuroBERT-210m +
XGBoost, and the current data inventory is FLORES sent (2,006 kept rows; 997/504/505) and FLORES agg
(562 rows; 276/140/139), 12/11 classes, en/it/de/fr. The entry itself does not state backbone, split,
threshold policy, seed, or per-language values. Do not assume the surviving FLORES split was the one
used.

**Claims (all UNVERIFIED).** "Mean macro-F1 over targets" is the entry's only metric definition.

| dataset | classes | condition | reported raw | reported best | best method | status |
|---|---|---|---:|---:|---|---|
| FLORES sent | 12 | EN → it/de/fr | 0.234 | 0.334 | `inlp` | **UNVERIFIED** |
| FLORES agg | 11 | EN → it/de/fr | 0.246 | 0.356 | `zero90` / `subspace` | **UNVERIFIED** |

These four numbers are the complete quantitative record. There is no per-target breakdown, no seed, no
threshold policy, no English-self value, and no artifact to check them against.

**Study bar.** The study bar is **0.975** (target/English ratio). **N/A** for this entry: it reports
absolute macro-F1 only and stores no per-condition English-self reference, so no ratio can be computed
without inventing a denominator. The historical `21ab888` text mentions "EN 0.559" for agg, but that is
an unverified narrative note, is absent from the current entry, and does not define a per-condition
ratio.

**Nearest surviving equivalents (different protocols; none reproduces these numbers).**

| script | protocol difference | relevant surviving cells | reproduces E3/E6/E7? |
|---|---|---|---|
| `flores_close.py` | 210m mean, fp32; same FLORES split; EN-only training; primary metric = OVR **micro**-F1 at a global F1-optimal threshold tuned on EN val; INLP fitted on train+val (6 rounds); reports macro-F1 per target | sent raw 0.2908 micro (macro mean 0.2503); sent `inlp` 0.3558 micro (macro mean 0.3043); agg raw 0.2774 micro (macro mean 0.2623); agg `inlp` 0.3155 micro (macro mean 0.2860) | **No.** 0.3558 rounds to 0.356, but it is sent micro-F1, not agg best; raw values are far above the claimed 0.234/0.246 |
| `dim_removal.py` | bf16 re-embedded features, fixed XGB, **XL-mean macro-F1 / AUC**; INLP is a fixed k=5 projection fitted on train only; PC series k=0–5 | sent raw 0.206, `inlp` 0.247, `pc` k=3 0.259; agg raw 0.204, `inlp` 0.334, `pc` k=3 0.380 | **No.** 0.334 is agg/210m INLP, not sent; no cell equals 0.234/0.246/0.356 |
| `proof_mean_alllangs.py` | one EN XGB, mean embeddings, **model-free feature transforms only** (no INLP, no `zero90`); headline metric = AUC | sent raw AUC EN 0.856 vs it/de/fr 0.759/0.763/0.768; agg raw AUC EN 0.895 vs 0.824/0.807/0.795 | **No.** Different metric; no neutralization arm |
| `agg_pr_2models.py` | FLORES-agg precision/recall vs threshold, 210m/610m; **non-runnable** (imports missing `agg_pr_methods`) and cites missing `crosslingual_topic_agg` / `lang_inlp_agg.npz` | `agg_pr2.*` exists, but it is a threshold sweep, not the E3/E6/E7 prototype | **No.** Provenance chain is broken |

Macro means shown in parentheses for `flores_close.py` are arithmetic means of the three stored 3-dp
per-target cells, not stored fields. Running any of the above would create a **new**
protocol-specific result, not reproduce the claimed 2026-10-12 numbers.

**Verdict.** **UNK / UNVERIFIED.** The entry cannot support a 0.975 claim or any comparative claim. If
the numbers are retained, label them "reported 2026-10-12; no surviving script or artifact; not
reproduced" and keep them out of verified result tables.

**Caveats.**

- `zero90` is named only in documentation (`21ab888`, `6cff2d5`, `d37e338`) and is never defined in
  code. Only `lang_neutralize.zero_dims`, `language_subspace` and `project_out` survive; no record says
  which dims/rank/basis `zero90` used.
- `agg_pr_2models.py:21,121,185` calls `crosslingual_topic_agg.run_inlp` and expects
  `lang_inlp_agg.npz`; neither exists (`*.npz` is gitignored), so even the INLP basis cannot be
  recovered from the repo.
- `EXPERIMENTS.md:384-385` records a destructive cleanup ("a buggy cleanup deleted all files under
  `multi/`"). Git proves the files were never tracked; it cannot prove they never existed as untracked
  local files.
- Single seed, no confidence intervals, no per-language rows: none of this can be checked.

---

## E9/E10/E11 — MultiEURLEX long-document transfer — PARTIAL (audited 2026-10-19)

**Script to reproduce:** `mel_close.py` — `python mel_close.py --launch` (foreground: `python
mel_close.py`). Reproduction status: **PARTIAL**; `multieurlex_*.py` is **NOT AVAILABLE (UNK)**.

**Question.** With one English-trained one-vs-rest XGBoost on mean-pooled EuroBERT features, how close
does English -> de/fr transfer get to the target language's own in-language model on MultiEURLEX
(21 EuroVoc concepts, long EU legal documents)? Do `l2_thenZ` (per-language L2 + z-score), threshold
calibration, and a larger backbone (610m) close the gap to the **0.975** comparability bar?

**Three IDs, one surviving artifact.** E9, E10 and E11 are the same line of work (long-document
transfer, calibration, 610m). Only the 210m `mel_close.py` run survives in full.

**Scripts / outputs.**

| item | path | status |
|---|---|---|
| surviving implementation | `mel_close.py` | present, 974 lines; 210m only; committed unchanged since `21ab888` |
| 210m results | `mel_close.json`, `mel_close.md`, `mel_close_plots.png` | present; generated in one 1877 s run (PID 30324, 2026-10-12 04:55 -> 05:26) |
| run log / status | `mel_close.log`, `mel_close_status.json` | present; log matches the JSON/MD exactly |
| cited E9/E10/E11 scripts | `multieurlex_*.py` | **absent**; never committed (`git rev-list --all` finds no `multieurlex` path) |
| 610m results | none | **absent**; no JSON/MD/PNG for 610m MultiEURLEX |

The 610m row in the E9 table exists only as prose in the **initial commit** (`21ab888:EXPERIMENTS.md`,
section "E9/E9b/E10/E11") and in the old E16 note; no run artifact was ever committed. The current
`multieurlex_610/` caches were created 2026-10-12 12:46–16:19, **after** that commit (08:05), and no
current script reads them (`backbone_grid.py:93` explicitly says it never touches `multieurlex_610/`).
`REPRODUCE.md` has no command that produces a 610m MultiEURLEX number.

**Inputs / protocol (`mel_close.py`, 210m).**

| item | value |
|---|---|
| dataset | `nlpaueb/multi_eurlex`, en/de/fr; 11,000 train / 5,000 test; official validation **not** used |
| labels | 21 level-1 EuroVoc concepts, `multi_hot` from each language's own JSONL |
| embeddings | `multieurlex/emb_mean_{en,de,fr}_{train,test}.npy`; mean pool, 512-token truncation; 768-d |
| split | 1,000-row VAL carved from EN train (seed 0); 10,000 train; 5,000 test. The old E9/E15 numbers may use the full 11,000 train rows (E15 v2 config), so the E9 table mixes 10k and 11k protocols |
| transform | `l2_thenZ [both]`: L2, then z-score with each language's own **unlabeled** train mean/std |
| model | 21 × `XGBClassifier(binary:logistic, hist, depth 6, mcw 1, 200 trees, lr 0.1)` |
| threshold | global threshold tuned on EN val to maximise EN micro-F1; `qm` = per-concept quantile matching to EN-val true positive rates |

The 610m block (E9) used a different recipe: 1152-d, `l2_thenZ`, `quantile_match`, and per the old
write-up depth 8 / min_child_weight 5 (the E15 v2 / `lang_matrix` recipe also uses 400 trees). No
script or output for it survives.

**Splits / leakage checks (verified directly).**

- **Test split is parallel.** `en/de/fr_test.jsonl` have 5,000 rows each and identical `multi_hot`
  row-for-row (5000/5000 for both de and fr). Test labels can be shared.
- **Train split is NOT parallel.** `en_train.jsonl` vs `de_train.jsonl` share only 169/11,000 label
  rows; en vs fr share 183/11,000 (chance level). The `mel_close.py` docstring (lines 16-17) and report
  header (line 824) claim "documents are parallel across languages" — **false for train**. Only test is.
- No target labels are used to fit any English-trained method. VAL is an EN-train slice. `qm` uses
  target **test scores** (not labels) to match EN-val positive rates: label-free but transductive.
- The ensemble's top-3 members and the kNN `k` are selected on **target test** micro-F1
  (`mel_close.py:626-629`, `734-737`), so the ensemble number is test-selected / optimistic.

**Results (micro-F1, 21 labels).**

| backbone | representation | de | fr | source | status |
|---|---|---:|---:|---|---|
| 210m | raw | 0.266 | 0.334 | E9 table (fixed 0.5 threshold) | reproduced in spirit: E18 raw@0.5 = 0.256 / 0.341 |
| 210m | raw | 0.343 | 0.464 | `mel_close.json` (tuned t=0.34) | verified in artifact |
| 210m | `l2_thenZ` (no calib) | 0.643 | 0.641 | `mel_close.json` (t=0.30) | verified in artifact |
| 210m | `l2_thenZ` + calib (`qm`) | 0.661 | 0.662 | `mel_close.json` | verified in artifact |
| 210m | `l2_thenZ` + calib | **0.688** | **0.683** | E9 table (old depth8/mcw5 run) | **NOT REPRODUCED** (1.6–4.5 pp above every surviving artifact) |
| 210m | translate-train | 0.716 | 0.707 | `mel_close.json` | verified (depends on external MT cache) |
| 210m | ensemble (tt + kNN20 + kNN10) | 0.728 | 0.710 | `mel_close.json` | verified; members chosen on test |
| 610m | `l2_thenZ` + calib | **0.711** | **0.707** | E9 prose only | **UNVERIFIED / NOT AVAILABLE** |
| — | in-language ceiling | **0.714** | **0.712** | E9/E16 prose only | **UNVERIFIED / NOT AVAILABLE** |
| 210m | in-language ceiling (mel_close JSON) | 0.437 | 0.460 | `mel_close.json` | **BUGGY** — trained on EN labels; AUC 0.470/0.523 |

**Independent 210m cross-check.** `proof_mean_alllangs.py` (E18) uses the same 10k/1k/5k split with a
different script and depth-8 XGB: `perlang_z` gives EN 0.732, de 0.651, fr 0.652 (EN-val-tuned
threshold) and de 0.670 / fr 0.667 with quantile matching. `raw` at threshold 0.5 gives de 0.256 /
fr 0.341. These bracket the `mel_close` numbers and confirm that the E9 210m `raw` row is a fixed-0.5
number, while the E9 0.688/0.683 row is above both surviving implementations.

**Independent 610m cross-check (partial).** `lang_matrix.py` (same `perlang_z`, depth 8 / mcw 5 / 400,
threshold 0.5, per-language train labels) reports for 610m mel: EN-self macro 0.600, EN->de 0.371,
EN->fr 0.449, native de 0.567, native fr 0.562, AUC 0.889/0.891 (EN->de/fr). The E9 610m EN-self macro
0.601 and native macro 0.546/0.545 are in the same range, so the run is **plausible**, but
`lang_matrix.json` stores no micro-F1 and cannot confirm 0.711/0.707 or AUC 0.872/0.875.

**The 0.975 bar.**

| row | ratio definition | de | fr | >= 0.975? |
|---|---|---:|---:|:--:|
| 610m + `l2_thenZ` + calib | vs in-language ceiling 0.714 / 0.712 | 0.996 | 0.993 | **PASS** (numerator unverified) |
| 610m + `l2_thenZ` + calib | vs English self 0.752 | 0.946 | 0.940 | FAIL |
| 210m + `l2_thenZ` + calib | vs in-language ceiling 0.714 / 0.712 | 0.964 | 0.959 | **FAIL** |
| 210m `mel_close` `l2_thenZ` | vs in-language ceiling | 0.901 | 0.900 | FAIL |
| 210m `mel_close` `l2_thenZ` (qm) | vs EN self micro-qm 0.717 | 0.922 | 0.923 | FAIL |
| 210m `mel_close` ensemble | vs in-language ceiling | 1.019 | 0.997 | PASS (test-selected) |
| 210m translate-train | vs in-language ceiling | 1.003 | 0.993 | PASS (uses MT) |

The verdict is denominator-dependent. Against the **in-language ceiling** the 610m calibrated row
passes (0.996 / 0.993) and the 210m calibrated row fails (0.964 / 0.959). Against **English self** (the
E18 definition of the 0.975 bar) the 610m row fails (0.946 / 0.940) and the E9 210m row fails as well
(0.915 / 0.909 at EN self 0.752). Only translate-train (which uses a second model, MT) clears 0.975 on
both targets against its own English self.

**Verdict.** The 210m `mel_close` results are real and internally consistent, but they do **not**
support the E9 210m 0.688/0.683 row: the committed script gives 0.643/0.641 (tuned) or 0.661/0.662
(quantile-matched), and an independent script gives 0.651/0.652 (0.669/0.667 qm). The 610m 0.711/0.707
row and the 0.714/0.712 native ceiling are **text-only claims** from missing `multieurlex_*` scripts;
they are plausible (corroborated in macro-F1 range by `lang_matrix`) but have no surviving artifact. At
0.975, 610m passes only if the denominator is the (unverified) native ceiling; 210m does not pass
against that ceiling.

**Caveats.**

- Single seed, no CIs. 5,000 test documents, only 21 labels.
- Test documents are parallel translations, so the effective independent sample is 5,000 documents;
  label counts are dominated by a few concepts.
- `qm` is transductive: thresholds are computed from the target test score distribution.
- `mel_close.py` does not build the embeddings it reads, despite `REPRODUCE.md:41-42` saying it does; a
  clean clone fails at `np.load` unless `imbalanced_recall_mel.py` (which calls
  `imbalanced_recall.py:285 ensure_mel_embeddings`) is run first.
- translate-train requires `%TEMP%\opencode\tt_cache\mt_emb_mel_{de,fr}.npy`; no script in the repo
  builds those caches. If absent, the run silently skips translate-train and the ensemble changes.
- The E9 table mixes at least four protocols: fixed-0.5 raw, tuned-threshold raw, depth8/mcw5/400
  quantile-matched, and depth6/mcw1/200 `mel_close`. Threshold/runtime definitions must be stated per
  row.

---

### E16 (alias) — MultiEURLEX gap-closers

**Script to reproduce:** same script as E9/E10/E11 — `mel_close.py` — `python mel_close.py --launch`.
Reproduction status: **PARTIAL**; `multieurlex_*.py` is **NOT AVAILABLE (UNK)**.

E16 is **not a separate experiment**. `REPORT.md:109` and `REPRODUCE.md:62` call the MultiEURLEX
gap-closer "E16"; `EXPERIMENTS.md` files the same content under E9/E10/E11. The surviving artifact is
`mel_close.*`, and the missing `multieurlex_*.py` scripts are the same missing scripts. Treat E16 as an
alias: **E16 = E9/E10/E11 + the old E16 note**. Do not cite E16 as an independent replication. The note
in the initial commit adds only the 610m prose row and the in-language ceiling (0.711/0.707 vs
0.714/0.712); neither has a surviving artifact or a producing command.

---

## E14/E14b — FLORES English→non-English gap-closers — DONE (audited 2026-10-19)

**Script to reproduce:** `flores_close.py` — `python flores_close.py --launch` (detached hidden full
run; foreground `python flores_close.py`). Reproduction status: **REPRODUCIBLE**. 11 cells re-ran
exactly; the MD was rebuilt byte-identically from the JSON.

**Question.** With one English-trained XGBoost on EuroBERT-210m mean features, which method best closes
the English→it/de/fr gap on FLORES sent (12-class) and agg (11-class), relative to a model trained in
the target language? E14 is the primary (micro-F1) view; **E14b** is the macro-F1 view of the same
artifact. Both IDs share one script, one JSON and one set of tables.

**Script / outputs.**

- `flores_close.py` -> `flores_close.json` (numbers), `flores_close.md` (full tables), plus the
  operational `.log` / `.pid` / `.status.json`.
- `--launch` starts a hidden detached run; `--smoke` runs a 60-tree sent-only plumbing check.
- MT and the translated embeddings are cached under `%TEMP%\opencode\flores_close_cache`.
- The script is **not resumable**. It rewrites `flores_close.json` / `.md` on every run, including
  `--smoke` and `--datasets sent`.

**Inputs.**

| dataset | embeddings | labels / split | model |
|---|---|---|---|
| sent | `embeddings/mean/emb_{en,it,de,fr}.npy`, 2009×768, sliced to 2006 kept rows | `flores200_en_it_de_fr.csv`; 12 classes; 997/504/505; parallel rows | one EN XGB |
| agg | `embeddings_agg/mean/emb_{en,it,de,fr}.npy`, 562×768 | `flores_agg.csv`; class 0 dropped, 11 classes; 276/140/139 | one EN XGB |

Split: train = `dev`; devtest (1009 rows) permuted with seed 0, first half val, second half test. Agg
inherits the sentence split by URL, then drops class 0 from each split. Test rows are parallel across
languages. The target-language model ("in-language reference") trains on target train+val labels; every
transfer method trains on English only, except the MT / alignment variants below.

**Protocol.** Fixed XGBoost for every cell and every reference: `multi:softprob`, hist, cpu, max_bin 64,
n_jobs 2, seed 0, 200 trees, depth 6, lr 0.1, subsample 0.8, colsample 0.4. Features: raw, `l2_thenZ`
(per-language z-score), `pc2`/`pc3` (global PCA zeroed on the stacked 4-language train pool), `inlp` (6
rounds, 18 dims removed on sent / 3 on agg), and two combinations. kNN: cosine over the EN train+val
gallery, k∈{1,3,5,10,20}, majority and similarity-weighted. translate-train: MT EN train+val into the
target (opus-mt-en-{it,de,fr}), embed, train on EN + translated. `affine_moment`: DiTTO-style
target→EN per-dim affine from unlabeled target train stats. `mlp_align`: 1×256 ReLU MLP target→EN on
parallel train+val features, 200 epochs. Ensembles: mean probability of the 3 best feature methods,
optionally + translate-train.

**Metrics.** Primary = **OVR micro-F1** at the global F1-optimal threshold; macro-F1 is reported
separately at its own optimal threshold. Both are multi-label OVR decisions (`class c` positive iff
`p_c >= t`), **not** single-label argmax F1. At t = 0.04 the EN sent model predicts 2.8 classes per row;
EN accuracy is 0.602 while OVR micro-F1 is 0.459.

**Results — mean target OVR micro-F1 (primary metric).**

| method | sent micro (it/de/fr) | sent mean | agg micro (it/de/fr) | agg mean | gap closed sent / agg |
|---|---|---:|---|---:|---:|
| raw | .306/.260/.307 | 0.2908 | .299/.258/.276 | 0.2774 | 0% / 0% |
| `l2_thenZ` | .359/.347/.367 | 0.3580 | .333/.341/.354 | 0.3426 | 51.6% / 42.0% |
| `pc2` | .376/.356/.308 | 0.3466 | .288/.288/.276 | 0.2839 | 42.8% / 4.2% |
| `pc3` | .374/.361/.374 | 0.3699 | .314/.326/.324 | 0.3212 | 60.7% / 28.2% |
| `inlp` | .361/.343/.364 | 0.3558 | .320/.300/.326 | 0.3155 | 49.9% / 24.6% |
| `pc+inlp` | .371/.364/.378 | 0.3707 | .311/.318/.299 | 0.3095 | 61.3% / 20.7% |
| `pc+l2_thenZ` | .364/.351/.368 | 0.3609 | .333/.330/.344 | 0.3357 | 53.7% / 37.5% |
| `inlp+l2_thenZ` | .357/.350/.368 | 0.3582 | .344/.348/.349 | 0.3471 | 51.7% / 44.9% |
| `affine_moment` | .514/.499/.520 | 0.5108 | .547/.581/.539 | 0.5554 | 168.7% / 179.1% |
| translate-train | .567/.557/.576 | 0.5666 | .623/.631/.599 | 0.6175 | 211.5% / 219.2% |
| `mlp_align` | .536/.538/.550 | 0.5411 | .635/.677/.636 | 0.6492 | 191.9% / 239.6% |
| `ens_top3` | .373/.360/.371 | 0.3678 | .331/.333/.344 | 0.3363 | 59.1% / 38.0% |
| `ens_top3+tt` | .509/.496/.533 | 0.5127 | .569/.638/.571 | 0.5929 | 170.1% / 203.3% |
| **kNN (`l2_thenZ`,w10)** | **.604/.602/.599** | **0.6018** | **.674/.674/.681** | **0.6760** | **238.5% / 256.8%** |

In-language micro-F1 reference: sent it/de/fr .4291/.4169/.4175 (mean 0.4212), EN .4594; agg
it/de/fr .4422/.4306/.4250 (mean 0.4326), EN .4495.

**Results — mean target macro-F1 (secondary metric, own optimal threshold).**

| method | sent mean | agg mean |
|---|---:|---:|
| raw | 0.2505 | 0.2624 |
| `l2_thenZ` | 0.3074 | 0.3085 |
| `pc+inlp` | 0.3140 | 0.2793 |
| `affine_moment` | 0.4000 | 0.4792 |
| translate-train | 0.4207 | 0.5225 |
| `mlp_align` | 0.4242 | 0.5690 |
| `ens_top3` | 0.3195 | 0.3042 |
| `ens_top3+tt` | 0.3142 | 0.4319 |
| kNN (`l2_thenZ`,w10) | 0.5112 | 0.6064 |

In-language macro-F1 reference: sent mean 0.3617 (it .3676 / de .3681 / fr .3495); agg mean 0.3936
(it .4090 / de .3916 / fr .3803).

**0.975 bar (method ÷ in-language reference, per target).**

| method | sent micro | agg micro | sent macro | agg macro |
|---|---|---|---|---|
| kNN (`l2_thenZ`,w10) | 1.407/1.445/1.435 **3/3** | 1.524/1.564/1.601 **3/3** | 1.355/1.377/1.513 **3/3** | 1.491/1.516/1.619 **3/3** |
| translate-train | 1.321/1.337/1.379 **3/3** | 1.408/1.465/1.410 **3/3** | 1.119/1.162/1.210 **3/3** | 1.267/1.406/1.311 **3/3** |
| `mlp_align` | 1.249/1.290/1.316 **3/3** | 1.437/1.572/1.496 **3/3** | 1.114/1.179/1.228 **3/3** | 1.385/1.442/1.514 **3/3** |
| `affine_moment` | 1.197/1.197/1.245 **3/3** | 1.237/1.349/1.267 **3/3** | 1.078/0.963/1.285 **2/3** | 1.227/1.269/1.154 **3/3** |
| `ens_top3+tt` | 1.186/1.189/1.277 **3/3** | 1.287/1.482/1.344 **3/3** | 0.848/0.770/0.995 **1/3** | 0.993/1.233/1.069 **3/3** |
| raw, `l2_thenZ`, `pc2`, `pc3`, `inlp`, `pc+inlp`, `pc+l2_thenZ`, `inlp+l2_thenZ`, `ens_top3` | <=0.904 **0/3** | <=0.822 **0/3** | <=0.928 **0/3** | <=0.837 **0/3** |

**Verdict.** At the **0.975** bar on the primary metric, **5 methods pass 3/3 on both datasets**: kNN
(`l2_thenZ`, w10), translate-train, `mlp_align`, `affine_moment`, `ens_top3+tt`. No pure
English-fitted feature transform (raw / l2 / PC / INLP / their ensemble) passes a single cell; the
label-free target→EN affine does. On macro-F1 the same three leaders (kNN, translate-train,
`mlp_align`) pass 6/6 target cells; `affine_moment` fails sent-de (0.963) and `ens_top3+tt` fails two
sent cells. The best cell is kNN on agg (0.676 vs 0.433 reference). kNN is the cleanest
label-cost-matched method (no target labels at all); `affine_moment` uses only unlabeled target train
rows for the transform, but its threshold is still tuned on in-sample val labels.

**Caveats.**

- **Not label-cost matched for the top rows.** translate-train adds MT of the EN train+val text (2×
  training rows, machine-generated labels); `mlp_align` uses parallel target train text; `ens_top3+tt`
  inherits translate-train. kNN uses no extra target-side data or labels. The in-language reference
  uses 1501 target labels. The word "ceiling" is a misnomer: 5 methods exceed it, so it is an
  in-language *reference*, not an upper bound.
- **Threshold protocol is broken.** The code fits on train+val and then tunes the threshold on val,
  which is part of the fit set. Val F1 is 1.0000, so the tuner returns the lowest grid threshold
  (0.04). kNN is the only method whose threshold is tuned out-of-sample. Every XGB row and every
  reference is therefore depressed relative to an honest protocol, and the method families are not
  comparable.
- Model selection: the kNN variant and the ensemble members are chosen by target **test** micro-F1, so
  both rows are optimistic.
- Single seed (0); MLP training is unseeded, so `mlp_align` is not bit-reproducible.
- Test sizes are 505 (sent) and 139 (agg) rows; no CIs.
- The metric is OVR multi-label micro-F1, not single-label argmax F1. Do not compare with the argmax
  macro-F1 in E19/E21 (E14's EN sent argmax macro is 0.4915; its threshold macro is 0.4147).

---

## E4 — Sentiment / emotion cross-lingual — UNVERIFIABLE (audited 2026-10-19)

**Script to reproduce:** NOT AVAILABLE — `crosslingual_sentiment.py` never existed in git; no result
artifact survives. **(UNK)**

**Question.** Does an English-trained sentiment/emotion classifier keep its macro-F1 when it is applied
to machine-translated Italian, German, and French, and does a centroid neutralization close the gap?

**Script / outputs.**

- `crosslingual_sentiment.py` is cited at `EXPERIMENTS.md:211`. It was never tracked: 98 commits, no
  path, no blob, no deleted path.
- No `crosslingual_sentiment.json`, `.md`, `.png`, log, PID, or status file survives. A search of
  `C:\Users\pedro` and `%TEMP%\opencode` for `crosslingual*` returns nothing.
- The script existed on disk before the 2026-10-11 cleanup. Temp diagnostics from 2026-10-10
  18:44–19:05 import it and read its JSON:
  - `cl_sent_sanity.py:3` — `import crosslingual_sentiment as C`.
  - `cl_sent_embedtest.py:2` — same import; `:8` calls `C.load_translated_texts(...)`.
  - `chk_c.py:50` — `json.load(... "crosslingual_sentiment.json")`.
  - `chk_a.py:30` — quotes `JSON full/en en_threshold macro_f1 = 0.7915994045485419`.
  - `chk_a.py:41` — quotes `JSON full/it target_threshold macro_f1 = 0.5962806222431762`.

**Claims (UNVERIFIED).** All values are macro-F1, copied from `EXPERIMENTS.md:214-217`. Only two cells
have an exact 4-dp value quoted in a surviving temp diagnostic; the rest exist only at 3 dp.

| task | config | it | de | fr | EN-self | status |
|---|---|---:|---:|---:|---:|---|
| SST-2 (binary) | raw | 0.596 | 0.593 | 0.636 | 0.792 | UNVERIFIED |
| SST-2 (binary) | centroid | 0.630 | 0.627 | 0.649 | 0.792 | UNVERIFIED |
| emotion (6-class) | raw | 0.170 | 0.118 | 0.174 | 0.370 | UNVERIFIED |
| emotion (6-class) | centroid | 0.203 | 0.214 | 0.219 | 0.370 | UNVERIFIED |

Provenance notes: SST-2 EN 0.792 = 0.7915994 (quoted `chk_a.py:30`); SST-2 it raw 0.596 = 0.5962806
(quoted `chk_a.py:41`). The quoted JSON is gone; the quotes are not independent verification. The other
12 cells have no 4-dp source. The entry does not state pooling, classifier, hyperparameters, split,
seed, or threshold protocol; the old log says "SST-2 (mean)". The EN references are not the repo's
tuned XGBoost results: `README.md:269,276` reports test macro-F1 0.8052 (SST-2 mean) and 0.3740
(emotion mean); E4 reports 0.792 and 0.370. The gap is unexplained.

**Inputs that survive.** All files are git-ignored (`.gitignore:69-70`); they exist only as local
derived caches, checked 2026-10-19.

| path | files | rows / shape | notes |
|---|---|---:|---|
| `translated/sst2_{it,de,fr}_{validation,test}.csv` | 6 | 6,822 rows each | columns `text,label`; row order and labels match `sst2.csv` split |
| `translated/emotion_{it,de,fr}_{validation,test}.csv` | 6 | 2,000 rows each | columns `text,label`; labels match `emotion.csv` split |
| `translated_emb/sst2_{mean,bos}_{it,de,fr}_{validation,test}.npy` | 12 | 6,822 × 768 float32 | rows match the CSV |
| `translated_emb/emotion_{mean,bos}_{it,de,fr}_{validation,test}.npy` | 12 | 2,000 × 768 float32 | rows match the CSV |
| `sst2_emb/{mean,bos}_{train,val,test}.npy` | 6 | 54,577 / 6,822 / 6,822 × 768 | English source cache |
| `emotion_emb/{mean,bos}_{train,val,test}.npy` | 6 | 16,000 / 2,000 / 2,000 × 768 | English source cache |

There is **no translated train split**. `rebuild_caches.py:336-340` translates validation and test only.
Translate-train cannot be run from these caches.

Data defects: `translated/sst2_it_test.csv` has 3 rows with empty `text`; `..._it_validation.csv` has
2. They were embedded as empty strings (pairwise cosine = 1.0) and are evaluated with their source
labels. Duplicate translated texts exist: 19–33 per SST-2 split; up to 4 text groups in one split carry
conflicting labels. Emotion has 0–2 duplicates. The CSVs are valid UTF-8.

**Can the surviving data support a new reproduction? Partly.**

- A new script can reproduce the **raw** arm: load `sst2_emb/mean_train.npy`, train an XGBoost, score
  `translated_emb/sst2_mean_{lang}_test.npy`, tune a threshold on the translated validation split,
  report test macro-F1. The same path works for emotion with argmax.
- The **centroid** arm cannot be reproduced as documented. The transform is lost. `lang_neutralize.py`
  has `centroids()` (`:38-40`) and `center_by_lang()` (`:64-65`), but no surviving script calls either.
- No surviving script consumes `translated_emb/*.npy`. `xgboost_sst2.py` and `xgboost_emotion.py`
  evaluate English only. `en_vs_enit_translate.py:225` and `translate_test_eval.py:243` use a different
  MT-to-English cache.
- `rebuild_caches.py` can regenerate the 12 CSVs and 24 `.npy` files with the same names and the
  documented recipe (`:13-27`, `:324-353`). It needs a GPU, the HF models, and several hours.

**Protocol inferred from temp diagnostics (not verified).**

| item | inferred value | evidence |
|---|---|---|
| encoder | EuroBERT-210m, mean pooling | `chk_a.py:21-23`; `21ab888` log |
| SST-2 model | XGBoost binary, depth 6, lr 0.1, 400 trees, hist, max_bin 64, seed 0 | `chk_a.py:24` |
| emotion model | XGBoost `multi:softprob`, 6 classes | `chk_c.py:52-62` |
| SST-2 threshold | grid 0.05–0.95 step 0.01, tuned on validation | `chk_a.py:9-15,27` |
| EN reference | EN-validation threshold, EN-test score | `chk_a.py:26-27` |
| target score | target-validation threshold, target-test score | `chk_a.py:34-41` |
| conditions | `full` (raw), `centroid`, `zero90`, `subspace`, `abtt` | `chk_c.py:57-61` |

`chk_a.py` tunes the SST-2 threshold by **accuracy** (`:13`), not macro-F1; the original objective is
not recoverable. If E4 used the same rule, the table is a per-target threshold oracle.

**At the 0.975 bar.** The bar is E18's criterion (AUC); E4 reports macro-F1. The comparison is a
cross-experiment consistency check, not a criterion E4 states, and it uses the unverified numbers.

| task | config | best lang | target | EN-self | ratio | short of 0.975 |
|---|---|---|---:|---:|---:|---:|
| SST-2 | raw | fr | 0.636 | 0.792 | 0.8030 | 0.1720 |
| SST-2 | centroid | fr | 0.649 | 0.792 | 0.8194 | 0.1556 |
| emotion | raw | fr | 0.174 | 0.370 | 0.4703 | 0.5047 |
| emotion | centroid | fr | 0.219 | 0.370 | 0.5919 | 0.3831 |

**Result.** **0/12 target cells reach 0.975.** The best cell is SST-2 fr centroid at **0.819**; it still
needs **+0.123 macro-F1** to pass. The best emotion cell is fr centroid at **0.592**; it needs +0.142.
Raw SST-2 is 0.749–0.803 (ratios 0.749–0.803). Every ratio is far below the bar.

**Verdict.** The E4 numbers are a single, unverified run. No script and no result artifact survive. The
surviving caches can support a new **partial** reproduction of the raw arm; they cannot reproduce the
centroid arm, because its definition was lost. The reported numbers fail the 0.975 bar by a wide margin
in every cell.

**Caveats.**

- The ratios rest entirely on unverified 3-dp claims.
- Single run, no seed record, no CIs. Test sets are 6,822 (SST-2) and 2,000 (emotion); a 1–2 pp
  difference is not meaningful.
- Per-target thresholds (if used) need target labels. They make the cross-lingual cells optimistic
  compared with a fixed 0.5 threshold, the deployable setting.
- The current caches were rebuilt on 2026-10-11/12, after the reported run, and are not proven to be
  the exact arrays used.
- `0.975 × EN` is a ratio on macro-F1, not AUC. The absolute bars are only 0.772 (SST-2) and 0.361
  (emotion); neither is reached.

---

## E2 — Translated SST-2 / emotion eval sets — PARTIAL (verified 2026-10-20)

**Script to reproduce:** `rebuild_caches.py` — `python rebuild_caches.py` (REPRODUCE.md line 66);
detached form `wscript //B //Nologo launch_hidden.vbs "<python> -u rebuild_caches.py"` (script
docstring, lines 34-35). Reproduction status: **PARTIAL**; `translate_datasets.py` is **NOT AVAILABLE
(UNK)**.

**Question.** Do the translated sentiment/emotion eval sets exist, are they complete and correctly
built, and do they cover the E4 "English -> translated it/de/fr" evaluation?

**Script / outputs.**

- `EXPERIMENTS.md:383` names two scripts and gives no command: `translate_datasets.py` (missing) and
  `rebuild_caches.py` (present, tracked, committed in `21ab888`, unchanged since).
- REPRODUCE.md line 66: `python rebuild_caches.py   # rebuild SST-2/emotion embeddings + translated text (optional)`.
- The script is create-only and resumable: every target is skipped when it already exists, so a rebuild
  requires deleting the target files first. It never deletes or overwrites data.

**Artifacts (measured on disk, 2026-10-20).**

| group | files | content |
|---|---:|---|
| `translated/` | 12 | `{sst2,emotion}_{it,de,fr}_{validation,test}.csv`, columns `text,label` |
| `translated_emb/` | 24 | `{dataset}_{mean,bos}_{lang}_{validation,test}.npy` |
| `sst2_emb/`, `emotion_emb/` (part 1 of the same script) | 12 | `{mean,bos}_{train,val,test}.npy`, the English side E4 needs |

| dataset | lang | MT model | validation rows | test rows | labels | blank translations |
|---|---|---|---|---:|---|---|
| sst2 | it | `Helsinki-NLP/opus-mt-en-it` | 6,822 | 6,822 | 0/1, identical to EN row-wise | 2 val / 3 test |
| sst2 | de | `Helsinki-NLP/opus-mt-en-de` | 6,822 | 6,822 | 0/1, identical to EN row-wise | 0 |
| sst2 | fr | `Helsinki-NLP/opus-mt-en-fr` | 6,822 | 6,822 | 0/1, identical to EN row-wise | 0 |
| emotion | it | `Helsinki-NLP/opus-mt-en-it` | 2,000 | 2,000 | 6 classes, identical to EN row-wise | 0 |
| emotion | de | `Helsinki-NLP/opus-mt-en-de` | 2,000 | 2,000 | 6 classes, identical to EN row-wise | 0 |
| emotion | fr | `Helsinki-NLP/opus-mt-en-fr` | 2,000 | 2,000 | 6 classes, identical to EN row-wise | 0 |

Totals: 36 data files, 314.0 MB (310.2 MB embeddings + 3.8 MB CSVs). No `train` translations exist.

**Protocol actually used (recovered from code and confirmed against the artifacts).**

- MT: Marian `opus-mt-en-{it,de,fr}`, greedy (`num_beams=1`, `do_sample=False`), batch 64, `max_length=512`,
  fp16 on CUDA, `torch.set_num_threads(2)` (lines 234-263, 269-340).
- Embedding: EuroBERT-210m native model, fp32, batch 256, max_len 512, mean = masked mean over non-pad
  tokens of the plain forward; bos = position 0 after manually prepending the tokenizer's BOS id 128000
  (`<|begin_of_text|>`, verified) (lines 183-214).

**Verification performed.**

1. **MT provenance, exact.** Re-decoding individual rows reproduces the stored text character for
   character for it/de/fr on both datasets (10/10 rows), including the bad ones. The 5 blank Italian
   rows reproduce as `""` only when the original 64-row batch boundaries are used.
2. **Embeddings, exact to float noise.** Re-embedding 10 rows gives `max|delta| <= 4.6e-4`, cosine
   >= 0.99999994 against the stored vectors.
3. **Cache validity.** Loading every artifact through the script's own `cached_npy` / `csv_is_complete`
   gives 48/48 valid with matching row counts.
4. **Row alignment.** For all 12 CSVs, row count and every label equal the English source split in
   order. The English data are the repo's own re-splits (`sst2.csv` 54,577/6,822/6,822; `emotion.csv`
   16,000/2,000/2,000), not the official GLUE SST-2 test.
5. **No zero or non-finite vectors.** Row norms 49.6–120.1; no zero rows; mean and bos differ.

**Coverage of the E4 use case — yes, data-wise.** E4 needs EN train embeddings, EN test embeddings, and
translated it/de/fr embeddings with labels. All exist and align. A light re-run of the E4 protocol
(XGBoost, 200 trees, depth 6, mean pooling, threshold 0.5; a sanity check, not the missing E4 script)
lands on the entry's numbers:

| task | metric | EN | MT it | MT de | MT fr |
|---|---|---:|---:|---:|---:|
| SST-2 | macro-F1 (E4 entry) | 0.792 | 0.596 | 0.593 | 0.636 |
| SST-2 | macro-F1 (audit re-run) | 0.775 | 0.576 | 0.608 | 0.622 |
| emotion | macro-F1 (E4 entry) | 0.370 | 0.170 | 0.118 | 0.174 |
| emotion | macro-F1 (audit re-run) | 0.367 | 0.169 | 0.114 | 0.184 |

The audit re-run is within 0.02 of every E4 cell, so the translated artifacts are the data E4 used.

**0.975 bar: N/A.** E2 is data preparation; it produces no model metric and no target/English ratio.

**Verdict.** The E2 artifacts are real, complete for validation+test, internally consistent, and
sufficient for the E4 evaluation. E2 is nevertheless **PARTIAL** as documentation: the first cited
script does not exist, the E4 consumer script does not exist, no script in the repo reads `translated/`
or `translated_emb/`, and a full clean rebuild was not run in this audit (it needs ~18 min of GPU and
deletion of 310 MB of artifacts).

**Caveats.**

- MT quality is weak by construction: greedy base Marian on sentence fragments and informal text. The
  emotion source rows are already degraded (lowercase, apostrophes stripped), and the German/Italian
  outputs are literal. Part of E4's emotion collapse is therefore MT-induced and cannot be separated
  without a stronger-MT baseline.
- 5 Italian rows have no translation (the model emits EOS immediately on quote-only inputs); they carry
  an identical, signal-free embedding.
- Decode is batch-composition dependent on those degenerate inputs, so the CSVs are tied to
  `MT_BATCH=64`.
- Embeddings are not bitwise reproducible (~1e-4 absolute, cosine 1.0); duplicate target texts can have
  differing vectors at that scale.
- All artifacts are git-ignored and restorable only by re-running the script.

---

## E33 — FLORES-agg precision/recall vs threshold, 210m vs 610m (Optuna) — DONE, PARTIALLY REPRODUCIBLE (audited 2026-10-20)

**Script to reproduce:** NOT AVAILABLE for a full rerun — `agg_pr_2models.py` imports the missing
`agg_pr_methods`; only `--reemit-from-json` works. **(UNK for a full rerun.)**
`python agg_pr_2models.py --reemit-from-json` regenerates figures + md/json only.

**Question.** For the 11-class FLORES URL-aggregate task, what do the micro precision/recall and
F1-versus-threshold curves look like for an English-trained XGBoost evaluated on English and on
Italian, and does the 610m backbone move Italian closer to English?

**Script / outputs.**

- `agg_pr_2models.py` (39 KB, mtime 2026-10-11 20:48:57) -> `agg_pr2.json` (3,060 curve points +
  tuning records, 111 KB), `agg_pr2.md` (auto-generated), and 8 PNGs.
- 8 PNGs: `agg_pr2_{raw,pc,inlp,pc+inlp,l2_thenZ}.png` (per-method 2×2: rows = Precision/Recall,
  columns = 210m / 610m) plus `agg_pr2_grid_{precision,recall}.png` and `agg_pr2_grid.png`.
- Modes: default (full, 20 Optuna trials per cell), `--smoke` (2 trials), `--resume` (per-cell
  checkpoint in `%TEMP%`), `--recompute-inlp`, `--reemit-from-json` (only mode that runs today).
- **FULL RERUN NOT POSSIBLE** — the script dies at line 746 with
  `ModuleNotFoundError: No module named 'agg_pr_methods'` before any data is loaded.
  **RE-EMIT REPRODUCIBLE** — `--reemit-from-json` on a copy reproduced `agg_pr2.md`, `agg_pr2.json`
  and two sampled PNGs byte-identically (SHA-256 match, 2.9 s).

**Inputs.**

| item | file | shape / content |
|---|---|---|
| labels + splits | `flores_agg.csv` (562 rows) + `flores200_en_it_de_fr.csv` (URL -> split) | 12 topics, 7 `business and economy` rows dropped -> 11 classes, 555 docs |
| 210m embeddings | `embeddings_agg/mean/emb_{en,it,de,fr}.npy` | 562x768 float32 (row-aligned to CSV) |
| 610m embeddings | `embeddings_agg_610/mean/emb_{en,it,de,fr}.npy` | 562x1152 float32 |
| 210m INLP basis | `lang_inlp_agg.npz` | **MISSING** (md claims 3 directions) |
| 610m INLP basis | `lang_inlp_agg_610.npz` | **MISSING** (JSON meta: 3 directions, 1 round) |
| PC ablation | `pc_ablation_flores.py` | **MISSING** |

Class list (11): crime and law, education and society, entertainment and culture, geography, health,
history, politics, safety, science and technology, sports, travel. Split = URL-level: `dev` URLs ->
train (276 after exclusion), `devtest` URLs shuffled with `default_rng(0)` and halved -> val 140 /
test 139. Test rows are parallel across EN/IT (same `test_idx`), so every EN/IT comparison is paired.

**Methods (5 per backbone).**

| method | transform | extra data |
|---|---|---|
| `raw` | none | none |
| `pc` | global iterative PC ablation k=2, fit on the stacked-language TRAIN pool | train only |
| `inlp` | project out the backbone's language-subspace basis | full 562-row pool per language (label-free) |
| `pc+inlp` | pc (k=2) then inlp | as above |
| `l2_thenZ` | L2-normalize, then per-language TRAIN z-score | unlabeled target train rows |

**Protocol.** Optuna `TPESampler(seed=0)`, **20 trials** per (backbone, method) = 200 trials,
`n_trials_failed=0` in all 10 cells. Search space mirrors `xgboost_topic_optuna._optuna_search_space`
exactly (`xgboost_topic_optuna.py:946-957`) with `n_estimators` capped at 500 instead of 800. Base
params: softprob, 11 classes, hist, max_bin 64, cpu, `n_jobs=2`, seed 0. Each trial fits EN train (276)
and scores EN val (140) by OVR-macro AUC. Best params are retrained on EN train+val (416) and scored on
EN test and IT test over `t = 0.00..1.00` step 0.02 (51 points); class c is positive iff `p_c >= t`;
micro P/R/F1 pooled over the 139×11 = 1,529 (document, class) pairs. F1-optimal threshold is selected
**per language on the test split** (see caveats).

**Tuning result per cell (stored in `agg_pr2.json["tuning"]`).**

| backbone | method | val AUC | val macro-F1 | best params |
|---|---|---:|---:|---|
| 210m | raw | 0.8935 | 0.5268 | d6 lr0.024 n286 ss0.70 cs0.21 mcw2 rl0.422 g0.69 |
| 210m | pc | 0.8905 | 0.4439 | d7 lr0.012 n249 ss0.71 cs0.12 mcw4 rl0.775 g2.17 |
| 210m | inlp | 0.8864 | 0.4282 | d7 lr0.010 n223 ss0.71 cs0.11 mcw4 rl0.585 g1.65 |
| 210m | pc+inlp | 0.8865 | 0.4301 | d7 lr0.010 n223 ss0.71 cs0.11 mcw4 rl0.585 g1.65 |
| 210m | l2_thenZ | 0.8930 | 0.4174 | d7 lr0.012 n249 ss0.71 cs0.12 mcw4 rl0.775 g2.17 |
| 610m | raw | 0.9033 | 0.4939 | d6 lr0.024 n286 ss0.70 cs0.21 mcw2 rl0.422 g0.69 |
| 610m | pc | 0.9039 | 0.5715 | d8 lr0.078 n396 ss0.62 cs0.30 mcw2 rl0.0153 g0.59 |
| 610m | inlp | 0.9058 | 0.4944 | d6 lr0.024 n286 ss0.70 cs0.21 mcw2 rl0.422 g0.69 |
| 610m | pc+inlp | 0.9044 | 0.4775 | d7 lr0.024 n196 ss0.89 cs0.26 mcw1 rl1.53 g0.08 |
| 610m | l2_thenZ | 0.8987 | 0.5827 | d8 lr0.078 n396 ss0.62 cs0.30 mcw2 rl0.0153 g0.59 |

Only **5 distinct parameter vectors** cover the 10 cells; `raw` (both backbones) and 610m `inlp` share
one vector, and 610m `pc` = 610m `l2_thenZ`.

**Test results at each language's own F1-optimal threshold.**

| backbone | method | t_EN | EN P | EN R | EN F1 | t_IT | IT P | IT R | IT F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 210m | raw | 0.28 | 0.6794 | 0.6403 | **0.6593** | 0.20 | 0.4269 | 0.5252 | 0.4710 |
| 210m | pc | 0.22 | 0.6720 | 0.6043 | 0.6364 | 0.22 | 0.4706 | 0.4029 | 0.4341 |
| 210m | inlp | 0.20 | 0.6176 | 0.6043 | 0.6109 | 0.22 | 0.4603 | 0.4173 | 0.4377 |
| 210m | pc+inlp | 0.22 | 0.6583 | 0.5683 | 0.6100 | 0.26 | 0.6351 | 0.3381 | 0.4413 |
| 210m | l2_thenZ | 0.20 | 0.6312 | 0.6403 | 0.6357 | 0.22 | 0.5556 | 0.4676 | **0.5078** |
| 610m | raw | 0.26 | 0.6519 | 0.6331 | 0.6423 | 0.20 | 0.5200 | 0.5612 | 0.5398 |
| 610m | pc | 0.28 | 0.6462 | 0.6043 | 0.6245 | 0.20 | 0.4557 | 0.5180 | 0.4848 |
| 610m | inlp | 0.26 | 0.6617 | 0.6331 | **0.6471** | 0.28 | 0.5546 | 0.4748 | 0.5116 |
| 610m | pc+inlp | 0.22 | 0.5570 | 0.6331 | 0.5926 | 0.24 | 0.5000 | 0.5108 | 0.5053 |
| 610m | l2_thenZ | 0.26 | 0.6250 | 0.6475 | 0.6360 | 0.26 | 0.6241 | 0.5971 | **0.6103** |

**EN-IT gap (EN peak F1 minus IT peak F1).**

| method | 210m dF1 | 610m dF1 | d(gap) | IT/EN ratio 210m | IT/EN ratio 610m |
|---|---:|---:|---:|---:|---:|
| raw | +0.1883 | +0.1025 | -0.0857 | 0.7144 | 0.8404 |
| pc | +0.2023 | +0.1397 | -0.0626 | 0.6822 | 0.7763 |
| inlp | +0.1732 | +0.1354 | -0.0377 | 0.7165 | 0.7907 |
| pc+inlp | +0.1687 | +0.0873 | -0.0815 | 0.7234 | 0.8528 |
| l2_thenZ | +0.1279 | +0.0257 | -0.1022 | 0.7988 | **0.9595** |
| mean | +0.1721 | +0.0981 | -0.0739 | | |

The last two columns are **derived by the audit** (the script stores no ratio); they are the closest
thing to an E18-style target/English ratio in this experiment.

**0.975 bar: N/A.** This is a threshold sweep, not a ratio benchmark; `agg_pr2.json` and `agg_pr2.md`
contain no target/English ratio and no `0.975` target anywhere. Read as a ratio anyway, the derived
IT/EN peak-F1 ratios top out at **0.9595** (610m `l2_thenZ`); **0/10 cells reach 0.975**.

**Verdict.** 610m improves Italian in every method (5/5) and narrows the mean EN-IT peak-F1 gap from
+0.1721 to +0.0981. Best Italian transfer is `l2_thenZ` on both backbones (0.5078 -> 0.6103, +0.1025).
`pc`/`inlp` alone never beat `raw` on Italian F1 at either backbone; only the combined per-language
standardization (`l2_thenZ`) does. The AUC-optimal method and the Italian-F1-optimal method differ at
both backbones (210m: val AUC `raw` vs best IT `l2_thenZ`; 610m: val AUC `inlp` vs best IT `l2_thenZ`),
so the tuning objective does not predict transfer.

**Caveats.**

- **F1-optimal thresholds are picked on the test split** (per language, argmax over 51 grid points on
  139 documents). Every P/R/F1 above is an oracle-threshold number and is optimistically biased; the
  EN-IT gap and the "best method" ranking inherit it.
- Test is 139 documents / 1,529 pairs, one split, one seed, no CIs. Audit re-fits with `n_jobs=1` or
  `seed=1` moved IT F1 by 0.002–0.02 and EN F1 by up to 0.02, so the 4-dp values overstate precision by
  roughly an order of magnitude.
- 610m INLP removes **3 of 1,152 dimensions (0.26%)** after one round; 210m removes 3 of 768 (0.39%).
  "inlp ~ raw" at 610m is mechanically expected, not an empirical result.
- Both INLP bases were fitted on the full 562-row pool per language (train+val+test features, language
  labels only) — transductive but label-free.
- Precision is stored as 0.0 when no pair clears the threshold (TP+FP = 0), which draws a 1.0 -> 0
  cliff at `t > max(p)`; IT recall hits 0 at t = 0.24 (210m) / 0.46 (210m raw), so the right-hand part
  of every Italian panel is a flat zero line.
- The full numbers cannot be regenerated: `agg_pr_methods.py`, `pc_ablation_flores.py`,
  `crosslingual_topic_agg.py`, `agg_pr_lang.py`, `lang_inlp_agg.npz` and `lang_inlp_agg_610.npz` are
  all absent from `multi/`.

---

## E34 — Full-document chunk pooling vs 512-token truncation on MultiEURLEX — DONE (verified 2026-10-20)

**Script to reproduce:** `chunk_pool_mel.py` — `python chunk_pool_mel.py`. Reproduction status:
**PARTIAL** (training / metric stage re-ran bit-exact for 2 of 4 schemes; the embedding stage was not
re-executed).

**Question.** The MultiEURLEX pipeline keeps only the first 512 tokens of every document. The documents
are long: train means 1431 (en) / 2230 (de) / 2109 (fr) tokens, test means 2404 / 3776 / 3637, the test
p99 is 29k–47k, and the longest is 390,984 tokens (1527 chunks). Does embedding ALL chunks and pooling
them into one document vector (a) beat first-512 truncation, and (b) close the EN->target gap against an
in-language ceiling?

**Script / outputs.**

- `chunk_pool_mel.py` -> `chunk_pool_mel.json` (numbers + `chunk_stats`), `chunk_pool_mel.md` (3 tables).
- Feature caches: `multieurlex_chunk/<lang>_<split>_{trunc,mean,max,concat}.npy` (9 cells × 4 schemes,
  773 MB), `multieurlex_chunk/stats.json`.
- Documented smoke: `python chunk_pool_mel.py --limit 200 --tag _smoke --n-estimators 60` ->
  `chunk_pool_mel_smoke.json/.md` (the stored smoke used `--limit 150`, not 200).
- `--skip-train` stops after embedding. `--out-json/--out-md/--status/--n-estimators/--tag/--limit`
  override paths and sizes. There is no `--force-train` flag, although the docstring advertises one.
- Resumable at the embedding level only: every `.npy` cache hit is reused, but the whole XGBoost stage
  (24 model fits, 2.5 h) always re-runs.

**Inputs.**

| item | file | shape / size | note |
|---|---|---|---|
| text + labels | `multieurlex/{en,de,fr}_{train,validation,test}.jsonl`, key `multi_hot` | 11000 / 1000 / 5000 per language, 21 labels | official MultiEURLEX split, cached jsonl |
| features | `multieurlex_chunk/<lang>_<split>_{trunc,mean,max,concat}.npy` | 768 / 768 / 768 / 1536 | `trunc` is scheme (0) = chunk 0 |
| split used | train 11000, validation 1000, test 5000 | — | **no 10000/1000 re-split**: this is not the E18 protocol |

Test rows are parallel across en/de/fr (identical 21-vector labels on all 5000 rows; per-document word
counts correlate at r = 0.9993). Train rows are NOT parallel (label cells differ on ~48k of 231k
positions; word-count r ~ 0.00–0.01). Validation rows are also not parallel.

**Protocol.**

- Backbone EuroBERT-210m, fp32, batch 64, GPU; masked mean over each window. Window = 512 raw token
  ids, stride 256. A document of n > 512 ids yields `1 + ceil((n-512)/256)` windows. Documents of
  <= 512 ids are a single window. Only the final window of a long document contains the end token.
- Schemes: `trunc` = chunk 0 only (baseline); `mean` = mean of chunk-means (768-d); `max` = elementwise
  max of chunk-means (768-d); `concat` = concat(mean, max) (1536-d).
- Transform `l2_thenZ [both]`: L2-normalize, then z-score with per-language TRAIN statistics (unlabeled
  target-train rows are used; target labels never are).
- Model: 21 one-vs-rest `XGBClassifier(binary:logistic, hist, max_bin 64, cpu, n_jobs 2, seed 0, lr 0.1,
  depth 8, min_child_weight 5, reg_lambda 1, 400 trees)`, fit on EN train only.
- Calibration `quantile_match_enval`: per concept, the target-test threshold whose predicted-positive
  rate equals the EN-validation TRUE positive rate. `global_thr` (one EN-val macro-F1-optimal
  threshold) is also reported. AUC is threshold-free.
- Ceilings: DE->DE and FR->FR models with the same recipe on their own train split.
- `gap = ceiling - EN->target`; `closure = gap(trunc) - gap(scheme)`, computed per scheme with that
  scheme's own ceiling.

**Results (quantile_match_enval).**

| scheme | target | EN-> micro | EN-> macro | EN-> auc | ceil micro | ceil macro | ceil auc | gap macro | gap auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trunc | de | 0.6855 | 0.5325 | 0.8607 | 0.7354 | 0.5834 | 0.8993 | 0.0509 | 0.0386 |
| trunc | fr | 0.6852 | 0.5325 | 0.8743 | 0.7275 | 0.5826 | 0.9007 | 0.0501 | 0.0264 |
| mean | de | 0.6843 | 0.5350 | 0.8610 | 0.7306 | 0.5800 | 0.9044 | 0.0450 | 0.0434 |
| mean | fr | 0.6961 | 0.5363 | 0.8651 | 0.7279 | 0.5854 | 0.9018 | 0.0491 | 0.0367 |
| max | de | 0.6665 | 0.5187 | 0.8473 | 0.7279 | 0.5744 | 0.8962 | 0.0557 | 0.0489 |
| max | fr | 0.6749 | 0.5271 | 0.8601 | 0.7166 | 0.5737 | 0.8980 | 0.0466 | 0.0379 |
| concat | de | 0.6846 | 0.5347 | 0.8611 | 0.7386 | 0.5896 | 0.9016 | 0.0549 | 0.0405 |
| concat | fr | 0.6902 | 0.5320 | 0.8707 | 0.7237 | 0.5823 | 0.9025 | 0.0503 | 0.0318 |

| scheme | EN->en micro | EN->en macro | EN->en auc | en-val global thr | en-val macro@thr |
|---|---:|---:|---:|---:|---:|
| trunc | 0.7433 | 0.5986 | 0.9095 | 0.10127 | 0.6393 |
| mean | 0.7500 | 0.6047 | 0.9059 | 0.12500 | 0.6333 |
| max | 0.7399 | 0.5864 | 0.9025 | 0.10254 | 0.6212 |
| concat | 0.7385 | 0.5944 | 0.9034 | 0.11312 | 0.6280 |

**Gap closure** (positive = pooling closes part of the truncation-era gap; % of the truncation gap in
brackets).

| target | scheme | micro | macro | auc |
|---|---|---|---:|---:|---:|
| de | mean | +0.0036 (7.2%) | +0.0059 (11.6%) | -0.0048 (-12.4%) |
| de | max | -0.0115 (-23.0%) | -0.0048 (-9.4%) | -0.0103 (-26.7%) |
| de | concat | -0.0041 (-8.2%) | -0.0040 (-7.9%) | -0.0019 (-4.9%) |
| fr | mean | +0.0105 (24.8%) | +0.0010 (2.0%) | -0.0103 (-39.0%) |
| fr | max | +0.0006 (1.4%) | +0.0035 (7.0%) | -0.0115 (-43.6%) |
| fr | concat | +0.0088 (20.8%) | -0.0002 (-0.4%) | -0.0054 (-20.5%) |

**Verdict.** Full-document pooling does **not** close the EN->target gap.

- `mean` is the only scheme with a positive macro-F1 closure on both targets: +0.6 pp on de, +0.1 pp on
  fr, and +1.1 pp micro on fr. `max` is below the truncation baseline on all six F1 cells and on both
  AUC cells; `concat` stays within ±0.5 pp of the baseline.
- AUC closure is **negative for all three schemes on both targets** (-0.0019 to -0.0115). The ranking
  signal that crosses the language boundary does not improve.
- The pooled representation really does differ: for documents > 512 tokens (94% of de/fr test, 68% of
  en test) the mean-pooled vector has mean cosine 0.960 (en) / 0.970 (de) / 0.971 (fr) with chunk 0, and
  2–8% of documents drop below 0.9. The change is large; the metric move is not.
- Restricting the evaluation to documents with > 512 tokens does not rescue it: EN->de micro 0.6855 ->
  0.6862 (AUC 0.8618 -> 0.8648), EN->fr 0.6867 -> 0.6986 (AUC 0.8753 -> 0.8694), EN 0.6997 -> 0.7077
  (AUC 0.8965 -> 0.8950). Still noise-level, and AUC moves both ways.

**The 0.975 bar.** The bar is an EN-parity bar, so it applies to the ratio target/EN-self, not to a
"chunk vs truncation" ratio. On that reading, **0 of 24 cells pass** (4 schemes × 2 targets × 3
metrics); the best cell is `concat` FR AUC at **0.9638**. If the bar is read as "pooling is at least
0.975 × truncation", it is meaningless here: every metric ratio is 0.972–1.016, i.e. inside the seed
noise of a single run. The chunk/truncation *context* ratio is the only well-defined large number:
pooling sees 4.7× (en test), 7.4× (de test) and 7.1× (fr test) the tokens of the baseline (8.87 / 14.26
/ 13.73 chunks per document), and it still buys at most 1 pp.

**Caveats.**

- Single seed, no CIs, no repeated splits. The largest scheme-to-scheme F1 difference is 1.1 pp and the
  largest AUC difference is 1.3 pp, so the ranking of the schemes is not stable.
- The truncation cost is language-asymmetric. Only 32.3% of EN test documents fit in 512 tokens, versus
  5.9% (de) and 6.8% (fr). 94.5% of EN test tokens fall beyond chunk 1, versus 99.3% (de) and 99.2%
  (fr). The design therefore gives the target languages more to gain, and they still gain nothing.
- `mean` is a mean of overlapping window means. The 256-token overlap bands are counted twice, so the
  pooled vector is not a token-level mean of the document.
- Windows 1..k-1 contain no end token (only the final window carries EOS). For documents > 512 tokens,
  the `trunc` baseline is a 512-token window with no EOS, which is not the E18 `emb_mean_*` array (cos
  mean 0.9996, cos min 0.984).
- `concat` changes the dimension (1536) under an unchanged tree budget (depth 8, 400 trees, no tuning),
  so part of its deficit is capacity, not pooling.
- `gap` is computed with a ceiling measured in the *same* representation, so closure mixes ceiling
  movement with transfer movement.
- The validation split is not parallel across languages. `quantile_match_enval` applies EN-val
  per-concept prevalence (mean 0.1754) to a test set whose true prevalence is 0.1711; per-concept error
  reaches 7 pp (7 of 21 concepts deviate > 20%). This shifts absolute F1 for every language, but not the
  between-scheme comparison.
- Numbers are not comparable with E18's `mel` rows: E18 re-splits train to 10000/1000 and quotes
  `raw`/`perlang_z`; E34 uses 11000 train + the official validation and `l2_thenZ [both]`.
- The 5.6 h run is 3.0 h embedding (9 cells, 441,693 chunks) and 2.5 h training (290–295 s per 768-d
  fit, 640–645 s per 1536-d fit).
- Re-verified 2026-10-20: schemes `trunc` and `mean` reproduce all stored cells to 4 dp from the cached
  features. The stored smoke outputs (`chunk_pool_mel_smoke.*`) do **not** reproduce; do not quote them.

---

## E35 — EN-trained vs IT-trained error buckets on FLORES, plus a general error lister — DONE (audited 2026-10-20)

**Script to reproduce:** `it_errors.py` / `list_errors.py` — `python it_errors.py`; `python
list_errors.py --dataset sent` (and `--dataset agg`). Reproduction status: **REPRODUCIBLE**.

**Question.** On the *same* Italian test rows, how much of the EN-trained model's error is language
transfer, and how much is items that no language gets right? Can the same item-level analysis be
produced for any (source, target) pair with text, human gold, MT and probabilities?

**Script / outputs.**

- `it_errors.py` -> `it_errors.json`, `it_errors.md` (fixed pair en->it, both datasets, one run).
- `list_errors.py --dataset {sent,agg} --source X --target Y --show N [--train-on train|trainval]` ->
  `list_errors.json`, `list_errors.md` (+ `mt_cache_<target>2<source>.json`).
- `list_errors.py --dataset sent` reproduces `it_errors.py` **byte for byte** on the sent cells
  (identical `it_errors.md`, `.json`; identical bucket counts on agg by inspection).
- Both scripts are single-shot: they overwrite their fixed output paths on every run.

**Inputs.**

| dataset | embeddings | labels / split | models |
|---|---|---|---|
| sent | `embeddings/mean/emb_{en,it,de,fr}.npy`, 2009x768, sliced to 2006 kept rows | `flores200_en_it_de_fr.csv`; 12 classes; 997/504/505 dev/test-half split, seed 0 | EN-trained + IT-trained |
| agg | `embeddings_agg/mean/emb_{en,it,de,fr}.npy`, 562x768 | `flores_agg.csv`; 11 classes (class 0 "business and economy" dropped); 276/140/139 | EN-trained + IT-trained |

Feature = `flores_calib.perlang_z [both]`: L2-normalize, then z-score with **each language's own train
rows** (unlabeled target train data is used). XGBoost = `flores_calib.XGB_COMMON` (softprob, depth 8,
mcw 5, lr 0.1, n 400, hist, max_bin 64, cpu, n_jobs 2, seed 0) + `num_class`. Both models fit the
**train split only** and are scored on the **same** target-language test rows, `Xt[target][te]`, against
the same `y[te]`.

**Buckets** (target test items, label = gold):

| bucket | definition | meaning |
|---|---|---|
| `both_wrong` | source wrong AND target wrong | hard / label-ambiguous items |
| `transfer_gap` | source wrong AND target right | transfer-specific misses |
| `it_only_wrong` (it_errors) / `target_only_wrong` (list_errors) | source right AND target wrong | damage from using the source model |
| `both_right` | source right AND target right | — |

The four buckets are disjoint and sum to the test size, so they are internally consistent with
`*_errors_total`.

**Results.**

| dataset | test n | EN acc | IT acc | both_wrong | transfer_gap | *_only_wrong | both_right | EN/IT acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sent | 505 | 0.4139 (296 err) | 0.4832 (261 err) | 209 | 87 | 52 | 157 | 0.857 |
| agg | 139 | 0.4388 (78 err) | 0.5827 (58 err) | 50 | 28 | 8 | 53 | 0.753 |

Context: majority-class rate is 0.261 (sent) / 0.223 (agg); chance is 0.083 / 0.091. The Italian
model's advantage is not noise: McNemar exact p = **0.0038** (sent, 139 discordant) and **0.0012** (agg,
36 discordant).

Top EN-trained confusions on the Italian test set (gold -> EN pred, out of 296 errors):
`sci/tech -> travel` 29, `history -> travel` 18, `ent./culture -> travel` 16, `travel -> sci/tech` 14,
`ent./culture -> sci/tech` 13, `geography -> travel` 13, `education -> sci/tech` 12, `politics ->
travel` 9, `health -> travel` 8. **113 of 296 (38%) of all EN errors predict `travel`**, 66 (22%)
predict `sci/tech`; two of twelve classes absorb **60%** of the error mass. `travel` is 132/505 (26%) of
the test rows. agg is flatter: `politics -> crime/law` 5, `education -> sci/tech` 4, `sports -> travel`
4, `travel -> sci/tech` 4.

**Verdict.** Transfer is a **minority** of the EN model's Italian error. In-language training recovers
only **87 of 296** EN errors (29%) and introduces 52 new ones; net gain is 35 rows (+6.9 pp) on sent and
20 rows (+14.4 pp) on agg. The dominant mass, **209 items (41% of the test set)**, is wrong for both
models, so it cannot be explained by language transfer. Error analysis at item level is reproducible,
and `list_errors.py` generalises it to any of the 4 languages.

**0.975 bar: N/A.** This is descriptive error analysis. No model ratio against the E18 headline metric
(AUC) is stored, so the bar does not apply. For reference only, the stored accuracy ratio EN/IT is 0.857
(sent) and 0.753 (agg); accuracy is **not** the E18 headline metric (AUC) and this ratio must not be
quoted as passing or failing the 0.975 bar.

**Caveats.**

- **Listing sample.** Each bucket lists the first 30 items in `test_pos` order, where test rows are a
  seed-0 permutation of the devtest half. This is a deterministic 30/209 (14%) prefix, not a random
  sample and not ranked by confidence. On sent the 30 `both_wrong` rows all fall in `test_pos` 1-62.
- **Small n / imbalance.** `business and economy` has 4 and `safety` 8 test rows (sent); agg has 276
  train rows for 11 classes. Per-class bucket counts below ~10 carry no signal.
- **Not a matched comparison.** The EN model sees EN-train z-statistics, the IT model sees IT-train
  z-statistics (`perlang_z [both]`), so the pair differs in training language *and* in feature
  normalisation.
- **IT model is below its ceiling.** It fits 997 train rows; `flores_ceiling` puts Italian micro-F1 at
  0.5624 with train+val (1501 rows). A stronger Italian model would shrink the transfer gap further, so
  **87 is a lower bound** on the in-language gain.
- Single seed, one target language, no CIs. Item lists are snippets (200 chars).
- `list_errors.json` as stored holds only the **sent, en->it** run; the earlier agg (and de) runs left
  only side-effect MT caches.

---

# PART D — Classifiers, MT, calibration, retrieval

Entries that test whether the headline is an XGBoost artefact (E21), whether the MT step can be
simplified (E22), whether the bar is reachable at all (E19), whether the embedding space retrieves
translations (E39), and what the pre-E18 baseline actually said (E40).

## E21 — Non-XGBoost classifiers — DONE (audited 2026-10-19; 0.975 claim not supported)

**Script to reproduce:** `models_compare.py` — `python models_compare.py`. Reproduction status:
**PARTIAL** (GaussianNB, MLP, XGBoost and SVC_linear cells re-ran exactly; QDA rows do not reproduce).

**Question.** Does any non-XGBoost classifier, trained on English mean-pooled EuroBERT features with the
model-free `perlang_z` transform, reach >= **0.975** × its own English self score on EN→IT transfer —
on macro-F1 and on OVR-macro AUC — and does it beat the XGBoost reference?

**Script / outputs.**

- `models_compare.py` -> `models_compare.json` (numbers), `.md` (17-model tables), `.png`, `.log`,
  `.pid`, `models_compare_status.json`.
- `python models_compare.py` runs both backbones in the foreground (~1449 s). `--launch` runs detached.
- `--backbone {210m,610m,both}`, `--only m1,m2`, `--smoke`.
- There is no render-only mode; the `.md` and `.png` are rebuilt only by retraining.
- **`--only` and `--smoke` overwrite the full JSON/MD/PNG with partial results** (code lines 553-554,
  601-602, 609-620), although the docstring calls the outputs "create-only". Back up the full artifacts
  before any partial run.

**Inputs.**

| item | value |
|---|---|
| dataset | `flores200_en_it_de_fr.csv`, 2006 aligned sentence rows, 12 topic classes |
| 210m embeddings | `embeddings/mean/emb_{en,it}.npy`, 2009 rows sliced by `kept` to 2006, 768-d |
| 610m embeddings | `embeddings_610/mean/emb_{en,it}.npy`, 1152-d |
| split | `dev` 997 train; `devtest` 1009 permuted with seed 0 -> val 504 / test 505 |
| final fit | train+val 1501, evaluate on untouched test 505 |
| test class counts | 4, 31, 30, 46, 20, 20, 45, 42, 8, 97, 30, 132 (min 4, max 132) |

The script loads all four languages (`LANGS`, line 78) but only EN and IT enter a model (lines
309-333). DE and FR are unused.

**Protocol.**

- Frozen EuroBERT **mean** sentence embeddings. No fine-tuning.
- `perlang_z [both]`: L2-normalize each row, then z-score with each language's **own train** mean/std.
  Transfer test rows therefore use unlabeled IT train statistics; the English model is not strictly
  target-free.
- 16 scikit-learn classifiers plus an XGBoost reference (17 total) per backbone.
- One fixed configuration per model, except `LogisticRegression` (C in {0.1, 1, 10}) and `KNN` (k in
  {5, 15}), which pick by own-language val macro-F1 and are then refit on train+val.
- Adaptations: `GradientBoosting` uses `max_features='sqrt'`; `SVC` uses `decision_function_shape='ovr'`
  scores for AUC.
- Evaluated four ways per cell: EN-self, IT-self, EN→IT, IT→EN. Metrics: macro-F1 (primary), accuracy,
  OVR-macro AUC. Macro-F1 uses labels 0..11 and `zero_division=0`.
- Seed 0, single split, no CIs, no bootstrap.
- Bar used below: **EN→IT / EN-self >= 0.975**, applied post hoc to macro-F1 and AUC. E21 itself did
  not set this bar; its text claimed only "beats XGBoost".

**Coverage.** `models_compare.json` has **17 models per backbone**; the E21 table in `EXPERIMENTS.md`
has **5**. This is an omission, not a clean top-5. KNN is **2nd on 210m transfer** (0.439) and 4th on
610m (0.502). QDA is the only stored 610m model that passes both 0.975 metrics. SVC_rbf and SGD also
pass the 610m AUC bar.

| set | models |
|---|---|
| shown in E21 (5) | SVC_linear, MLP, LogisticRegression, GaussianNB, XGBoost (ref) |
| **omitted (12)** | **SVC_rbf, SGD, LDA, HistGradientBoosting, Ridge, KNN, LinearSVC, GradientBoosting, ExtraTrees, RandomForest, DecisionTree, QDA** |

**Results — original five rows, with the two directions separated.** `transfer` in E21 is the **mean of
EN→IT and IT→EN**, not EN→IT.

| backbone | model | EN-self F1 | EN→IT F1 | IT→EN F1 | mean transfer F1 | EN-self AUC | EN→IT AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| 210m | SVC_linear | 0.5553 | 0.4072 | 0.4536 | 0.4304 | 0.9213 | 0.8805 |
| 210m | MLP | 0.5376 | 0.4132 | 0.4513 | 0.4322 | 0.9219 | 0.8684 |
| 210m | LogisticRegression | 0.5032 | 0.3383 | 0.4347 | 0.3865 | 0.9047 | 0.8288 |
| 210m | GaussianNB | 0.4625 | 0.4638 | 0.4427 | 0.4532 | 0.9019 | 0.8792 |
| 210m | XGBoost | 0.4865 | 0.3657 | 0.4461 | 0.4059 | 0.9038 | 0.8461 |
| 610m | SVC_linear | 0.5831 | 0.5187 | 0.6061 | 0.5624 | 0.9389 | 0.9182 |
| 610m | MLP | 0.5641 | 0.4917 | 0.5431 | 0.5174 | 0.9390 | 0.8967 |
| 610m | LogisticRegression | 0.5411 | 0.4742 | 0.5534 | 0.5138 | 0.9347 | 0.8848 |
| 610m | GaussianNB | 0.4843 | 0.4559 | 0.5318 | 0.4939 | 0.9140 | 0.8834 |
| 610m | XGBoost | 0.4948 | 0.4608 | 0.4718 | 0.4663 | 0.9098 | 0.8721 |

**Full 17-model results, 0.975 pass/fail.** Ratios are EN→IT / EN-self. `†` = omitted from the E21
table. Sorted by F1 ratio.

### 210m (768-d)

| model | EN-self F1 | EN→IT F1 | F1 ratio | F1 >=0.975 | EN-self AUC | EN→IT AUC | AUC ratio | AUC >=0.975 |
|---|---:|---:|---:|:--:|---:|---:|---:|:--:|
| QDA † | 0.0824 | 0.0846 | 1.0271 | PASS | 0.5057 | 0.5111 | 1.0106 | PASS |
| GaussianNB | 0.4625 | 0.4638 | **1.0027** | PASS | 0.9019 | 0.8792 | 0.9749 | fail |
| KNN † | 0.4713 | 0.4044 | 0.8581 | fail | 0.9187 | 0.8600 | 0.9361 | fail |
| MLP | 0.5376 | 0.4132 | 0.7686 | fail | 0.9219 | 0.8684 | 0.9419 | fail |
| XGBoost | 0.4865 | 0.3657 | 0.7518 | fail | 0.9038 | 0.8461 | 0.9362 | fail |
| SVC_linear | 0.5553 | 0.4072 | 0.7333 | fail | 0.9213 | 0.8805 | 0.9557 | fail |
| LogisticRegression | 0.5032 | 0.3383 | 0.6723 | fail | 0.9047 | 0.8288 | 0.9161 | fail |
| HistGradientBoosting † | 0.4857 | 0.3168 | 0.6522 | fail | 0.8982 | 0.8206 | 0.9136 | fail |
| SGD † | 0.5118 | 0.3323 | 0.6492 | fail | 0.9168 | 0.8733 | 0.9526 | fail |
| SVC_rbf † | 0.5138 | 0.3336 | 0.6492 | fail | 0.9288 | 0.9007 | 0.9697 | fail |
| GradientBoosting † | 0.3951 | 0.2504 | 0.6338 | fail | 0.8728 | 0.7966 | 0.9127 | fail |
| RandomForest † | 0.3877 | 0.2042 | 0.5266 | fail | 0.9036 | 0.8582 | 0.9497 | fail |
| ExtraTrees † | 0.3907 | 0.2048 | 0.5242 | fail | 0.9180 | 0.8611 | 0.9381 | fail |
| DecisionTree † | 0.2354 | 0.1203 | 0.5113 | fail | 0.5906 | 0.5233 | 0.8861 | fail |
| LinearSVC † | 0.4474 | 0.2017 | 0.4509 | fail | 0.8622 | 0.6978 | 0.8093 | fail |
| Ridge † | 0.4844 | 0.2165 | 0.4470 | fail | 0.8655 | 0.6808 | 0.7866 | fail |
| LDA † | 0.5015 | 0.2160 | 0.4307 | fail | 0.8760 | 0.6869 | 0.7841 | fail |

### 610m (1152-d)

| model | EN-self F1 | EN→IT F1 | F1 ratio | F1 >=0.975 | EN-self AUC | EN→IT AUC | AUC ratio | AUC >=0.975 |
|---|---:|---:|---:|:--:|---:|---:|---:|:--:|
| QDA † | 0.0739 | 0.0813 | 1.0995 | PASS | 0.4949 | 0.5125 | 1.0356 | PASS |
| GaussianNB | 0.4843 | 0.4559 | 0.9415 | fail | 0.9140 | 0.8834 | 0.9665 | fail |
| XGBoost | 0.4948 | 0.4608 | 0.9312 | fail | 0.9098 | 0.8721 | 0.9586 | fail |
| KNN † | 0.5061 | 0.4691 | 0.9270 | fail | 0.9382 | 0.9126 | 0.9727 | fail |
| SVC_linear | 0.5831 | 0.5187 | 0.8895 | fail | 0.9389 | 0.9182 | **0.9780** | PASS |
| LogisticRegression | 0.5411 | 0.4742 | 0.8763 | fail | 0.9347 | 0.8848 | 0.9466 | fail |
| MLP | 0.5641 | 0.4917 | 0.8716 | fail | 0.9390 | 0.8967 | 0.9549 | fail |
| SVC_rbf † | 0.5233 | 0.4470 | 0.8541 | fail | 0.9421 | 0.9202 | 0.9768 | PASS |
| HistGradientBoosting † | 0.4687 | 0.3941 | 0.8409 | fail | 0.8933 | 0.8332 | 0.9327 | fail |
| SGD † | 0.5708 | 0.4603 | 0.8065 | fail | 0.9357 | 0.9138 | 0.9765 | PASS |
| DecisionTree † | 0.1739 | 0.1354 | 0.7784 | fail | 0.5582 | 0.5303 | 0.9502 | fail |
| RandomForest † | 0.3192 | 0.2484 | 0.7781 | fail | 0.9101 | 0.8590 | 0.9438 | fail |
| ExtraTrees † | 0.2936 | 0.2265 | 0.7713 | fail | 0.9017 | 0.8599 | 0.9536 | fail |
| GradientBoosting † | 0.4088 | 0.2951 | 0.7220 | fail | 0.8805 | 0.8341 | 0.9473 | fail |
| LinearSVC † | 0.5212 | 0.3609 | 0.6924 | fail | 0.8879 | 0.8247 | 0.9288 | fail |
| LDA † | 0.4228 | 0.2513 | 0.5945 | fail | 0.8346 | 0.7286 | 0.8730 | fail |
| Ridge † | 0.4363 | 0.2464 | 0.5647 | fail | 0.8255 | 0.7409 | 0.8975 | fail |

**0.975 pass list (stored JSON; EN→IT / EN-self).**

| metric | backbone | model | EN-self | EN→IT | ratio | absolute verdict |
|---|---|---|---|---:|---:|---|
| macro-F1 | 210m | QDA | 0.0824 | 0.0846 | 1.0271 | **chance/noise; stored values do not reproduce** |
| macro-F1 | 210m | GaussianNB | 0.4625 | 0.4638 | **1.0027** | real but weak; AUC fails, reverse F1 0.957 |
| macro-F1 | 610m | QDA | 0.0739 | 0.0813 | 1.0995 | **chance/noise; flips to 0.797 on re-run** |
| AUC | 210m | QDA | 0.5057 | 0.5111 | 1.0106 | **chance/noise** |
| AUC | 610m | QDA | 0.4949 | 0.5125 | 1.0356 | **chance/noise; flips to 0.974 on re-run** |
| AUC | 610m | SVC_linear | 0.9389 | 0.9182 | 0.9780 | strong; F1 ratio 0.889 |
| AUC | 610m | SVC_rbf | 0.9421 | 0.9202 | 0.9768 | strong; F1 ratio 0.854 |
| AUC | 610m | SGD | 0.9357 | 0.9138 | 0.9765 | strong; F1 ratio 0.806 |

Counts: **macro-F1 3/34, AUC 5/34, both metrics 2/34**. The 2/34 both-metric cells are QDA, which is
at chance and unstable. **No non-degenerate classifier passes both metrics.** The claim "standard
strong classifiers fail" is true for **macro-F1**, but **not for AUC at 610m**: SVC_linear, SVC_rbf and
SGD pass the AUC bar.

**Absolute scores — why the ratio is misleading.** A 12-class random classifier has macro-F1 ~0.083 and
AUC ~0.5.

| cell | EN-self F1 | EN→IT F1 | ratio | EN-self AUC | EN→IT AUC | ratio |
|---|---:|---:|---:|---:|---:|---:|
| 210m QDA | 0.0824 (≈ chance) | 0.0846 | 1.0271 | 0.5057 | 0.5111 | 1.0106 |
| 210m GaussianNB | 0.4625 | 0.4638 | 1.0027 | 0.9019 | 0.8792 | 0.9749 |
| 610m QDA | 0.0739 (below chance) | 0.0813 | 1.0995 | 0.4949 | 0.5125 | 1.0356 |
| 610m SVC_linear | 0.5831 | 0.5187 | 0.8895 | 0.9389 | 0.9182 | 0.9780 |
| 610m MLP | 0.5641 | 0.4917 | 0.8716 | 0.9390 | 0.8967 | 0.9549 |
| 610m XGBoost | 0.4948 | 0.4608 | 0.9312 | 0.9098 | 0.8721 | 0.9586 |

The best absolute EN→IT macro-F1 is **0.5187** (610m SVC_linear). It **fails** the 0.975 ratio.
GaussianNB passes the F1 ratio with a **0.4638** transfer score because its EN-self denominator is only
0.4625. The ratio rewards a weak model. QDA is at or below chance, so its ratio >1 is noise.

**Direction sensitivity.** E21's `transfer` mean hides a large asymmetry.

| cell | EN→IT F1 ratio | IT→EN F1 ratio | EN→IT AUC ratio | IT→EN AUC ratio |
|---|---:|---:|---:|---:|
| 210m QDA | 1.0271 | 0.9105 | 1.0106 | 0.9936 |
| 210m GaussianNB | 1.0027 | 0.9571 | 0.9749 | 0.9807 |
| 610m QDA | 1.0995 | 1.1531 | 1.0356 | 1.0210 |
| 610m SVC_linear | 0.8895 | 1.0394 | 0.9780 | 1.0000 |
| 610m SVC_rbf | 0.8541 | 0.9393 | 0.9768 | 0.9953 |
| 610m SGD | 0.8065 | 0.9494 | 0.9765 | 0.9977 |

GaussianNB 210m fails F1 if both directions are required (IT→EN 0.9571). SVC_linear 610m passes AUC in
both directions but fails F1 in both (EN→IT 0.8895, IT→EN 1.0394).

**Re-run check (read-only, in-memory; 2 threads).**

| cell | stored EN-self / EN→IT F1 | re-run EN-self / EN→IT F1 | stored ratio | re-run ratio |
|---|---|---|---|---|
| 210m GaussianNB | 0.4625 / 0.4638 | 0.462504 / 0.463772 | 1.0027 | 1.0027 |
| 610m GaussianNB | 0.4843 / 0.4559 | 0.484265 / 0.455928 | 0.9415 | 0.9415 |
| 610m MLP | 0.5641 / 0.4917 | 0.564148 / 0.491700 | 0.8716 | 0.8716 |
| 610m XGBoost | 0.4948 / 0.4608 | 0.494829 / 0.460773 | 0.9312 | 0.9312 |
| 610m SVC_linear | 0.5831 / 0.5187 | 0.583107 / 0.518666 | 0.8895 | 0.8895 |
| 210m QDA | 0.0824 / 0.0846 | **0.0460 / 0.0569** | 1.0271 | 1.2358 |
| 610m QDA | 0.0739 / 0.0813 | **0.0715 / 0.0570** | **1.0995** | **0.7971** |

QDA has a singular covariance in every class (210m: 21–386 effective rank against 768 features; 610m is
rank-deficient by construction), uses the default `reg_param=0`, and emits 12 "not full rank" warnings
that `warnings.filterwarnings("ignore")` hides. Its row is numerically unstable. **The QDA rows must
not be used as evidence.**

**Verdict.** At the **0.975** bar the stored JSON has **3/34 macro-F1 passes and 5/34 AUC passes**. Only
**GaussianNB at 210m** is a stable, non-chance macro-F1 pass (ratio 1.0027), and it is metric- and
direction-fragile: AUC ratio **0.9749** (0.000085 below the bar) and IT→EN F1 ratio **0.9571**. The
610m QDA pass does not survive a re-run. At AUC, SVC_linear, SVC_rbf and SGD at 610m pass with strong
absolute scores (EN→IT AUC 0.914–0.920), but their F1 ratios are 0.89, 0.85 and 0.81. **No classifier
passes both metrics legitimately.** E21's original finding — a linear SVM beats XGBoost on EN-self F1
and on mean transfer F1 — holds (0.5553 vs 0.4865 and 0.4304 vs 0.4059 at 210m; 0.5831 vs 0.4948 and
0.5624 vs 0.4663 at 610m). It is **not** a 0.975 parity result. The 5-row table also omits KNN
(2nd-best 210m transfer) and two of the three 610m AUC passers (SVC_rbf, SGD).

**Caveats.**

- **QDA is degenerate.** Rank-deficient covariance, chance-level absolute scores, hidden warnings, and
  a non-reproducible row. Exclude it or set `reg_param` and log the rank.
- **The ratio denominator rewards weak models.** The bar compares a model to its own self score. A weak
  self score makes a high ratio easy. Report absolute EN-self and transfer scores with every ratio.
- **AUC ratios are compressed by the 0.5 floor.** 0.9389 -> 0.9182 is ratio 0.978 but an absolute drop
  of 0.021. 0.5831 -> 0.5187 is ratio 0.889 but an absolute drop of 0.064. The two metrics are not on
  the same scale.
- **`transfer` is a two-direction mean.** EN→IT is the harder direction. Report EN→IT and IT→EN
  separately for any parity claim.
- **Single seed, no CIs.** Test n = 505; per-class n = 4–132. Macro-F1 gives the 4-row class the same
  weight as the 132-row class. A 0.975 ratio is inside the noise band.
- **Most models are not tuned.** Only LogisticRegression and KNN use a light val grid. SVC, MLP,
  forests, SGD and boosting use fixed configurations; several are **not** scikit-learn defaults (MLP
  hidden 256, RF/ET 300 trees, SGD max_iter 2000/tol 1e-4, LinearSVC max_iter 5000). The comparison is
  "one fixed configuration per model vs a fixed XGBoost reference config".
- **`perlang_z` uses unlabeled target train rows.** The English model sees IT train statistics.
- **Partial runs are destructive.** `--only` and `--smoke` replace the full JSON/MD/PNG.
- Reproduced on 2026-10-19: GaussianNB (both), MLP 610m, XGBoost 610m, SVC_linear 610m match at 6 dp.
  QDA does not.

---

## E22 — MT: bilingual vs multilingual → English — DONE (verified 2026-10-20)

**Script to reproduce:** `mt_compare.py` — `python mt_compare.py` (run from `multi/`; no CLI flags).
Reproduction status: **PARTIAL** (all six BLEU/chrF cells and every delta re-derived exactly, to 4 dp,
from the run's own cached hypotheses; the decode itself was not re-run because a rerun overwrites
tracked outputs and requires GPU inference).

**Question.** Can ONE multilingual many→en MT model stand in for three bilingual models at >= 0.975 ×
their BLEU on FLORES — i.e. is per-language MT specialisation unnecessary for a translate-test
pipeline?

**Script / outputs.**

- `mt_compare.py` -> `mt_compare.json` (numbers), `mt_compare.md` (tables + verdict), `mt_compare.png`.
- No CLI arguments: `N_ROWS`, `BATCH_SIZE`, `MAX_LENGTH`, `NUM_BEAMS`, `USE_FP16` are module constants
  (lines 53-58). `N_ROWS = None` -> full 2006 rows; `subset: false` in the JSON confirms it.
- Hypotheses are cached in `%TEMP%\opencode\mtcache_<model>_<n>_<md5>.json` (lines 74-93). The six files
  from the reported run are timestamped 20:05:36–20:07:57; the JSON was written 20:08:03.
- There is no `--render` path: md/png are only regenerated by re-running the whole script.

**Inputs.**

| item | value |
|---|---|
| data | `flores200_en_it_de_fr.csv`, 2006 rows (997 `dev` / 1009 `devtest`); 3 rows fewer than the 2009-row source |
| reference | the `en` column, one reference per row, 2006 unique (no duplicate English text) |
| languages | it, de, fr — identical rows and row order for both systems |
| bilingual | `Helsinki-NLP/opus-mt-it-en`, `-de-en`, `-fr-en` (6+6 layers, d_model 512; 85.8M params for it-en) |
| multilingual | `Helsinki-NLP/opus-mt-mul-en`, one model reused for it/de/fr (opus2m-2020-08-01; 77.5M params; source set includes ita/deu/fra; no language tag needed) |
| decode | greedy `num_beams=1`, fp16 on cuda, batch 64, `max_length=512`, torch threads 2 |

Longest source is 71 words (it) — well inside 512 subword tokens, so truncation is not a factor. `id`
is a per-split row index (`dev` 0-996, `devtest` 3-1011 after the 3-row deletion), so rows must be
keyed by `split`+`id`, not `id` alone.

**Protocol.** Translate every it/de/fr row into English with each family; score corpus BLEU and chrF
(sacrebleu; at audit time 2.6.0, 13a tokenizer, default chrF `char_order=6`, `word_order=0`) against the
same single-reference `en` list; macro = unweighted mean of the three per-language corpus scores. Same
2006 source rows and the same reference list feed both systems, so the comparison is paired.

**Results (BLEU).**

| lang | bilingual | multilingual (mul-en) | Δ | ratio mul/bil | >= 0.975 |
|---|---:|---:|---:|---:|:--:|
| it | 29.7733 | 23.1609 | -6.6124 | 0.7779 | no |
| de | 40.6498 | 29.2159 | -11.4338 | 0.7187 | no |
| fr | 42.0677 | 31.7873 | -10.2804 | 0.7556 | no |
| **macro** | **37.4969** | **28.0547** | **-9.4422** | **0.7482** | **no** |

**Results (chrF, present in the JSON/MD but absent from the EXPERIMENTS.md entry).**

| lang | bilingual | multilingual | Δ | ratio |
|---|---:|---:|---:|---:|
| it | 60.0139 | 54.0181 | -5.9958 | 0.9001 |
| de | 66.3492 | 56.8221 | -9.5271 | 0.8564 |
| fr | 66.8887 | 58.9048 | -7.9839 | 0.8806 |
| **macro** | **64.4173** | **56.5817** | **-7.8356** | **0.8784** |

**Verdict.** At the **0.975** bar the multilingual/bilingual BLEU ratio is **0.778 / 0.719 / 0.756**
(it/de/fr), so **0/3 target languages pass** (chrF: 0.900 / 0.856 / 0.881, also 0/3). This is a
**quality ratio between two MT systems**, not a target/English-model ratio: there is no English-system
denominator here, and BLEU is not linear in quality, so 0.72 BLEU means far worse than "72% of the
bilingual quality". The documented finding — three bilingual models beat one many→en model by 9.4 BLEU
on this data and config — holds.

**Caveats.**

- Capacity / training-data confound: 3 × 85.8M specialised models vs 1 × 77.5M opus2m model whose ~2M
  mixed pairs are spread over ~200 source languages. The result does not isolate "multilingual
  architecture is worse"; it measures this specific 2020 opus2m pairing.
- One decode configuration (greedy, fp16, one GPU pass), one reference per sentence, no bootstrap CI and
  no significance test. The -9.4 BLEU margin is far outside plausible decoder noise, but the entry
  reports no uncertainty.
- Macro is an unweighted mean of per-language corpus BLEU (and is computed from the already-4dp-rounded
  per-language values, lines 163-172). Pooled 3-language corpus BLEU is 37.4706 vs 28.0267 (Δ -9.4439);
  the choice changes the headline by 0.03 BLEU.
- Not comparable to published FLORES-devtest numbers: dev and devtest are pooled (2006 rows).
- Cache is in `%TEMP%` and its key covers only model name + source text — not decode config, device,
  dtype or library version. Config changes silently reuse stale hypotheses.
- Verified 2026-10-20 by recomputing all BLEU/chrF cells from the six cache files at stored precision.

---

## E19 — Calibration and ceiling feasibility — DONE (audited 2026-10-19)

**Script to reproduce:** `flores_calib.py` + `flores_ceiling.py` — `python flores_calib.py` then
`python flores_ceiling.py` (detached: `python flores_ceiling.py --launch`), run from `multi/`.
Reproduction status: **PARTIAL**.

**Question.** Can ONE English-trained XGBoost on EuroBERT-210m mean features, with label-free
calibration, reach **0.975 × English self** single-label F1 in every target language? If not, is the bar
even reachable by the target's own in-language model (its ceiling)?

**Script / outputs.**

- `flores_calib.py` -> `flores_calib.json` (numbers), `flores_calib.md` (tables). Trains one EN XGBoost
  per dataset; EN val fits five label-free calibrations. `--smoke` runs a 60-tree check.
- `flores_ceiling.py` -> `flores_ceiling.json`, `flores_ceiling.md`. Reads the saved `flores_calib.json`;
  trains one in-language XGBoost per (dataset, language). `--launch` detaches; `--smoke` = 60 trees.
- Neither script has a render-only or resume mode. Both rewrite their JSON/MD unconditionally, and the
  docstrings' "create-only" claim is false. Run `flores_calib.py` first: the ceiling script reads the
  calib JSON and fails on a missing or half-written file.

**Inputs.**

| dataset | embeddings | labels / split | EN train / val / test |
|---|---|---|---|
| sent | `embeddings/mean/emb_{en,it,de,fr}.npy`, 2009x768, sliced to 2006 kept rows | `flores200_en_it_de_fr.csv`; 12 classes; parallel rows | 997 / 504 / 505 |
| agg | `embeddings_agg/mean/emb_{en,it,de,fr}.npy`, 562x768 | `flores_agg.csv`; 11 classes (business/economy dropped); parallel rows | 276 / 140 / 139 |

**Protocol.**

- Features `perlang_z`: L2-normalize, then z-score with each language's own **train** rows (unlabeled;
  EN stats for the EN model, target stats for the target test).
- One EN XGBoost per dataset: softprob, depth 8, 400 trees, lr 0.1, hist, max_bin 64, seed 0,
  `n_jobs=2`. Trained on EN **train** rows. EN **val** fits the calibrations; target test is scored.
- Calibrations are label-free: `baseline` argmax, `temperature` (EN-val NLL), `prior_sinkhorn`,
  `quantile_hardprior`, `bbse`. Target labels are never used.
- **Metric: single-label argmax micro-F1 and macro-F1** (micro-F1 == accuracy exactly; verified in
  every stored cell), plus macro OVR AUC. AUC is invariant across calibrations.
- **Ceiling:** each language's own model, trained on that language's **train+val** rows (1501 sent / 416
  agg) and tested on its own **test** rows (505 / 139). Same row indices, same parallel labels, plain
  argmax.
- **Two English-self references exist.** Train+val (same protocol as the ceiling): sent 0.6059 micro /
  0.4865 macro; agg 0.6187 / 0.5573. Train-only (the `flores_calib` ratio denominator): sent 0.5366 /
  0.4032; agg 0.5540 / 0.5014. The old E19 table uses the **train+val** reference.

**Results — in-language ceilings.**

| dataset | lang | micro-F1 | macro-F1 | AUC |
|---|---|---:|---:|---:|
| sent | en (self) | 0.6059 | 0.4865 | 0.9038 |
| sent | it | 0.5624 | 0.4343 | 0.8829 |
| sent | de | 0.5366 | 0.4355 | 0.8835 |
| sent | fr | 0.5822 | 0.4675 | 0.8741 |
| agg | en (self) | 0.6187 | 0.5573 | 0.9142 |
| agg | it | 0.6187 | 0.5588 | 0.9016 |
| agg | de | 0.6043 | 0.5322 | 0.8949 |
| agg | fr | 0.5324 | 0.4542 | 0.8852 |

**Results — feasibility at 0.975 (primary: train+val English-self, same protocol as the ceiling).**
Bar: sent 0.5908 micro / 0.4743 macro; agg 0.6032 / 0.5434. Margin = bar - ceiling; positive =
INFEASIBLE.

| dataset | target | ceil micro | margin micro | ceil macro | margin macro | infeasible? |
|---|---|---:|---:|---:|---:|:--:|
| sent | it | 0.5624 | **+0.0284** | 0.4343 | **+0.0400** | **both** |
| sent | de | 0.5366 | **+0.0542** | 0.4355 | **+0.0388** | **both** |
| sent | fr | 0.5822 | **+0.0086** | 0.4675 | **+0.0068** | **both** |
| agg | it | 0.6187 | -0.0155 | 0.5588 | -0.0154 | no |
| agg | de | 0.6043 | -0.0011 | 0.5322 | **+0.0111** | **macro** |
| agg | fr | 0.5324 | **+0.0709** | 0.4542 | **+0.0892** | **both** |

At **0.975** the split is **sent 6/6 infeasible, agg 3/6 infeasible**. At 0.9 it was sent 3/6, agg 2/6.
Every cell that was infeasible at 0.9 stays infeasible at 0.975.

**Results — the three rows carried by the old E19 table.**

| dataset | target (metric) | EN self | 0.9 bar | 0.975 bar | ceiling | margin @0.9 | margin @0.975 | 0.9 | 0.975 |
|---|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| sent | de (micro) | 0.6059 | 0.545 | **0.591** | 0.5366 | +0.0087 | **+0.0542** | NO | **NO** |
| sent | it (macro) | 0.4865 | 0.438 | **0.474** | 0.4343 | +0.0035 | **+0.0400** | NO | **NO** |
| agg | fr (micro) | 0.6187 | 0.557 | **0.603** | 0.5324 | +0.0245 | **+0.0709** | NO | **NO** |

All three stay infeasible and become **2.9–11.4× more infeasible** (margin grows by 0.037–0.046).

**Results — sensitivity to the English-self denominator at 0.975.**

| dataset | target | train+val verdict | train-only verdict | train-only margin (micro / macro) |
|---|---|---|---|---|
| sent | it | both infeasible | feasible | -0.0392 / -0.0412 |
| sent | de | both infeasible | feasible | -0.0134 / -0.0424 |
| sent | fr | both infeasible | feasible | -0.0590 / -0.0744 |
| agg | it | feasible | feasible | -0.0786 / -0.0699 |
| agg | de | macro infeasible | feasible | -0.0642 / -0.0433 |
| agg | fr | both infeasible | both infeasible | +0.0077 / +0.0347 |

The sent verdict **flips completely** with the denominator. The file must name which English self it
uses; the old table does not.

**Results — calibrated transfer / ceiling (best per metric).**

| dataset | target | micro | used by `flores_ceiling` | macro | used by `flores_ceiling` |
|---|---|---:|---:|---:|---:|
| sent | it | 0.736 | 0.708 | 0.687 | 0.687 |
| sent | de | 0.808 | 0.731 | 0.666 | 0.661 |
| sent | fr | 0.748 | 0.721 | 0.744 | 0.744 |
| agg | it | 0.814 | 0.814 | 0.625 | 0.625 |
| agg | de | 0.857 | 0.845 | 0.820 | 0.688 |
| agg | fr | 1.027 | 1.027 | 1.010 | 0.907 |

**Verdict.** At the **0.975** bar, **sent is fully infeasible (6/6 cells)** and **agg is partly
infeasible (3/6: de macro, fr micro, fr macro)**. The three rows in the old E19 table are correct at 0.9
and remain correct at 0.975, with larger margins. However, the old finding "the residual is
ceiling/label-noise bound, not transfer" is **not supported**: the EN-trained calibrated model reaches
0.67–0.81 of the target's own ceiling on sent (micro 0.71–0.81, macro 0.66–0.74) and 0.63–1.03 on agg
(micro 0.81–1.03, macro 0.63–1.01); on agg-fr it *exceeds* the "ceiling" (1.03 micro, 1.01 macro). A
large transfer gap remains inside the ceiling. The infeasibility result bounds the **bar**, not the
observed gap. The ceiling is a plain untuned argmax trained on 50% more rows than the transfer model.
It is a lower bound on what the target can achieve, and a training-budget-mismatched reference — not a
label-noise bound.

**Caveats.**

- Single seed, no CIs. Test sets are 505 (sent) / 139 (agg). The micro-F1 standard error is about 0.022
  (sent) and 0.042 (agg); most infeasibility margins at 0.9 and several at 0.975 are within 1–2 SE.
  Treat the verdicts as directional, not significant.
- The ceiling trains on train+val while the transfer model trains on train only (997 vs 1501; 276 vs
  416 rows). The bar and the ceiling share the train+val protocol, so the feasibility test is
  internally consistent; the transfer/ceiling ratio is not.
- The ceiling uses plain argmax. The transfer uses EN-val-tuned thresholds and prior matching. No target
  val is left to tune the ceiling, so the ceiling is systematically pessimistic there.
- `perlang_z` uses unlabeled target train rows, so "one English model, no target data" is not literally
  true.
- No per-row predictions are saved in either JSON, so a paired test (McNemar) on the same 505/139 rows
  cannot be run from stored artifacts.
- The run is not resumable. Re-running `flores_calib.py` overwrites the JSON that `flores_ceiling.json`
  cites; the two files are only consistent if run in order.

---

## E39 — Exact-translation retrieval (FLORES + Europarl) — DONE (audited 2026-10-20)

**Script to reproduce:** `table1.py` / `table1_agg.py` / `table1_europarl_multilang.py` — `python
table1.py` (currently fails: see Inputs), `python table1_agg.py`, `python
table1_europarl_multilang.py all`. Reproduction status: **PARTIAL** (FLORES and FLORES-agg reproduce;
Europarl cannot be re-run).

**Question.** How well does one EuroBERT-210m embedding space retrieve the exact translation of a
sentence when only a closed-form language-shift vector `D` is added to the keys? All directions are
foreign-to-English (`it/de/fr -> en`, `de/es/fr/it/nl/pt -> en`).

**Scripts / outputs.**

- `table1.py` — FLORES sentence-level retrieval, 4 pooling modes, stdout only (README table).
- `table1_agg.py` — same protocol on URL-aggregated FLORES (562 rows), `mean` and `bos`.
- `table1_europarl_multilang.py` — Europarl v7 retrieval for six languages against English; writes
  `europarl_multilang_table1.md`, `europarl_multilang_progress.json`, a run log, a status file and a PID
  file. **None of these files exist in the repo and the md was never committed.**
- `fetch_europarl_langs.py` — downloads/extracts the Europarl xx-en corpora and (re)writes
  `europarl_all_5k.csv`.
- `tools/fetch_flores_en_pl.py` — builds a 25-row EN->PL devtest sample (`flores_en_pl.csv`). No
  retrieval experiment reads it.

**Inputs.**

| corpus | data | embeddings | split |
|---|---|---|---|
| FLORES sent | `flores200_en_it_de_fr.csv`, 2006 rows (997 dev / 1009 devtest) | `embeddings/<mode>/emb_{en,it,de,fr}.npy`, **2009x768** | `split` column |
| FLORES agg | `flores_agg.csv`, 562 URL rows | `embeddings_agg/<mode>/emb_{en,it,de,fr}.npy`, 562x768 | inferred per URL from the sentence CSV |
| Europarl | `europarl_all_5k.csv`, 30,000 rows (6 × 5,000) | `europarl_{xx}_en_5k_embeddings/<mode>/emb_{xx,en}.npy` — **directories exist but are empty** | `split` column |

The FLORES caption table was produced on the 2009-row CSV from commit `196bbd3` (997 dev / 1012
devtest). The current CSV is the same file with three devtest rows removed (ids 0,1,2). `table1.py`
never slices, so it fails on the shipped CSV:

```
IndexError: boolean index did not match indexed array along axis 0;
size of axis is 2009 but size of corresponding boolean axis is 2006
```

To reproduce the README table, use the old CSV: `git show 196bbd3:flores200_en_it_de_fr.csv` as
`flores200_en_it_de_fr.csv` next to a copy of `table1.py` (unmodified) with `embeddings/` linked;
`python table1.py`.

**Protocol.**

- Shift vector: `D_ab = mean over dev rows of (emb_a - emb_b)`, fitted on **raw** embeddings. Meaning
  cancels in the paired difference. One 768-dim vector per direction, no training.
- Scoring: `sim = l2norm(emb_a[test]) @ l2norm(emb_b[test] + D).T`. `D` is added to the **key** side
  only; queries are never shifted. The shifted keys are re-normalized after the addition.
- Rank of query `i`: `1 + #{keys with similarity strictly greater than its own translation key}`. Rank 1
  means the exact translation is the nearest key. Ties favour the true key (strict `>`).
- `MRR = mean(1/rank)`, `top-1 = share(rank == 1)`, `median rank = median(rank)` (numpy).
- Chance: mean rank `(N+1)/2`; top-1 `1/N` (0.1% at N=1012).
- Language normalization: **none**. FLORES and agg use the literal codes `en/it/de/fr`; Europarl uses
  `de/es/fr/it/nl/pt` as CSV columns and filters rows on the exact `lang` string. Texts are not
  Unicode-normalized.

**Results — FLORES (reproduced, 997 dev / 1012 devtest, chance 506.5).**

| mode | top-1 raw | top-1 +D | MRR +D | mean rank raw | mean rank +D | median rank raw | median rank +D |
|---|---|---|---|---|---|---|---|
| mean | 38.5–60.6% | 83.9–90.0% | 0.882–0.931 | 2.1–8.5 | 1.6–4.4 | 1–2 | 1 |
| lead | 7.3–22.6% | 11.1–25.9% | 0.155–0.325 | 189.7–265.6 | 189.6–267.1 | 33–139 | 22–128 |
| bos | 2.5–6.2% | 4.4–10.7% | 0.075–0.150 | 280.0–364.6 | 257.0–330.0 | 166–291 | 116–224 |
| eos | 2.0–2.5% | 10.0–15.8% | 0.165–0.230 | 200.4–259.8 | 154.8–190.3 | 107–164 | 36–62 |

Cells are min–max over `it->en`, `de->en`, `fr->en`.

**Results — FLORES-agg (reproduced, 281 dev / 280 devtest, chance 140.5).**

| mode | top-1 raw | top-1 +D | MRR raw | MRR +D | median rank +D |
|---|---|---|---|---|---|
| mean | 92.9–98.9% | 98.2–100.0% | 0.962–0.994 | 0.989–1.000 | 1 |
| bos | 4.3–13.2% | 11.4–21.1% | 0.076–0.201 | 0.158–0.298 | 11–60 |

`mean` per direction (raw -> +D): it 92.9 -> 98.2, de 96.4 -> 100.0 (MRR 1.000), fr 98.9 -> 99.6. Row
281 (`Toronto_tea`, `n_sentences=0`, all texts empty) is the one URL with no unique source split; it is
excluded and reported.

**Results — Europarl (quoted from README, NOT re-run; chance 2000.5, 1,000 dev / 4,000 devtest).**

| pair | top-1 raw | top-1 +D | MRR raw | MRR +D | mean rank raw | mean rank +D | median rank raw | median rank +D |
|---|---|---|---|---|---|---|---|---|
| de-en | 4.2% | 41.5% | 0.102 | 0.493 | 177.2 | 111.4 | 54 | 3 |
| es-en | 23.6% | 66.9% | 0.355 | 0.723 | 62.0 | 60.2 | 6 | 1 |
| fr-en | 22.6% | 62.3% | 0.341 | 0.686 | 53.6 | 60.1 | 7 | 1 |
| it-en | 19.6% | 45.6% | 0.304 | 0.527 | 100.2 | 118.4 | 10 | 2 |
| nl-en | 6.0% | 45.9% | 0.167 | 0.531 | 109.9 | 115.1 | 14 | 2 |
| pt-en | 21.5% | 50.6% | 0.324 | 0.577 | 98.7 | 101.8 | 8 | 1 |

No stored JSON/markdown/log backs these six rows. They are unverified.

**The 0.975 bar: N/A.** These are rank metrics, not a target/English model score ratio, and no ratio is
stored. There is also no English self-retrieval baseline here, so a `foreign / English` retrieval ratio
cannot be formed at all.

**Verdict.** The retrieval protocol is a clean closed-form result. On FLORES, `mean` raw top-1 is
38.5–60.6% and `+D` lifts it to 83.9–90.0% (MRR 0.882–0.931, mean rank 1.6–4.4). Aggregated URL-level
texts saturate: 92.9–98.9% raw, 98.2–100.0% with `+D`. `bos`/`lead`/`eos` stay near chance. Europarl
(unverified) follows the same shape, 4.2–23.6% raw to 41.5–66.9% `+D`, with Romance languages above
Germanic ones. The shift is the dominant lever in every table.

**Caveats.**

- FLORES numbers belong to the 2009-row CSV (`196bbd3`). On the shipped 2006-row CSV the same protocol
  gives 1009 devtest, chance 505.0, and every cell within 0.1 pp of the table above (mean: 38.6–60.6%
  raw, 83.8–90.0% `+D`, MRR `+D` 0.882–0.931).
- One sample per language, no CIs. Agg devtest is only 280 queries.
- `D` is one constant per direction; per-sentence shift variance is the residual error and is not
  modelled.
- Europarl has no surviving embeddings, md, JSON or log. Only the sampled CSV is shipped, and its `it`
  block came from a legacy run whose per-language CSV is also gone.
- The Europarl `smoke` mode cannot run (legacy `europarl_it_en.csv` and `europarl_embeddings/` are
  absent), so there is no model-free regression check for that pipeline.

---

## E40 — Legacy pre-E18 XGBoost suite (FLORES topic / SST-2 / emotion) — SUPERSEDED (audited 2026-10-20)

**Script to reproduce:** NOT AVAILABLE for the published result tables — the scripts exist but their
JSON/MD outputs were deleted and not retained. **(UNK)** Partial re-runs are possible; see the table.

> **Recoverable, not retained.** The working-tree copies are gone (deleted 2026-10-04 in commits
> `c594b36` and `f768fb8`), but the published tables survive in git at **`aa3f6f6`**
> (`xgboost_perlang_lenient.md`, `xgboost_topic_results_lenient_{mean,bos,eos,lead}.md`,
> `xgboost_optuna_trials_lenient_{mode}.csv`, `xgboost_perlang_lenient_{lang}.json`). Those archived
> files are the *reference* for this entry, not a re-run.

**Question.** Before the single-model cross-lingual proof existed, what did a stock XGBoost on
EuroBERT-210m embeddings score on (a) FLORES 8-tag topic classification (multi-label, EN + it/de/fr,
with and without the `+D` language shift), (b) SST-2 sentiment, (c) 6-class emotion — and which
pooling/label/threshold protocol was in use?

**Script / outputs.**

| script | purpose | expected artifact | present? | can it be re-run cheaply? |
|---|---|---|---|---|
| `xgboost_topic_optuna.py` | FLORES topic sweep: 4 poolings × 4 languages, lenient labels, per-tag thresholds, 50 TPE trials/cell + `+D` transfer | `xgboost_topic_results{_lenient}_{mode}.md`, `xgboost_optuna_trials*_{mode}.csv`, `xgboost_perlang{_lenient}.md`, `..._trials_{lang}_{mode}.csv` | **no** (deleted; archived at `aa3f6f6`) | **partly**: retrain/eval from archived best params ≈5 s/cell; full grid 16 cells ≈3.4 h |
| `xgboost_infer.py` | standalone inference; embedded per-mode params, `--shift`, `--eval`, `--input` | `predictions_{transfer,perlang}_{mode}_{lang}[_shift].csv` | **no** (git-ignored) | **no on current data** (aborts at `len(df)==2009`); yes with the `aa3f6f6` CSV, ~30 s |
| `xgboost_sst2.py` | SST-2 binary, `mean`/`bos`, 50 trials (val-accuracy objective) | `xgboost_sst2_results_{mean+bos}.{json,md}`, `xgboost_sst2_trials_{mode}.csv`, `sst2_emb/` | **trials + emb cache yes; results no** | **yes**: one refit 142 s; full 50-trial × 2 modes ≈ hours |
| `xgboost_emotion.py` | 6-class emotion, `mean`/`bos`, 50 trials (val AUC objective) + per-emotion report | `xgboost_emotion_results_{mean+bos}.{json,md}`, `xgboost_emotion_trials_{mode}.csv`, `emotion_emb/` | **trials + emb cache yes; results no** | **yes**: one refit 90 s; full run ≈ hours |
| `xgboost_flores_lang.py` | EN/IT per-language reference + top-k η²-dim zeroing ablation | `xgboost_flores_lang.{md,json}` | **no** | **yes** ~1–2 min |
| `xgboost_flores_combined.py` | EN+IT combined training; EN→IT raw vs `+D`, in-sample tuned thresholds | `xgboost_flores_combined.{md,json}` | **no** | **yes** ~1 min |
| `xgboost_flores_matrix.py` | full 2×2 train/eval matrix + dim-zeroed EN→IT | `xgboost_flores_matrix.{md,json}` | **no** | **yes** ~1–2 min |
| `xgboost_flores_drop_optuna.py` | 4 cells (en/it × mean/bos) × 50 TPE trials with top-50% η² dims zeroed; saves `models/` bundles | status/log | **no** | **no** (~1–2 h); only `k` is recomputable in seconds |

**Inputs / split (legacy convention — differs from the current one).**

| item | value |
|---|---|
| data | `flores200_en_it_de_fr.csv` **as of `aa3f6f6`**: 2009 rows (997 `dev` / 1012 `devtest`). The current repo CSV is 2006 rows and its `topic` strings were re-canonicalised on 2026-10-05 |
| split | `dev` 997 / `devtest` **506** / `test` **506** = seed-0 permutation of the 1012 devtest rows |
| embeddings | `embeddings/{mean,bos,eos,lead}/emb_{en,it,de,fr}.npy`, 2009×768 |
| labels | 8 canonical topic tags, **lenient** alias mapping; 515/2009 all-zero rows (the script's own target constant, `AUDIT_LENIENT_ZERO_ROWS`, still says 504) |
| thresholds | one per tag, grid 0.05–0.95, tuned **in-sample** on the model's own training rows |
| XGBoost | one binary `binary:logistic` per tag, `hist`, `max_bin=64`, `n_jobs=2`, seed 0; 50-trial TPE seed 0 |
| SST-2 / emotion | `sst2.csv` 54,577 / 6,822 / 6,822 (80/10/10 re-split); `emotion.csv` 16,000 / 2,000 / 2,000 |

**Protocol notes that matter.**

- `--mode perlang` / the per-language grid tune on **devtest** and report **test** — clean.
- `--mode all` / `--mode <pooling>` tune the Optuna objective on **`test` macro-F1** and then report
  that same test set; the per-mode tables are *selection-set* scores.
- The transfer arm trains on all 2009 English rows (test included) and reports `en (in-sample)
  1.0000`; that row is an upper bound, not a baseline.

**Results — retained numbers, with verification status.**

Per-language × pooling, tuned **test** macro-F1 (`devtest → test`), from the archived `aa3f6f6`
`xgboost_perlang_lenient.md` = the current `README.md:198-203` table verbatim:

| language | mean | bos | eos | lead |
|---|---:|---:|---:|---:|
| en | 0.416 → **0.528** | 0.305 → 0.298 | 0.348 → 0.370 | 0.272 → 0.341 |
| it | 0.344 → **0.464** | 0.226 → 0.352 | 0.314 → 0.352 | 0.171 → 0.222 |
| de | 0.392 → **0.470** | 0.283 → 0.322 | 0.304 → 0.344 | 0.202 → 0.221 |
| fr | 0.411 → **0.503** | 0.248 → 0.251 | 0.281 → 0.365 | 0.235 → 0.246 |

- **`mean` column: VERIFIED** by retraining the archived best params (en/it/de): 0.5281 / 0.4640 /
  0.4702, subset acc 0.5455 / 0.4387 / 0.4664 — exact to 4 dp. `fr/mean` and every `bos`/`eos`/`lead`
  cell: **UNVERIFIED** (not re-run).
- The Optuna *search* itself was not re-run anywhere, so the best-param selection is unverified; only
  the retrain-and-score stage is.

EN `mean` per-mode table (archived `xgboost_topic_results_lenient_mean.md`):

| model | subset acc | macro-F1 | micro-F1 |
|---|---:|---:|---:|
| baseline @0.5 | 0.4901 | 0.4038 | 0.4772 |
| baseline tuned | 0.5277 | 0.5550 | 0.6127 |
| best params @0.5 | 0.5257 | 0.5049 | 0.5549 |
| best params tuned | 0.5198 | 0.5337 | 0.5941 |

- **VERIFIED in full** (4 rows × 3 metrics, exact) plus the transfer rows `it raw` 0.4032/0.2784/0.4402
  and `it+D` 0.4490/0.3450/0.4852 and the 8 tuned thresholds. Label counts also reproduce exactly
  (travel 518, sports 148, crime and law 121, politics 177, health 81, geography 117, science 317,
  safety 151; all-zero 515/2009).

Strict-label legacy table (`README.md:248-253`): **UNVERIFIED and currently unreproducible**. Its
source is archived (`aa3f6f6:xgboost_topic_allmodes.md`, character-for-character the README rows), but
`load_data("strict")` now aborts (`a tag has 0 positives in devtest`) before any fit, and that run's
Optuna objective was the test split itself.

SST-2 and emotion (`README.md:265-279`):

| task | mode | val | test acc | test macro-F1 | test AUC |
|---|---|---|---:|---:|---:|
| SST-2 | mean | acc 0.8147 | 0.8080 | 0.8052 | 0.8943 |
| SST-2 | bos | acc 0.7366 | 0.7382 | 0.7338 | 0.8144 |
| emotion | mean | AUC 0.8207 | 0.5635 | 0.3740 | 0.8196 |
| emotion | bos | AUC 0.7264 | 0.4780 | 0.2478 | 0.7241 |

- **`mean` rows: VERIFIED exactly** by refitting the best trial from the surviving trial CSVs on the
  cached embeddings (SST-2 142 s, emotion 90 s). **`bos` rows: UNVERIFIED.**
- The `val` columns are exactly the best objective values in the surviving trial CSVs
  (0.814717 / 0.736588 / 0.820689 / 0.726391), i.e. README val numbers are the trial log, not a
  separate run.

Language-dim ablation (README:216-223): en/mean 0.5093, en/bos 0.3585, it/mean 0.4773, it/bos 0.2911
and `k`=237 (mean) / 211 (bos) — **numbers UNVERIFIED** (status JSON and bundles deleted); **`k`
VERIFIED** by recomputing `lang_metrics.top_dims` (237 / 211).

**0.975 bar: N/A.** No retained target/English ratio exists for this suite. The only English reference
in the transfer arm is trained on all 2009 rows *including* the test rows (in-sample 1.0000), and the
per-language grid compares in-language models, not EN→target transfer. The bar cannot be applied, and
no ratio should be reconstructed from these tables.

**Verdict.** This suite is the pre-E18 baseline: it established that `mean` pooling beats
`bos`/`eos`/`lead`, that lenient labels plus per-tag thresholds lift macro-F1 (0.4038 → 0.5337 on EN
test), and that `+D` helps EN→IT/DE/FR on multi-label macro-F1 (0.2784 → 0.3450 IT). It is
**superseded**: the cross-lingual claim is now made by E18, the classifier comparison by E21, the
backbone/grid by E28, the train×test matrices by E25, and the sentiment/emotion cross-lingual line by E4
(script missing). Nothing here should be quoted as evidence in the current report.

**Caveats.**

- Two split conventions coexist: legacy 997 / 506 / 506 (this suite) vs 997 / 504 / 505
  (`train_flores12.py`, used by E18/E21/E42). Numbers are not interchangeable.
- Single seed, no CIs; test = 506 rows, 8 rare tags.
- The per-language grid thresholds are tuned in-sample on dev+devtest, so `devtest → test` gaps are
  selection optimism, not generalisation.
- All four `sst2`/`emotion` trial logs are COMPLETE, no NaN; the `mean` models were re-fitted from those
  logs only.
- Re-runs require the **`aa3f6f6`** CSV, not just any 2009-row file: `196bbd3` has 2009 rows but the
  relabelled `topic` strings, which change the lenient targets (469 all-zero rows vs 515). Neither is
  in the worktree; the current 2006-row CSV aborts every entry point at the row-count assert.

---

# PART E — Recall-first and toxicity

The operating-point half of the programme. At a fixed high recall, does the English model spend more
false-alarm budget than the native one (E15)? At recall 0.90, how much precision does it lose on
Italian (E27, E37)? And what did the pre-clean, leaky toxicity run say before E20 replaced it (E36)?

## E15 — English vs native at high recall — DONE (audited 2026-10-20)

**Script to reproduce:** `imbalanced_recall.py` / `imbalanced_recall_mel.py` — `python
imbalanced_recall.py` and `python imbalanced_recall_mel.py --launch --n-rare 8 --device auto`.
Reproduction status: **REPRODUCIBLE**.

**Question.** On highly unbalanced detection (rare positives, misses unacceptable), how much more
false-alarm budget does an English-trained detector spend than a natively trained one at the same high
recall?

**Script / outputs.**

- `imbalanced_recall.py` -> `imbalanced_recall.json` (24 cells: `sent` 12 + `agg` 12), `.md`,
  `imbalanced_pr_sent.png`, `imbalanced_pr_agg.png`, `_status.json`.
- `imbalanced_recall_mel.py` -> `imbalanced_recall_mel_v2.json` (16 cells: 8 concepts × de/fr), `.md`,
  `imbalanced_pr_mel_v2.png`, `.pid`, `_v2_status.json`.
- `VERSION = "v2"` is hardcoded in the mel driver; it cannot regenerate v1.
- Stored runs: FLORES 29.4 s (pid 31244, 2026-10-11 22:44), 24 cells. MultiEURLEX v2 467.1 s (pid 8752,
  2026-10-12 05:08, CPU), 16 cells.

**Inputs.**

| dataset | embeddings | labels / split | models |
|---|---|---|---|
| sent | `embeddings/mean/emb_{en,it,de,fr}.npy`, 2009 rows sliced to 2006 kept | `flores200_en_it_de_fr.csv`; 12 classes; 997/504/505; rows parallel across languages | 1 EN model vs 3 native |
| agg | `embeddings_agg/mean/emb_{en,it,de,fr}.npy`, 562x768 | `flores_agg.csv`; 11 classes (business dropped); 276/140/139; rows parallel | 1 EN model vs 3 native |
| mel | `multieurlex/emb_mean_{en,de,fr}_{train,test}.npy`, 11000/5000x768 | `multieurlex/{en,de,fr}_{train,test}.jsonl`, `multi_hot` (21); train 11000, test 5000; validation 1000 unused | 1 EN model vs 2 native |

MultiEURLEX: mean pool, 512-token truncation, batch 64, EuroBERT-210m. Train splits are NOT parallel
across languages; the test split is (verified: all 5000 `multi_hot` rows identical for en/de/fr;
10831/11000 de and 10817/11000 fr train rows differ from en). v2 therefore trains each native model on
its own language's train labels; the shared test labels score every model.

**Protocol.**

- Detector: one-vs-rest `XGBClassifier(binary:logistic, hist, max_bin=64, device=cpu, n_jobs=2,
  seed=0, max_depth=8, min_child_weight=5, lr=0.1, n_estimators=400)`.
- Features: mean embedding -> `l2_thenZ [both]` (L2, then per-language TRAIN z-score; each language
  uses its own train statistics, so no target labels are used).
- English-trained = fit EN train, score target test. Native = fit target train on the target's OWN
  labels, score target test. Both scored on the same (parallel) test labels.
- Rare class definition. `sent`: 4 rarest of 12 topics by total count: business and economy (25),
  health (68), geography (87), safety (97); train positives 18/32/44/64. `agg`: health, geography,
  safety, education and society (the 4 rarest of the 11 kept classes; test positives 4/4/5/7). `mel`:
  the 8 lowest EN-TRAIN-frequency level-1 EuroVoc concepts present in test, stable sort, lowest index
  on ties: 13 international organisations (108), 16 science (180), 14 energy (221), 9 employment (341),
  11 law (391), 10 politics (408), 20 environment (552), 8 transport (582).
- Splits / seeds. FLORES: train = 997 `dev` rows; `devtest` (1009) permuted with seed 0, first 504 val,
  last 505 test; val is unused. agg: same rule by URL, then business rows dropped -> 276/140/139.
  MultiEURLEX: official train/test, no re-split. XGBoost seed 0. No other randomness.
- Metric: `fpr_for_recall_90` = false-positive rate needed to reach recall = 0.90 (`RECALL_TARGET`).
  **Lower is better.** Headline = unweighted mean over cells (24 for FLORES, 16 for mel).
- Threshold selection: none. The operating point is the oracle threshold on the test ranking; no
  validation split, calibration, or held-out threshold is used. Ranking metrics (AP, R@FPR, R@FA) need
  no threshold; FPR@R90 is optimistic relative to a fixed deployed threshold.
- 0.975 rule: English FPR <= native FPR / 0.975, i.e. EN/native <= 1/0.975 = **1.025641**.

**Results.**

| dataset | cells | EN FPR@R90 | native FPR@R90 | EN - native | EN/native | bound | at 0.975 |
|---|---:|---:|---:|---:|---:|---:|:--:|
| FLORES (sent+agg) | 24 | 0.5429 (0.543) | 0.5404 (0.540) | +0.0026 | **1.005** | 1.0256 | **PASS** |
| MultiEURLEX v2 | 16 | 0.4264 (0.426) | 0.3566 (0.357) | +0.0699 | **1.196** | 1.0256 | **FAIL** |

Component means (same rule):

| dataset | cells | EN | native | ratio | at 0.975 | cells passing individually |
|---|---:|---:|---:|---:|:--:|---:|
| sent | 12 | 0.4740 | 0.4638 | 1.022 | PASS (narrow) | 4/12 |
| agg | 12 | 0.6119 | 0.6170 | 0.992 | PASS | 8/12 |
| mel v2 | 16 | 0.4264 | 0.3566 | 1.196 | FAIL | 2/16 |

Rounded-value ratios: FLORES 0.543/0.540 = **1.006** (pass), MultiEURLEX 0.426/0.357 = **1.193**
(fail). Exact unrounded means give 1.0047 and 1.1959; both confirm the same verdicts.

**Verdict.**

- **FLORES: PASS at 0.975.** The English model spends 0.543 vs 0.540 FPR at 90% recall (+0.3%). The
  pass is marginal in coverage: only 12/24 cells pass individually. `sent` passes narrowly (1.022 vs
  1.026); `agg` passes (0.992) and carries the mean. Health and geography fail on all or most targets.
- **MultiEURLEX: FAIL at 0.975.** English spends 0.426 vs 0.357 (+19.6% relative); only 2/16 cells pass
  (science/de 0.516, energy/fr 0.989). English is worse in 14/16 cells.
- The E15 sentence "the English detector spends ~the same false-alarm budget as a native model" is
  supported by FLORES only. It is **not** supported by the MultiEURLEX row of the same table.

**Bug history (v1 -> v2).** v1 trained the native de/fr models with the ENGLISH train labels even though
the train splits are not parallel. v2 loads per-language train labels. A simulation of the v1 behaviour
(concept 13, DE/FR features + EN labels) gives native AP 0.016/0.031 and FPR@R90 0.964/0.866, against
corrected 0.145/0.143 and 0.484/0.309; the English model (0.508/0.422) then looks far better, matching
the reported "English >> native" v1 result. The v1 outputs (`imbalanced_recall_mel.json`, `.md`,
`imbalanced_pr_mel.png`) are **missing**; only the orphan `imbalanced_recall_mel_status.json` and
`imbalanced_recall_mel.pid` remain. v1 is not reconstructible from the repo.

**Caveats.**

- Unweighted cell means. FLORES cells have very few positives: `sent` test positives 4–20 (negatives
  ~485); `agg` 4–7 positives (negatives 132–135). FPR granularity is ~1/132 in agg.
- `agg` / education and society saturates: with 7/7 positives required, both models report FPR@R90
  0.9242 in all three languages (6/24 FLORES cells). The value is uninformative there.
- Single seed, no CIs, no repeated splits. FLORES test rows: 505 (sent), 139 (agg).
- Tie handling: `fpr_for_recall` breaks score ties by row order (oracle tie-break). Under threshold
  semantics (all ties above/below the cut), the agg education cells move 0.9242 -> 1.000 (max shift
  0.076 on 6/24 FLORES cells); mel cells checked show no material shift.
- MultiEURLEX embeddings truncate long documents to the first 512 tokens (mean document 6.7k–13k
  characters), so "document-level" detection is really "first-512-token detection".
- v1 and v2 also differ in device (v1 CUDA, v2 CPU), not only in labels.
- Re-running `imbalanced_recall.py` overwrites the published FLORES JSON/MD/PNGs (no versioning).

---

## E27 — Toxicity precision/recall, XGBoost (two pipelines) — PARTIAL (audited 2026-10-20)

**Script to reproduce:** `toxi_pr.py` / `toxi_pr_all.py` — `python toxi_pr.py`;
`python toxi_pr_all.py` (both from `C:\Users\pedro\Documents\random_projects\multi`). Reproduction
status: **PARTIAL**.

**Question.** At recall 0.90, how much precision does an English toxicity detector lose on Italian test
rows relative to the Italian-native detector? Does EuroBERT-610m close the gap? The E27 table answers
this for two backbones. It does not state a pass bar.

**Split of the old wildcard.** `toxi_pr*.py` used to be documented as one wildcard; it covers three
experiments. They are reported separately here.

| id | experiment | script | outputs | seeds |
|---|---|---|---|---:|
| E27 | XGBoost P/R 2×2, per-language z-scored features | `toxi_pr.py` | `toxi_pr.json` (the `.md/.png` this script writes are currently overwritten) | 0/1/2 |
| E27 | XGBoost P/R 2×2, both backbones, raw features, fresh fit | `toxi_pr_all.py` | `toxi_pr_all.json` + regenerated `toxi_pr.md/.png` | 0 |
| **E37** | Seven classifiers, P/R vs decision-score percentile | `toxi_pr_models.py` | `toxi_pr_models.json/.md` + 7 per-classifier PNGs + 3 grids + 2 smoke files | 0 |
| — | Figure/markdown provenance | `toxi_pr_all.py` overwrites `toxi_pr.png/.md`; `toxi_pr_610.png` comes from `backbone_grid.py` (E28) | — | — |

**Inputs.** `data/toxi_clean/{en,it}_{train,validation,test}.parquet`, label `is_toxic`. EN test 3000
rows, 363 positive (12.10%). IT test 2310 rows, 678 positive (29.35%). Embeddings
`emb_bf16/toxic/{210m,610m}/emb_<lang>_<split>.npy`, masked mean, 768/1152-d. `toxi_pr.py` uses
per-language z-score fit on the test language's own train rows and the persisted
`models_toxi/{en,it}_seed{0,1,2}.ubj`. `toxi_pr_all.py` uses raw mean embeddings and a fresh XGBoost
fit (depth 8, 400 trees, lr 0.1, hist, max_bin 64, n_jobs 2). The two E27 pipelines are not
interchangeable. Do not mix their numbers in one column.

**Results — the E27 table versus committed JSON.** The E27 cells are `precision @ recall >= 0.90`.

| cell | E27 210m | `toxi_pr_all.json` 210m | diff | E27 610m | `toxi_pr_all.json` 610m | diff | E27 Δ | JSON Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EN model / IT test | 0.324 | 0.318395 | +0.005605 | 0.359 | 0.360047 | −0.001047 | +0.035 | +0.041652 |
| IT model / IT test | 0.448 | 0.449595 | −0.001595 | 0.460 | 0.467125 | −0.007125 | +0.012 | +0.017530 |
| EN model / EN test | 0.276 | 0.270920 | +0.005080 | 0.278 | 0.287346 | −0.009346 | +0.002 | +0.016427 |

`toxi_pr.json` (per-language z-score, three seeds, 210m only) gives 0.319230 EN/IT, 0.440405 IT/IT,
0.272147 EN/EN. No committed artifact contains 0.324 / 0.448 / 0.276 or 0.359 / 0.460 / 0.278. The E27
deltas are all wrong: EN/EN is +0.016, not +0.002. The E27 table is hand-built; its cells are orphaned.

**0.975 bar.** The table does not state a bar. Applying the E18 parity bar (target/native >= 0.975) to
the only valid transfer comparison, EN-on-IT versus IT-on-IT on the same Italian test rows:

| backbone | E27 EN/IT | E27 IT/IT | E27 ratio | JSON EN/IT | JSON IT/IT | JSON ratio | pass |
|---|---:|---:|---:|---:|---:|---:|:--:|
| 210m | 0.324 | 0.448 | **0.7232** | 0.318395 | 0.449595 | **0.7082** | no |
| 610m | 0.359 | 0.460 | **0.7804** | 0.360047 | 0.467125 | **0.7708** | no |

Reverse direction, IT-on-EN versus EN-on-EN: 210m 0.194991 / 0.270920 = **0.7197**; 610m 0.226611 /
0.287346 = **0.7886**. Both fail. At 0.975, **0/2 backbones pass, 0/4 directions pass**. To pass, 210m
EN/IT would need 0.4384 (actual 0.3184, short 0.1200); 610m would need 0.4554 (actual 0.3600, short
0.0954). Using the E27 table, the deficits are 0.1128 and 0.0895.

The EN/EN row is not a transfer comparison; a self-ratio is 1 by construction. Its only cross-backbone
ratio is the 210m-to-610m control: 0.270920 / 0.287346 = 0.943 (table 0.276 / 0.278 = 0.993, i.e. "~1"
at the table's rounding). It says nothing about English-to-Italian transfer. Comparing EN/EN with IT/IT
would mix two different test sets with 12.1% versus 29.35% prevalence, which changes precision at fixed
recall.

**Provenance.**

- `toxi_pr.md/.png` are from `toxi_pr_all.py` (raw features); `toxi_pr.json` is from `toxi_pr.py`
  (per-language z-score). Three pipelines write the same `toxi_pr` names.
- `toxi_pr_all.py:366` writes `toxi_pr.md` only if the file already exists, so a clean checkout cannot
  regenerate the markdown; the script also overwrites `toxi_pr.json` despite a create-only claim.
- `toxi_pr.py:251` silently skips missing seed models: a cell can be computed from fewer than three
  seeds, or zero, without an error. `toxi_pr.py:275` checks only row counts against the embedding cache.
- `toxi_pr_610.png` is an E28 side output of `backbone_grid.py:89,862` (610m, per-language z-score). It
  is not an E27 artifact and should not be cited under E27.
- `toxi_pr_models.json` stamps `generated = 2026-10-13 14:35:44`, file mtime 14:38:17, script mtime
  14:37:55. The numbers predate the current script revision; only the figures reflect it. See E37.

**Verdict.** At the **0.975** bar, both E27 transfer cells fail. EN/IT versus IT/IT is **0.7232** (210m)
and **0.7804** (610m) from the E27 table, or **0.7082** and **0.7708** from the committed JSON. The
reverse direction fails at **0.7197** and **0.7886**. EN/EN is a backbone control (0.276 to 0.278, ratio
1.007), not a transfer comparison. E27's qualitative finding survives: the English model is below
native on Italian, and 610m helps. Its exact cells, deltas, and provenance do not survive.

**Caveats.**

- E27's cells are single-seed for `toxi_pr_all.py` (seed 0). `toxi_pr.py` has three seeds but only 210m.
  No CIs.
- EN and IT test sets differ in size and prevalence. Only same-test comparisons (EN/IT versus IT/IT) are
  valid for the 0.975 parity bar.
- P@R90 uses "first threshold whose recall reaches >= 0.90". Observed recall is 0.9008 (EN) and 0.9012
  (IT), so the operating point is close but not exactly 0.90.
- E27's "threshold is miscalibrated" is a threshold-shift observation, not a calibration result. At the
  model's own validation threshold, EN-on-IT recall is 0.125 (210m) and 0.474 (610m), versus 0.673 and
  0.690 for IT-on-IT. No Brier score, ECE, or reliability curve is reported.
- "610m helps" holds for all three cells, but IT/IT AUC is flat (0.8420 to 0.8430). The gain is mostly
  cross-lingual; the EN/IT gap narrows from 0.1312 to 0.1071, still far from 0.975.
- The raw-probability axis in `toxi_pr_all.py` is compressed: all P@R90 thresholds are below 0.031 on a
  0..1 axis, so the reported R90 point is not visible in `toxi_pr.png`. See E37 for the percentile-axis
  fix.
- `requirements.txt` pins no versions. The run emitted a SciPy/NumPy version warning from XGBoost.

---

## E37 — Seven classifiers at the 0.975 toxicity-parity bar — DONE (verified 2026-10-20)

**Script to reproduce:** `toxi_pr_models.py` / `toxi_pr_all.py` — `python toxi_pr_models.py` (full
7 × 2 refit; also accepts `--figures-only`, `--only`, `--backbone`, `--smoke`), `python toxi_pr_all.py`
(XGBoost reference), both run from `C:\Users\pedro\Documents\random_projects\multi`. Reproduction
status: **PARTIAL**.

**Question.** At recall 0.90, does any standard classifier, on either EuroBERT backbone, bring
English-model-on-Italian-test precision within 2.5% of the Italian-native detector (ratio >= 0.975)?
And does the percentile sweep axis make the seven-classifier comparison valid?

**Scope.** Seven classifiers × two backbones × three cells (`EN model / EN test`, `EN model / IT test`,
`IT model / IT test`) = 42 cells. Fourteen transfer comparisons (`EN/IT ÷ IT/IT`). Seed 0. This is the
seven-classifier experiment named in the old E27 wildcard; `EXPERIMENTS.md` has no E37 section.

**Inputs.**

| item | value |
|---|---|
| data | `data/toxi_clean/{en,it}_{train,validation,test}.parquet`, label `is_toxic` |
| splits | EN 24000 / 3000 / 3000 (363 test positives, 12.10%); IT 18472 / 2310 / 2310 (678, 29.35%) |
| embeddings | `emb_bf16/toxic/{210m,610m}/emb_<lang>_<split>.npy`, mean-pooled float32, 768 / 1152-d |
| transform | none. RAW mean vectors. No L2, no z-score, no `perlang_z` |

**Classifiers and scores.** All scores are monotone in toxicity and (except KNN) continuous.

| model | fit | score |
|---|---|---|
| XGBoost | depth 8, 400 trees, lr 0.1, hist, max_bin 64, mcw 5, seed 0 | `predict_proba[:,1]` |
| SVC_linear | `LinearSVC(C=1, squared_hinge, dual=False, max_iter=20000)` | `decision_function` |
| MLPClassifier | sklearn defaults, hidden (100,), max_iter 300 | `predict_proba[:,1]` |
| LogisticRegression | max_iter 2000 | `decision_function` |
| RandomForestClassifier | 200 trees, max_features sqrt | `predict_proba[:,1]` |
| KNeighborsClassifier | k=5 | distance-weighted positive fraction over the 5 NN |
| GaussianNB | sklearn defaults | log-likelihood ratio |

XGBoost uses the exact `toxi_pr_all.py` recipe. `SVC_linear` is the liblinear primal SVM because libsvm
`SVC(kernel='linear')` did not converge at 24k × 768 (>11 min/fit).

**Protocol.** Sweep axis = decision-score percentile q = 0..100 (101 points), threshold at q =
`np.percentile(score, q)`. Precision at recall 0.90 = the smallest threshold whose recall first reaches
>= 0.90 on the test cell (`at_recall`, ranking-based). Validation threshold = F1-optimal on the model's
own-language validation split, searched over that split's percentiles. CPU, 2 threads.

**Results.** P@R90 at 3 dp (matches `toxi_pr_models.json` 14/14 rows at 3 dp).

| model | bb | AUC EN/EN | AUC EN/IT | AUC IT/IT | P EN/EN | P EN/IT | P IT/IT | EN/IT ÷ IT/IT |
|---|---|---|---|---|---|---|---:|
| XGBoost | 210m | 0.907 | 0.621 | 0.842 | 0.271 | 0.318 | 0.450 | 0.7082 |
| XGBoost | 610m | 0.913 | 0.715 | 0.843 | 0.287 | 0.360 | 0.467 | 0.7708 |
| SVC_linear | 210m | 0.896 | 0.633 | 0.823 | 0.264 | 0.321 | 0.442 | 0.7256 |
| SVC_linear | 610m | 0.906 | 0.706 | 0.822 | 0.299 | 0.341 | 0.442 | 0.7713 |
| MLPClassifier | 210m | 0.882 | 0.572 | 0.812 | 0.253 | 0.307 | 0.430 | 0.7136 |
| MLPClassifier | 610m | 0.892 | 0.693 | 0.822 | 0.265 | 0.346 | 0.454 | 0.7616 |
| LogisticRegression | 210m | 0.897 | 0.635 | 0.823 | 0.257 | 0.323 | 0.443 | 0.7295 |
| LogisticRegression | 610m | 0.905 | 0.711 | 0.823 | 0.298 | 0.346 | 0.444 | 0.7806 |
| RandomForestClassifier | 210m | 0.856 | 0.686 | 0.804 | 0.189 | 0.333 | 0.394 | 0.8452 |
| RandomForestClassifier | 610m | 0.845 | 0.692 | 0.795 | 0.189 | 0.346 | 0.398 | 0.8682 |
| KNeighborsClassifier | 210m | 0.810 | 0.454 | 0.777 | 0.121* | 0.294* | 0.414 | 0.7095* |
| KNeighborsClassifier | 610m | 0.795 | 0.648 | 0.782 | 0.121* | 0.294* | 0.418 | 0.7028* |
| GaussianNB | 210m | 0.851 | 0.663 | 0.729 | 0.235 | 0.329 | 0.344 | 0.9559 |
| GaussianNB | 610m | 0.811 | 0.679 | 0.705 | 0.221 | 0.346 | 0.324 | **1.0675** |

`*` KNN's EN-model cells are degenerate. `at_recall` lands on threshold 0.000, which predicts every
row: recall 1.000, precision = the test base rate (0.121000 EN, 0.293506 IT). About 52% (210m) and 43%
(610m) of EN-on-IT test rows score exactly 0 (all five neighbours non-toxic), so no threshold attains
recall 0.90: the recall curve falls from 1.000 to 0.386 (210m) or 0.727 (610m) in one q step. The
identical KNN EN/IT values across backbones, and `IT/EN = EN/EN = 0.121`, are this artifact.

**0.975 bar.** Applying target/native >= 0.975 to EN-on-IT versus IT-on-IT (same Italian test rows):
**13 of 14 fail.** The only nominal pass is GaussianNB 610m at **1.0675** (EN/IT 0.3464 > IT/IT
0.3245). That pass is not parity. GNB has the weakest Italian-native baseline on 610m (P@R90 0.3245,
AUC 0.705, both lowest of the seven). Against the next-weakest native baseline (KNN 610m, 0.4176) the
same cell is 0.8294, a fail. The best non-degenerate ratios are RandomForest 0.8452 (210m) and 0.8682
(610m); XGBoost is 0.7082 / 0.7708. To pass, XGBoost would need P 0.4384 / 0.4554 (short 0.1200 /
0.0954); RandomForest would need 0.3841 / 0.3885 (short 0.0512 / 0.0426); GNB 210m misses by 0.0066.
At the old 0.90 bar only the two GNB cells pass (0.9559, 1.0675). Reverse direction (IT/EN ÷ EN/EN)
fails everywhere except the two degenerate KNN cells (1.0000).

**Percentile-axis fix — verified.** `QS = 0..100` (`toxi_pr_models.py:134`), all curve thresholds from
`np.percentile` (`sweep_pct`, line 293). All 42 curves have 101 points, no nulls, and zero
recall-monotonicity violations. Distinct score counts: XGB/SVC/LogReg = test size (3000 / 2310); MLP
2202–2955; GNB 3000 / 2310 (LLR spread, e.g. thresholds 19.04 and -41.03); KNN 1093–1439; RF 104–155. A
raw 0..1 probability axis would have collapsed GNB (saturated probabilities, F1-opt val threshold
0.99–1.00) and KNN (6 discrete values). The fix is real and necessary. It fixes the figures; it does
not fix the tie problem in `at_recall`.

**XGBoost reference — verified.** The XGBoost rows equal `toxi_pr_all.json` exactly on AUC, P@R90,
recall and the R90 threshold in all 6 cells and both directions (independent refits). Validation
thresholds differ only by search rule: 210m EN 0.149348 vs 0.15, 210m IT 0.222012 vs 0.22, 610m EN
0.115547 vs 0.14, 610m IT 0.187634 vs 0.18. The E27 table's 0.324 / 0.448 / 0.276 match neither JSON.

**Provenance — mixed run.** The committed JSON is not the output of one 7-model run.

- Last run: PID 5324 at 14:33:54, log line `models=['XGBoost']`, `ALL DONE in 119.0 s` (14:35:52). Only
  XGBoost was refit; the other six were merged from the previous JSON (`existing` / `merged_snapshot`,
  `toxi_pr_models.py:757-769`).
- `toxi_pr_models.json` stamps `generated = 2026-10-13 14:35:44`, file mtime 14:38:17. Script mtime
  14:37:55; committed as `0a16d79` at 14:38:58.
- The 10 PNGs were re-plotted at 14:38:00–06 by a `--figures-only` pass and reproduce **byte-identically**
  from the committed JSON with the current script (10/10 SHA-256).
- JSON and MD were re-written at 14:38:17 without retraining, keeping the 14:35:44 stamp.

So the numbers predate the current revision (the six non-XGBoost rows by an unrecorded earlier run;
XGBoost by ~2 min). Figures and numbers agree with each other; the numbers were not produced by the
script revision now on disk as a single run.

**Verdict.** At the **0.975** bar, **13/14 cells fail; 1/14 nominal pass (GaussianNB, 610m, 1.0675)**,
and that pass rests on the weakest native baseline in the table. Six of seven classifiers are far below
parity (best 0.8682). The percentile-axis fix is present and works; the KNN row needs a tie-aware
P@R90 before it can be quoted. Reproduction status: **PARTIAL** (GNB and KNN refit exactly; the other
four classifiers were not refit; the JSON is a merged, older artifact).

**Caveats.**

- Single seed, no CIs. Test sizes: EN 3000 rows / 363 positives, IT 2310 / 678.
- EN and IT test sets differ in size and prevalence. Only same-test comparisons (EN/IT vs IT/IT) are
  valid for the 0.975 bar. EN/EN is a backbone control, not a transfer cell.
- P@R90 overshoots 0.90 to 0.9008 (EN) / 0.9012 (IT). KNN overshoots to 1.000.
- "Closest EN/IT to IT/IT" in `toxi_pr_models.md` ranks by absolute gap, which rewards a weak native
  model. GNB 210m (gap 0.015) is not evidence of better transfer.
- `toxi_pr_models.py:266-277` picks the first percentile on an F1 plateau (strict `>`); RF and KNN have
  many ties. `toxi_pr_models.py:318` uses `mean(s <= v)` for the marker while the curve uses
  `np.percentile(s, q)`; they are not exact inverses.
- `toxi_pr_models.py:426` loops over both backbones, but `main` passes only the selected backbones at
  line 866. `--backbone 210m` makes every per-model figure fail with `KeyError('610m')`; the exception
  is caught and only logged.
- Smoke mode writes `toxi_pr_models.json.smoke` and `toxi_pr_models_smoke.json` into the repo, shares
  the status file, and writes no PID; both smoke files contain 6 models, no XGBoost, 200-row splits.

---

## E36 — Toxicity on the ORIGINAL leaky `toxi-text-3M` splits — PROVENANCE ONLY, DO NOT QUOTE

**Script to reproduce:** `toxi_proof.py` — `python toxi_proof.py` (detached:
`python toxi_proof.py --launch`). Reproduction status: **PARTIAL** (test composition, pool sizes and
every `leak_removed` count re-derived exactly; the stored AUCs were not re-derived because the
embedding cache is empty and no model was saved).

**Question.** Does ONE English-trained XGBoost on EuroBERT-210m mean features match a native
per-language model on binary toxicity, using the **shipped** `toxi-text-3M` train/test splits without
rebuilding them?

**Script / outputs.**

- `toxi_proof.py` -> `toxi_proof.json` (numbers), `.md` (table; reproduces identically from the JSON via
  `build_md`), `.png`.
- Run control: `toxi_proof.pid` (PID 4168) and `toxi_proof_status.json` (`phase=done`, 533.6 s,
  2026-10-12 08:54:56). PID 4168 is no longer alive.
- Cache: `%TEMP%\opencode\toxi_cache` (`test.parquet`, `samples.parquet`, `emb_*.npy`). **It is now
  empty.** No models are saved; there is no `--render` mode.
- Why PARTIAL: the test composition, the per-language pool sizes and every `leak_removed` count
  reproduce exactly. The stored AUCs cannot be re-derived: the embedding cache is gone and no model was
  persisted, so a re-check means re-embedding ~95k texts and refitting 5 models, which overwrites the
  provenance artifacts.

**Inputs.** Shipped files of `FredZhang7/toxi-text-3M` (local HF cache revision `e1e50df2`; the script
passes no `revision=`). Train file: 2,880,667 rows.

| item | en | it | de | fr | es |
|---|---:|---:|---:|---:|---:|
| rows in shipped train file | 2,528,002 | 20,611 | 12,255 | 26,175 | 26,726 |
| sampled train rows | 20,000 | 20,000 | 12,255 | 20,000 | 20,000 |
| eval-test rows used | 525 | 5,000 | 190 | 5,000 | 5,000 |
| rows in shipped test file | 525 | 8,359 | 190 | 10,719 | 7,980 |
| train pos rate | 0.117 | 0.295 | 0.299 | 0.365 | 0.328 |
| **test pos rate** | **0.625** | **0.185** | **0.716** | **0.298** | **0.374** |

`en` and `de` use the whole shipped test split; `it`/`fr`/`es` are capped at 5,000 (`random_state=0`).

**The test split does not match the train pool.** The EN test is 62.5% positive against 11.7% in the EN
train pool; the DE test is 71.6% positive against 29.9%. The IT test is 18.5% against 29.5%. The shipped
test split is therefore not a random holdout of the same distribution. E20's rebuilt split has
0.121 / 0.294 / 0.310 / 0.369 / 0.313.

**Leakage: these are the original leaky splits.**

- `dedup_test` drops eval-test texts from the **sampled** train rows only. It does not rebuild the
  splits.
- Measured on the cached revision: 90.1% (en), 99.9% (it), 84.7% (de), 99.9% (fr), 99.8% (es) of the
  eval-test texts appear **verbatim** in the shipped train CSV.
- The stored removal counts (en 3, it 4,847, de 161, fr 3,807, es 3,755) reproduce exactly, but they
  measure the sampling rate, not the contamination. Only 20,000 of 2.53M EN rows are drawn, so EN drops
  3 rows while 90% of its test texts sit in the pool.
- No normalized-text or near-duplicate dedup, and no translation-contamination check, is done.
- **The numbers below are NOT leakage-corrected and must not be quoted as evidence.**

**Protocol.**

- EuroBERT-210m, masked mean, max length 512, batch 64, fp32, GPU.
- `perlang_z`: L2-normalize, then z-score with each language's own train mean/std. The EN model uses
  each target's train statistics at test time; those rows are unlabeled for the transform.
- XGBoost `binary:logistic`, hist, max_bin 64, depth 8, min_child_weight 5, lr 0.1, 400 trees,
  `n_jobs=2`, **single seed 0**. No bootstrap, no second seed.
- Validation = 20% carve-out of the sampled train rows (the shipped validation file was deleted).
  Thresholds tuned there; AUC is threshold-free.
- EN model: train on EN, test every target. Native model: train on the target's own train, test the
  target's own test.

**Results.** Single run, seed 0.

| target | EN-trained AUC | native AUC | EN/native | target/EN | AP EN / native | FPR@R90 EN / native |
|---|---:|---:|---:|---:|---|---|
| it | 0.7626 | 0.8153 | 0.9353 | 0.9702 | 0.454 / 0.552 | 0.597 / 0.513 |
| de | 0.6254 | 0.5888 | **1.0622** | 0.7956 | 0.826 / 0.779 | 0.815 / 0.759 |
| fr | 0.7498 | 0.8212 | 0.9131 | 0.9538 | 0.554 / 0.650 | 0.619 / 0.445 |
| es | 0.7752 | 0.8342 | 0.9293 | **0.9862** | 0.683 / 0.728 | 0.571 / 0.415 |

EN self reference: AUC **0.7860**, AP 0.8182, 328 pos / 197 neg. Mean AP: EN-trained 0.629 vs native
0.677. Mean FPR@R90: EN-trained 0.650 vs native 0.533.

**0.975 bar.** The JSON stores both AUCs, so both ratios are computable.

| bar | it | de | fr | es | pass |
|---|---:|---:|---:|---:|:--:|
| target/EN | 0.9702 | 0.7956 | 0.9538 | **0.9862** | **1/4** |
| EN/native | 0.9353 | **1.0622** | 0.9131 | 0.9293 | **1/4** |
| target/EN at the code's own 0.9 bar | 0.9702 | 0.7956 | 0.9538 | 0.9862 | 3/4 |
| EN/native at the code's own 0.9 bar | 0.9353 | 1.0622 | 0.9131 | 0.9293 | 4/4 |

Both 0.975 passes are artifacts and must not be quoted:

- `es` passes target/EN by 0.011 on a test set whose label rate (0.374) differs from its train pool
  (0.328).
- `de` "passes" EN/native because the native model, trained on 29.9% positives, scores 0.589 on 190 rows
  (54 negatives) whose positive rate is 71.6%.

**Delta vs E20 (`toxi_proof_clean.py`, leakage-corrected).**

| target | EN-trained E36 -> E20 | native E36 -> E20 | EN/native E36 -> E20 | target/EN E36 -> E20 |
|---|---|---|---|---|
| it | 0.7626 -> 0.6489 (-0.114) | 0.8153 -> 0.8337 (+0.018) | 0.9353 -> 0.7783 (**-0.157**) | 0.9702 -> 0.7201 (**-0.250**) |
| de | 0.6254 -> 0.6776 (+0.052) | 0.5888 -> 0.8688 (**+0.280**) | 1.0622 -> 0.7800 (**-0.282**) | 0.7956 -> 0.7521 (-0.044) |
| fr | 0.7498 -> 0.7760 (+0.026) | 0.8212 -> 0.8662 (+0.045) | 0.9131 -> 0.8959 (-0.017) | 0.9538 -> 0.8612 (-0.093) |
| es | 0.7752 -> 0.6390 (-0.136) | 0.8342 -> 0.7969 (-0.037) | 0.9293 -> 0.8019 (-0.127) | 0.9862 -> 0.7092 (**-0.277**) |

EN self: 0.7860 -> 0.9010 (+0.115). Every one of the eight ratios is **higher** in the leaky run, by
0.017 to 0.282. The code's own verdict (0.9 bar, EN/native only) prints "comparable" at 4/4; E20 prints
0/4 on both bars. **E20 supersedes E36**: it pools all shipped splits, dedups, re-splits 80/10/10 with
0 exact-text cross-split overlap, runs 3 bootstrap seeds and saves 15 models.

**Verdict.** Provenance only. E36 is the leaky-split predecessor of E20. It does **not** measure
leakage-corrected cross-lingual toxicity transfer, and it is **not comparable** to E20 (different
splits, different label rates, different test sizes). Quote E20; cite E36 only as the pre-clean run.

**Caveats.**

- Original leaky splits. Dedup is exact-text and sample-only; near-duplicate and translation
  contamination is unmeasured and unremoved.
- Test label rates contradict the train pool (en 0.625 vs 0.117; de 0.716 vs 0.299; it 0.185 vs 0.295).
  The EN reference AUC is therefore not the same quantity as E20's.
- Tiny test sets: de 190 rows (54 negatives), en 525 rows (197 negatives). No CIs.
- Single seed, one run, no saved models, no embedding cache. Metric-level reproduction is not possible
  from what survives.
- `perlang_z` uses unlabeled target train rows, so the English model is not strictly target-free.
- Train sizes are not matched: EN 20,000, de 12,094 after leak removal.
- The dataset revision is not pinned in the script.

---

# PART F — Fine-tuning, generation and translation

The escape hatches from frozen features: train the last block (E29), generate the target text instead
of translating the test (E30), vary the generator (E31), pool all generators (E32), and quantize the
translator (E38).

## E29 — Fine-tune last layer + linear head, raw — DONE (verified 2026-10-20)

**Script to reproduce:** `finetune_lastlayer.py` — `python finetune_lastlayer.py --launch` (the only
invocation documented anywhere; the run actually used an undocumented `--caps` override). Reproduction
status: **PARTIAL**.

**Question.** Does training the last EuroBERT block + a linear head, on raw mean-pooled features, beat
the frozen-raw + XGBoost baseline by at least 0.975 per cell and metric? And does fine-tuning close the
cross-lingual gap?

**Script / outputs.**

- `finetune_lastlayer.py` -> `finetune_lastlayer.json` (numbers), `.md` (full tables), `.png`.
- `--launch [args]` starts the run detached and hidden (VBS launcher); `--regen-report` rebuilds md/png
  from the JSON only.
- `--model-id`, `--tag`, `--caps ds=epochs/steps`, `--max-epochs`, `--max-steps`, `--no-bf16`,
  `--smoke`.
- **No command is documented in `REPRODUCE.md`, `README.md`, `EXPERIMENTS.md` or the `.md`.** The run
  log resolves the caps to `sent 30/3000, agg 30/3000, toxic 30/400`, which requires an undocumented
  `--caps toxic=30/400` (or equivalent) override. The JSON records `toxic 1/400`. The script default is
  `toxic 2/1200`. Three values, one run.

**Protocol (as coded).**

- Trainable: `model.layers[-1]` + `Linear(768, n_labels)`. Embeddings, lower layers, final norm frozen.
- Pooling: masked mean of `last_hidden_state` over plain tokenization (content + `<|end_of_text|>`).
- Features: raw. No L2, z-score, `perlang_z`, PC/INLP, `D`. Frozen baseline also raw.
- AdamW (0.9, 0.95), eps 1e-5, wd 0.1, grad-clip 1.0, linear schedule, 10% warmup, batch 32.
- Early stopping: patience 1 epoch, on EN val. LR grid {1e-5, 3.6e-5, 6e-5, 1e-4}, selected by val AUC.
- bf16 autocast + fp32 master weights (log: `bf16_supported=True`).
- 210m. **One seed (0).** 9,438,720 trainable parameters.

**Inputs.**

| dataset | rows (train/val/test) | classes | frozen features |
|---|---|---|---|
| FLORES sent | 997 / 504 / 505 | 12 | `embeddings/mean/emb_{en,it,de,fr}.npy`, 2009x768, sliced to 2006 kept rows |
| FLORES agg | 276 / 140 / 139 | 11 | `embeddings_agg/mean/emb_*`, 562x768 |
| toxic | en 24000/3000/3000; it 18472/2310/2310; de 9836/1230/1230; fr 21011/2627/2627; es 23349/2919/2919 | binary | `toxi_emb_210/emb_{L}_{split}.npy` |

**Results.** Test AUC / macro-F1. `ratio` = fine-tuned / frozen, same cell and metric.

| dataset | eval | FT AUC | FT F1 | frozen AUC | frozen F1 | ratio AUC | ratio F1 | in-lang FT AUC/F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sent | EN-self | 0.9156 | 0.5508 | 0.9106 | 0.4878 | 1.006 | 1.129 | - |
| sent | EN->it | 0.8124 | 0.2480 | 0.7987 | 0.1843 | 1.017 | 1.346 | 0.8413 / 0.3059 |
| sent | EN->de | 0.8139 | 0.2465 | 0.8321 | 0.2926 | **0.978** | **0.842** | 0.8565 / 0.3512 |
| sent | EN->fr | 0.8211 | 0.2504 | 0.8295 | 0.2330 | 0.990 | 1.075 | 0.8360 / 0.3877 |
| agg | EN-self | 0.9254 | 0.5450 | 0.9134 | 0.5555 | 1.013 | 0.981 | - |
| agg | EN->it | 0.7905 | 0.2270 | 0.8530 | 0.2131 | **0.927** | 1.065 | 0.9194 / 0.5492 |
| agg | EN->de | 0.8526 | 0.2015 | 0.8359 | 0.2779 | 1.020 | **0.725** | 0.9177 / 0.5092 |
| agg | EN->fr | 0.8337 | 0.2689 | 0.8100 | 0.2939 | 1.029 | **0.915** | 0.9162 / 0.4806 |
| toxic | EN-self | 0.9287 | 0.7971 | 0.9040 | 0.7601 | 1.027 | 1.049 | - |
| toxic | EN->it | 0.6425 | 0.5006 | 0.6692 | 0.4490 | **0.960** | 1.115 | - |
| toxic | EN->de | 0.7389 | 0.5535 | 0.7052 | 0.5106 | 1.048 | 1.084 | - |
| toxic | EN->fr | 0.7425 | 0.4864 | 0.7714 | 0.4457 | **0.963** | 1.091 | - |
| toxic | EN->es | 0.6891 | 0.5302 | 0.6293 | 0.4297 | 1.095 | 1.234 | - |

**In-language reference (toxic).** The run has no in-language FT for toxic. Recomputed with the same
frozen XGB config (train -> own test): it 0.8413 / 0.7223, de 0.8648 / 0.7656, fr 0.8700 / 0.7746, es
0.7997 / 0.6756. Fine-tuned EN-transfer reaches 0.764 / 0.854 / 0.853 / 0.862 of that native AUC.

**Verdict at the 0.975 bar (fine-tuned vs frozen, per cell and metric).** **20/26 cells pass, 6 fail.**

| fail | ratio |
|---|---:|
| sent EN->de F1 | 0.842 |
| agg EN->it AUC | 0.927 |
| agg EN->de F1 | 0.725 |
| agg EN->fr F1 | 0.915 |
| toxic EN->it AUC | 0.960 |
| toxic EN->fr AUC | 0.963 |

The preliminary list of five failures is correct but incomplete: it omits **agg EN->it AUC 0.927**, the
worst AUC cell in the study. Two passes are inside noise: `sent EN->de` AUC 0.978 (margin 0.003) and
`agg EN-self` F1 0.981 (margin 0.011). Bootstrap SE of a single AUC here is 0.010–0.016 (sent),
0.021–0.032 (agg), 0.008–0.016 (toxic). The bar tolerance is 0.017–0.023 absolute at these levels.

**Verdict at the 0.975 bar (target / EN-self, the E18 "comparable" definition).** **0/20** transfer
cells pass, for fine-tuned and frozen alike. Best is agg EN->de AUC 0.921. Fine-tuning does not make any
language comparable.

**Verdict (fine-tuned vs native).** EN-transfer FT reaches 0.98 of in-language FT on FLORES sent fr AUC
and 0.95–0.97 on it/de. On FLORES agg the residual gap is large: 0.860 / 0.929 / 0.910. In-language FT
is **not** a ceiling: on FLORES sent the frozen in-language XGB is better than the in-language FT
(0.877 / 0.884 / 0.875 AUC vs 0.841 / 0.857 / 0.836), because the FT arm trains on train only while the
frozen arm trains on train+val. On agg AUC the two are equal (ratio 1.009–1.025).

**Caveats.**

- Single seed, no CIs. FLORES agg test is 139 rows and 11 classes (smallest class 4 rows). No agg
  pass/fail is statistically separable at 0.975.
- Toxic fine-tuning is capped at 400 steps = 12,800 samples = 53% of one EN epoch. Early stopping cannot
  trigger. All four LR trials report `best_epoch=1` from one validation point, and the chosen LR (1e-4)
  is the grid edge. The toxic fine-tuned columns are undertrained and LR-confounded.
- Toxic "F1" is binary **macro-F1** at threshold 0.5, not F1(pos) as `EXPERIMENTS.md` states. On the
  frozen model, macro-F1 exceeds F1(pos) by 0.20–0.38 (it: 0.449 vs 0.068); transfer recall at 0.5 is
  3–11%. These columns mostly measure the majority class.
- The in-script XGB baseline is fixed and untuned (depth 6/8, 400 trees). The repo's tuned same-feature
  baseline is stronger: sent 0.9186 self / 0.8470 it, agg 0.9458 / 0.8750 (`models_registry.json`,
  `REPO_REF`). Against it, fine-tuning loses all four FLORES AUC cells (ratios 0.997, 0.959, 0.978,
  0.903), so "FT improves EN-self" holds only for the fixed baseline. On toxic the comparison is mixed
  (self 1.031, it 0.990, de 1.090, fr 0.957, es 1.078).
- Toxic baseline asymmetry: frozen XGB trains on EN train (24,000), FT on train+val (27,000).
- `toxi_emb_210/` has no producing script in the repo and differs from the canonical `emb_bf16/toxic/210m`
  (row cosine 0.9996 mean, 0.769 min). Truncation also differs by one token (FT 512,
  `extract_embeddings.py` 511).
- The JSON's `toxic max_epochs=1` contradicts the run log (`30/400`). JSON/md/png mtimes are 08:42, two
  minutes after the run log ends at 08:40:09.
- `--seeds` ignores all but the first value; there is no multi-seed path.

---

## E30 — Synthetic Italian from English — DONE (FLORES) / partial (toxic)

**Script to reproduce:** `synth_it.py` / `synth_toxic_it.py` — `python synth_it.py --generator
HuggingFaceTB/SmolLM3-3B --gen-batch 16` (FLORES) and `python synth_toxic_it.py --generator
HuggingFaceTB/SmolLM3-3B --en-cap 160 --gen-batch 32 --max-new 256` (toxic, stored cap-160 run).
Reproduction status: **REPRODUCIBLE** (cached generation hash-verified; 4/4 210m XGBoost cells re-ran
exactly; cold generation not re-run).

**Question.** Can Italian training text generated from the **English** train text by a small instruct
LLM approach the real-Italian ceiling, and does the same recipe transfer to imbalanced toxicity?

**Script / outputs.**

- `synth_it.py` -> `synth_it.json` (numbers), `.md` (tables), `.png`.
- `synth_toxic_it.py` -> `synth_toxic_it.json`, `.md`, `.png`.
- `--from-json` (both) rebuilds `.md`/`.png` from the JSON only.
- Create-only outputs. Generation + synthetic-embedding caches live in `%TEMP%\opencode\synth_it\` and
  `%TEMP%\opencode\synth_toxic_it\`.
- FLORES generation is cached as one whole-list JSON; toxic as 128-row chunk JSONs (resumable).
- The script defaults do **not** reproduce the stored FLORES artifact: the default generator is
  `Qwen/Qwen3-0.6B`, default `--gen-batch` is 32, while the JSON records `HuggingFaceTB/SmolLM3-3B` and
  batch 16.

**Inputs.**

| task | data | labels / split | features |
|---|---|---|---|
| FLORES sent | `flores200_en_it_de_fr.csv`, 2,006 rows aligned to the 2,009-row embedding arrays | 12 topic classes; train 997 (dev), val 504 (seed-0 devtest), test 505 | RAW EuroBERT mean: `embeddings/mean/` (210m), `embeddings_610/mean/` (610m) |
| toxic | `data/toxi_clean/{en,it}_{train,validation,test}.parquet` | binary `is_toxic`; EN train 24,000; IT train 18,472; IT val 2,310; IT test 2,310 (29.35% positive) | `emb_bf16/toxic/<bb>/emb_<lang>_<split>.npy`, bf16 forward -> fp32 |

**Generation.**

| task | generator | prompt | decode | batch | max_new | source rows |
|---|---|---|---|---|---:|---:|---|
| FLORES | `HuggingFaceTB/SmolLM3-3B` | custom short system prompt: "You rewrite/translate English text into natural Italian, preserving meaning; output only the Italian text." + raw English user text | greedy | 16 | 384 | EN train+val = 1,501 |
| toxic | `HuggingFaceTB/SmolLM3-3B` | `paraphrase_prompts.build_messages("crosslingual", src="English", tgt="Italian")` | greedy | 32 | 256 | EN train cap = 160 (seed-0 subsample) |

Both run bf16 on CUDA with `enable_thinking=False` when the chat template accepts it. Generation input
is truncated at 1,024 tokens. **No filtering, dedup, or label check** is applied to either generation
set, although `PARAPHRASE_TEMPLATE.md` §5 recommends it.

**Protocol.** Four arms per backbone: `synth_it` (synthetic IT), `real_it` (same-size real-IT ceiling),
`en_transfer` (English), `mixed` (EN + synthetic IT; **not** EN + real IT). Fixed XGBoost, CPU,
`n_jobs=2`, seed 0, depth 8, 400 trees, lr 0.1, `max_bin=64`: `multi:softprob` / 12 classes for FLORES,
`binary:logistic` for toxic. Train on each arm's own train(+val) rows, evaluate on the **real Italian
test** only. FLORES headline = macro-F1; toxic headline = F1(positive) at 0.5, with a val-tuned
threshold and recall-first view also reported. "Comparable" = target / real ratio >= **0.975** on the
headline metric.

**Results — FLORES sent (real Italian test, 505 rows).**

| backbone | arm | accuracy | macro-F1 | OVR-AUC | macro-F1 / real | >=0.975 |
|---|---|---:|---:|---:|---:|:--:|
| 210m | `synth_it` | 0.5386 | **0.4103** | 0.8809 | **0.9888** | yes |
| 210m | `real_it` | 0.5545 | 0.4149 | 0.8838 | 1.0000 | — |
| 210m | `en_transfer` | 0.4158 | 0.2332 | 0.7920 | 0.5620 | no |
| 210m | `mixed` | 0.5545 | 0.4258 | 0.8726 | **1.0263** | yes |
| 610m | `synth_it` | 0.5604 | **0.4691** | 0.8804 | **0.9834** | yes |
| 610m | `real_it` | 0.5604 | 0.4770 | 0.8976 | 1.0000 | — |
| 610m | `en_transfer` | 0.4832 | 0.3275 | 0.8565 | 0.6866 | no |
| 610m | `mixed` | 0.5663 | 0.4702 | 0.9013 | **0.9859** | yes |

**0.975 check (exact JSON ratios).**

- `synth_it` / `real_it`: **0.988781** (210m), **0.983436** (610m) -> **both pass**.
- `mixed` / `real_it`: **1.026329** (210m), **0.985865** (610m) -> **both pass**.
- AUC ratios also pass: `synth_it` 0.996671 / 0.980874; `mixed` 0.987330 / 1.004078.
- Accuracy ratio: 210m `synth_it` = **0.971429**, which is **below 0.975**; all other 11
  accuracy/AUC/macro-F1 cells pass. The pass is therefore metric-specific, not universal.
- `en_transfer` = **0.233 / 0.328** macro-F1 (ratios 0.562 / 0.687), confirmed.

**Results — toxic, stored run is cap=160 only (real Italian test, 2,310 rows).**

| backbone | arm | accuracy | F1(pos)@0.5 | AUC | F1@val-thr | F1/real @0.5 | F1/real @val |
|---|---|---:|---:|---:|---:|---:|---:|
| 210m | `synth_it` | 0.7069 | 0.0636 | 0.5709 | 0.4615 | 0.1095 | 0.7104 |
| 210m | `real_it` | 0.7965 | 0.5811 | 0.8420 | 0.6496 | 1.0000 | 1.0000 |
| 210m | `en_transfer` | 0.7082 | 0.0203 | 0.6211 | 0.4665 | 0.0350 | 0.7181 |
| 210m | `mixed` | 0.7091 | 0.1340 | 0.5971 | 0.4520 | 0.2306 | 0.6958 |
| 610m | `synth_it` | 0.7095 | 0.0821 | 0.6258 | 0.4750 | 0.1396 | 0.7427 |
| 610m | `real_it` | 0.7978 | 0.5878 | 0.8430 | 0.6396 | 1.0000 | 1.0000 |
| 610m | `en_transfer` | 0.7069 | 0.0688 | 0.6264 | 0.4696 | 0.1170 | 0.7342 |
| 610m | `mixed` | 0.7195 | 0.2193 | 0.6829 | 0.4971 | 0.3730 | 0.7771 |

At 0.975, **0/8** synth/mixed toxic cells pass on any metric. The best cell is 610m `mixed` (F1@val
ratio 0.777). The `real_it` arm trains on 18,472 real-IT rows vs **160** synthetic rows, so this is a
115× size mismatch, not a size-matched ceiling.

**Verdict.**

- FLORES: synthetic Italian from the EN train+val text reaches **98.9% (210m) / 98.3% (610m)** of the
  same-size real-IT macro-F1 ceiling. At 0.975, both `synth_it` cells and both `mixed` cells pass on
  macro-F1 (and on AUC); EN transfer is far below (0.233 / 0.328). 610m raises absolute macro-F1 by
  +0.059 (`synth`) / +0.044 (`mixed`) but changes the ratio little.
- Toxic: only a **cap=160 plumbing run** exists. At 0.5 it reaches 11% / 14% of the ceiling; at a
  val-tuned threshold 71% / 74%. The full run is **NOT DONE / NOT AVAILABLE**; no toxic cell passes
  0.975.
- Leakage: generation sources are FLORES EN train+val and toxic EN train only. The generation-cache
  hashes match those row lists exactly; FLORES EN test rows have **0** overlap with the generation
  source. No test leakage was found.

**Caveats.**

- Single seed; FLORES test = 505 rows. The 0.9888 / 0.9834 margins over 0.975 are small and have no CIs.
- The pass depends on macro-F1: 210m `synth_it` accuracy ratio is 0.9714 (< 0.975).
- "Ceiling" = same-size real-IT model on FLORES; for toxic it is 18,472 real rows, so the comparison
  confounds language, scale, and class balance (cap-160 positive rate 0.125 vs IT train 0.283 / test
  0.294).
- The two tasks use different prompts (FLORES custom prompt, toxic `paraphrase_prompts` template), so
  this is not a single-prompt comparison.
- No filtering: toxic generations contain 79/160 `"""`-wrapped outputs, 146/160 with quote characters,
  and 9/160 outside the recommended length ratio [0.6, 2.0] (max 19.3×).
- The 0.975 bar is borrowed from E18, where the FLORES headline metric was AUC; E30 switches to
  macro-F1.

---

## E31 — Generator comparison — FLORES DONE, toxic NOT DONE (verified 2026-10-20)

**Script to reproduce:** `gen_compare.py` — `python gen_compare.py --datasets flores --backbones 210m`
(FLORES table from the on-disk cache; ~2 min) and `python gen_compare.py --from-json` (rebuild
`.md`/`.png` only). Reproduction status: **PARTIAL**.

**Question.** Does the choice of small instruct LLM used to generate synthetic Italian training text
change the downstream FLORES score? Does the best generator reach >= **0.975** × the real-Italian
ceiling on macro-F1? Is the toxic arm complete?

FLORES reproduces exactly from the cache. The toxic table does not reproduce from this script; see
"Toxic arm". `--datasets flores` **rewrites** `gen_compare.json` and drops the toxic block, so run it on
a copy if the committed toxic numbers matter.

**Script / outputs.**

- `gen_compare.py` -> `gen_compare.json` (numbers), `gen_compare.md` (tables + samples), `.png`.
- Caches in `%TEMP%\opencode\gen_compare\`: one generation JSON per (generator, dataset), plus the
  synthetic-Italian embedding `.npy` files.
- `--from-json` renders `.md`/`.png` from the stored JSON and does no training.
- `--launch` runs detached hidden; `--smoke` writes no repo outputs.

**Inputs.**

| item | value |
|---|---|
| generators | `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| prompt | `paraphrase_prompts.build_messages('crosslingual', src='English', tgt='Italian')` |
| decode | greedy (`do_sample=False`, `num_beams=1`), bf16, `max_new_tokens=384`, prompt truncation 1024, left padding, `enable_thinking=False` when accepted |
| batch | 8 in the stored run (current script default is 16; `--gen-batch` is dead, `--start-batch` controls it) |
| embeddings | EuroBERT-210m masked mean: FLORES fp32, max_len 511; toxic bf16, max_len 512 |
| classifier | XGBoost `hist`, max_bin 64, depth 8, mcw 5, lr 0.1, 400 trees, `n_jobs=2`, seed 0; FLORES `multi:softprob` (12 classes), toxic `binary:logistic` |
| FLORES rows | train 997 + val 504 = **1501**; real Italian test **505** |
| toxic rows | EN train cap **8000** (seed 0); real Italian test **2310** |
| seeds | FLORES split 0; toxic cap 0; XGBoost 0; generation is greedy (no sampling seed) |

The FLORES split is the seed-0 half-split of the devtest rows (`train_flores12.prepare_sentence`). The
3 generators are embedded in the same EuroBERT-210m space as the stored real-Italian vectors.

**Protocol.** Every arm uses the **same 1501 FLORES train rows and the same 505 real-Italian test rows**.
Training text differs only in the arm:

- `Qwen3.5-0.8B`, `Qwen3.5-2B`, `SmolLM2-1.7B`: synthetic Italian built from the EN `en` column,
  labelled with the row's 12-class topic.
- `real_it`: the real Italian `it` column of the same 1501 rows (same-size ceiling).
- `en_transfer`: the English source of the same 1501 rows (floor).

All arms share one fixed XGBoost configuration. Macro-F1 is argmax, 12 classes, `zero_division=0`; AUC
is OVR macro. Each cache filename hash equals the hash of the current 1501-row source list, so row order
is identical.

**Results (FLORES sent, EuroBERT-210m, real Italian test).**

| generator | accuracy | macro-F1 | AUC | macro-F1 / real_it | >= 0.975 |
|---|---:|---:|---:|---:|:--:|
| **SmolLM2-1.7B** | 0.5248 | **0.4125** | 0.8648 | **0.994** | pass |
| Qwen3.5-0.8B | 0.5465 | 0.4067 | 0.8533 | **0.980** | pass |
| Qwen3.5-2B | 0.5228 | 0.3968 | 0.8823 | **0.956** | fail |
| real_it (ceiling) | 0.5545 | 0.4149 | 0.8838 | 1.000 | — |
| en_transfer (floor) | 0.4158 | 0.2332 | 0.7920 | 0.562 | fail |

Exact ratios: SmolLM2 **0.994144**, Qwen3.5-0.8B **0.980116**, Qwen3.5-2B **0.956221**. JSON,
`gen_compare.md`, and the `EXPERIMENTS.md` table agree at 4 dp. `gen_compare.md` is byte-identical to
`build_md()` of the stored JSON.

**Toxic arm: NOT DONE.** SmolLM2-1.7B has no toxic result.

| arm | n_train | accuracy | F1 (positive) | AUC | status |
|---|---:|---:|---:|---:|---|
| Qwen3.5-0.8B | 8000 | 0.7186 | 0.1402 | 0.6994 | stored |
| Qwen3.5-2B | 8000 | 0.7234 | 0.1860 | 0.6840 | stored |
| SmolLM2-1.7B | — | — | — | — | **skipped: greedy generation incomplete, 4800/8000** |
| real_it (ceiling) | 18472 | 0.7965 | 0.5811 | 0.8420 | stored |
| en_transfer (floor) | 8000 | 0.7095 | 0.0455 | 0.6535 | stored |
| union of 2 Qwen arms | 16000 | 0.7251 | 0.2170 | 0.6836 | JSON only (not in `.md`) |

The SmolLM2 partial checkpoint has 4,800 of 8,000 rows (0 empty). The last status write is 2026-10-14
09:04:03; PID 4156 is dead. The run was slow, not hung: it fell from 4.2 to ~0.7 rows/s and still
advanced 200 -> 4800. The two Qwen toxic rows were computed by an out-of-repo helper
(`%TEMP%\opencode\gen_compare\inject_toxic.py`) from cached generations. The current script can train
those two arms from the same caches, but it cannot emit the `skipped` entry: a missing generation raises
and aborts the run. The JSON nevertheless lists `toxic` in `completed_datasets`.

**Verdict.** On macro-F1, **2 of 3 generators pass the 0.975 bar**: SmolLM2-1.7B at **0.994** and
Qwen3.5-0.8B at **0.980**; Qwen3.5-2B fails at **0.956**. SmolLM2 is the best single generator. This
result is **metric-specific**: on AUC the ratios are 0.978 / 0.965 / 0.998, so Qwen3.5-0.8B fails and
Qwen3.5-2B passes; on accuracy they are 0.946 / 0.986 / 0.943. The set of passing generators changes
with the metric. The toxic arm is **NOT DONE** (2/3 generators, no SmolLM2).

**Caveats.**

- Single seed, 505 test rows, no CIs. A paired bootstrap (2,000 draws) gives macro-F1 ratio 95% CIs of
  0.879–1.122 (SmolLM2), 0.890–1.079 (Qwen0.8), 0.853–1.079 (Qwen2B); P(ratio >= 0.975) is 0.62, 0.54,
  0.37. The pass/fail labels are point estimates.
- **"Qwen3.5-2B is worse than Qwen3.5-0.8B" is a ranking, not a ratio, and it is not robust.** At the
  0.975 bar the order does hold (0.980 pass vs 0.956 fail), but each ratio is against real_it, not 2B vs
  0.8B. The macro-F1 gap is 0.0099 (0.3968 vs 0.4067; bootstrap 95% CI -0.051 to +0.030). AUC runs the
  other way: 0.8823 vs 0.8533, a paired bootstrap gap of +0.029 with 95% CI +0.011 to +0.047. Accuracy
  gap is -0.024, CI -0.057 to +0.008.
- The toxic ceiling is **not size-matched** (18,472 real rows vs 8,000 synthetic) and the base rate
  differs (EN train 11.7% positive; IT test 29.4%). F1 at a fixed 0.5 threshold therefore compares
  priors and row counts as well as generation quality.
- Generated text is not filtered. `_clean` removes think tags and four prefixes but keeps enclosing
  quotes and leading bullets: 1,367/1,501 SmolLM2 FLORES outputs are fully quoted, 189 Qwen0.8 FLORES
  outputs start with a bullet, and the toxic caches contain 114 (Qwen0.8) / 146 (Qwen2B) prompt-echo
  rows. Those rows add label noise.
- Current defaults differ from the stored run: 16/16 batch and `torch.compile=True` now, 8/8 and `False`
  in the stored config. A from-scratch rerun needs `--start-batch 8 --no-compile` to match, but
  `--gen-batch` changes nothing.
- Cache keys cover generator + dataset + source texts only. A change to the prompt, decode length,
  model revision, embedding dtype, or max length will silently reuse stale generations / vectors.
- `python gen_compare.py --datasets flores` overwrites `gen_compare.json`; there is no merge with the
  stored toxic results. `--from-json` is the safe render path.

---

## E32 — Union of all generators' paraphrases — FLORES DONE (verified 2026-10-20), toxic PARTIAL

**Script to reproduce:** `all_paraphrases.py` — `python all_paraphrases.py` (default `--datasets
flores,toxic`, 273.7 s). FLORES only: `python all_paraphrases.py --datasets flores` (181.7 s). Rebuild
`.md`/`.png` from the JSON only: `python all_paraphrases.py --from-json`. Reproduction status:
**REPRODUCIBLE** (all six FLORES arms re-fit from the cached embeddings on 2026-10-20 matched
`all_paraphrases.json` to 4 dp; no generation was rerun).

**Question.** If the synthetic Italian of the three generators (Qwen3.5-0.8B, Qwen3.5-2B,
SmolLM2-1.7B) is pooled into ONE training set, does the union beat the best single generator and the
real-Italian reference on the real Italian FLORES test?

**Script / outputs.**

- `all_paraphrases.py` -> `all_paraphrases.json` (numbers), `.md` (tables), `.png` (bar chart).
- The pipeline is imported from `gen_compare.py`; the per-generator Italian text is read from the
  `%TEMP%\opencode\gen_compare\gen_<gen>_<dataset>_<n>_<hash>.json` cache. **No generation was run.**
- Default runs FLORES, then toxic. Toxic is skipped to `pending` only when no generator cache is
  present; it runs with whatever generators are cached and records the others in `gaps`.
- The script writes the outputs after each dataset. The stored run took 181.7 s for FLORES and 91.9 s
  more for toxic. A later commit (`7ea97fc`) only added the `gpu_limit` call; the metric path is
  unchanged.

**Inputs.**

| dataset | embeddings | labels / split | models |
|---|---|---|---|
| FLORES sent | `embeddings/mean/emb_{en,it}.npy`, 2009x768, sliced to the 2006 kept rows | `flores200_en_it_de_fr.csv`; 12 topic classes; 997 train / 504 val / 505 test (seed 0) | 3 generators -> synth IT; XGBoost |
| toxic | `emb_bf16/toxic/210m/emb_{en,it}_{train,test}.npy` | `data/toxi_clean/{en,it}_*.parquet`; binary; EN train capped 8000 (seed 0); IT test 2310 | 3 generators, only 2 cached; XGBoost |

FLORES `n_train` is **train + val = 997 + 504 = 1501** (the `gen_compare` convention), and the test is
the other half of `devtest`: 505 real Italian rows. The three synthetic caches each hold exactly 1501
paraphrases of those EN train+val rows.

**Protocol.**

- Single arms: train on one generator's 1501 synthetic Italian rows. Real_it: train on 1501 real Italian
  rows. En_transfer: train on the 1501 English rows. All evaluated on the same 505 real Italian test
  rows.
- Union: `X = vstack(per-generator mean embeddings)`, `y` repeated once per generator -> **4503x768**,
  4503 labels. This equals embedding the pooled text because EuroBERT mean pooling is per-text. No
  per-language z-scoring.
- Fixed XGBoost: hist, `max_bin=64`, cpu, `n_jobs=2`, seed 0, depth 8, `min_child_weight=5`, lr 0.1,
  400 trees, `multi:softprob` / 12 classes. No tuning, no early stopping.
- Headline metric: macro-F1 (argmax). AUC is macro OVR. The 0.975 bar is applied to `all_combined /
  comparator` on macro-F1.

**Results (210m, real Italian test, 505 rows).**

| arm | n_train | accuracy | macro-F1 | AUC |
|---|---:|---:|---:|---:|
| **all_combined** | **4503** | **0.5881** | **0.4749** | **0.8821** |
| Qwen3.5-0.8B | 1501 | 0.5465 | 0.4067 | 0.8533 |
| Qwen3.5-2B | 1501 | 0.5228 | 0.3968 | 0.8823 |
| SmolLM2-1.7B (best single) | 1501 | 0.5248 | **0.4125** | 0.8648 |
| real_it (real-IT reference) | 1501 | 0.5545 | 0.4149 | 0.8838 |
| en_transfer (floor) | 1501 | 0.4158 | 0.2332 | 0.7920 |

`+0.0624` macro-F1 over the best single (0.4749 - 0.4125) and `+0.0599` over real_it.

**0.975 bar.**

| comparison | ratio, full precision | ratio from the 4-dp table | >= 0.975 |
|---|---:|---:|:--:|
| all_combined / real_it | **1.1445** | **1.145** | pass |
| all_combined / best_single | **1.1512** | **1.151** | pass |

The two passes are not close. The quoted **1.145** is the quotient of the 4-dp table values (0.4749 /
0.4149 = 1.144613); at full precision the ratio is 1.144472. The 0.0005 difference is rounding only.
The 0.975 bar was defined in E18 as target/English *parity*; here it is used as a superiority test,
which is a different use of the same threshold.

**Confound: the comparison is NOT volume-matched.** The union has **3× the rows** (4503 vs 1501) but
only **1501 unique source sentences and 1501 unique labels** — it is three paraphrastic views of the
same labelled examples, not 3× independent supervision. Every comparator has 1501 rows, and the
real-Italian reference is also 1501 rows. Therefore:

- the entry's phrase "the **same-size** real-Italian ceiling" is **false**: 4503 vs 1501;
- real_it cannot be a true ceiling once the union exceeds it — it is a 1501-row reference;
- the gain mixes extra rows/views with cross-generator diversity.

Audit-only controls (same cached embeddings, same XGB config, seed 0; not in the repo JSON) isolate the
volume effect:

| control | n_train | macro-F1 | AUC |
|---|---:|---:|---:|
| union of 3, subsampled to 1501 (500/500/501) | 1501 | 0.4089 | 0.8722 |
| best single, its 1501 rows duplicated 3× | 4503 | 0.4349 | 0.8517 |
| union of 2 (Qwen3.5-0.8B + SmolLM2), full | 3002 | 0.4598 | 0.8760 |
| union of 3, full | 4503 | 0.4749 | 0.8821 |

At matched volume the union is **below** the best single (0.4089 vs 0.4125). Duplicating one generator's
rows to 4503 recovers only +0.022 of the +0.062; the remaining +0.040 is diversity of distinct
paraphrases, not row count per se. The defensible claim is therefore: **three paraphrastic views per
source example improve macro-F1, with a gain that grows with the number of views and generators; it is
not evidence that pooled synthetic data beats a same-size real-Italian model.**

**Leakage / provenance.**

- `flores_struct()` (`all_paraphrases.py:79-90`) takes `en_src = cur.iloc[train_idx + val_idx]`; the test
  is `test_idx` only. Verified disjoint: 1501 train+val rows, 505 test rows, intersection 0.
- The generator cache path is keyed by the exact 1501 EN texts (`gen_cache_path`, `:117-121`; hash =
  `[generator, dataset] + texts`). All three FLORES caches exist and contain 1501 entries, so they were
  produced from train+val texts only. No test sentence was paraphrased.
- The union's features are the cached per-text embeddings of those 1501 texts, stacked; labels are the
  same train+val labels repeated. The test features are the real Italian test embeddings. **No leakage
  found.**
- The 504 val rows are inside `n_train` (fixed hyperparameters, so no tuning leak), but calling this
  "train" is imprecise; the pure training split is 997.
- Cache staleness is the only provenance risk: the hash covers the source texts and generator name, not
  the generation prompt/decoding settings.

**Toxic arm — NOT DONE (partial, 2 of 3 generators).** `all_paraphrases.json` contains a toxic result
for Qwen3.5-0.8B + Qwen3.5-2B only: n_train 16000 (2 × 8000), accuracy 0.7251, F1 0.2170, AUC 0.6836;
real_it 0.5811 F1, en_transfer 0.0455. SmolLM2-1.7B is in `gaps` because its generation hung at
**4800/8000 rows** (`smol_toxic_gen.log`; commit `3ae8703` says "SmolLM2 skipped - hung"). The MD table
still labels the row "ALL generators", which is misleading. The `EXPERIMENTS.md` E32 entry does not
report the toxic arm at all. Do not cite toxic as an all-generator result.

**Verdict.** On FLORES the union wins on accuracy and macro-F1, and it clears 0.975 against both the
best single (1.151) and real_it (1.145). AUC does **not** win: 0.8821 vs Qwen3.5-2B 0.8823 and real_it
0.8838, so "AUC ties at the top" is only approximately true. The effect is real but confounded with 3×
rows and 3 views; the "beats the real-Italian ceiling" wording should be replaced by "beats a 1501-row
real-Italian reference at 3× the rows".

**Caveats.**

- Single seed, one 505-row test split, no CIs. A 0.01 macro-F1 difference is well inside sampling noise
  at this test size.
- `1.145` is table-rounded; full precision is 1.1445.
- Macro-F1 uses argmax on an imbalanced 12-class test; AUC is macro OVR.
- The union gives every source example exactly 3× weight. No sample weighting or per-generator
  calibration was tried.
- Artifacts are create-only in effect: caches and JSON are reused/overwritten, so stale caches silently
  feed a rerun. Delete the FLORES `gen_*`/`synthemb_*` caches to regenerate from text.

---

## E38 — GGUF quantization quality (EN->IT) and EN->PL paraphrase->translate — PARTIAL (audited 2026-10-20)

**Script to reproduce:** `gguf_quant_quality.py` — `python gguf_quant_quality.py --n 150 --n-ctx 2048
--n-gpu-layers -1 --n-threads 4`. Reproduction status: **PARTIAL**.

**Question.** Does Q4_K_M keep >= 0.975 × the BF16 translation quality on FLORES EN->IT? How do two
Qwen GGUFs compare on a paraphrase-then-translate EN->PL job?

Two lines share the same machinery: a language-agnostic KV-cached head, greedy translation (temp 0), and
a paraphrase step (temp 0.8). E38 covers both, because both feed the "synthetic target text from
English" pipeline.

**Scripts / outputs.**

| producer | command (reconstructed) | artifacts | status |
|---|---|---|---|
| `gguf_quant_quality.py` | `python gguf_quant_quality.py --n 150 --n-gpu-layers -1` | `gguf_quant_quality.json`, `.md`, `.partial.json` | COMPLETE: 3/3 quants × 150 sents |
| `tools/run_dual_translate.py` -> `paraphrase_translate.py` | `python tools/run_dual_translate.py` | `out_qwen35_2b_pl.csv`, `out_qwen25_3b_pl.csv`, `out_timing.json` | COMPLETE, but 4/4 variants per row are duplicates |
| `tools/run_translate.py` -> `gguf_translate.py` | `python tools/run_translate.py` | `out_qwen25_15b_pl.csv` | output only; its `out_timing.json` was overwritten by the dual run |
| `tools/compare_pl.py` | `python tools/compare_pl.py` | `compare_pl.json`, `compare_pl.md` | COMPLETE for 2 models; per-variant block is vacuous |
| (none) | ad-hoc | `cmp_batched.csv`, `cmp_filtered.csv`, `cmp_seq.csv`, `seq12.csv`, `seq_2507.csv`, `out_div_test.csv`, `out_sep_test.csv`, `out_test_para.csv`, `out_test_q35_gguf.csv`, `out_smoke_qwen25.csv` | PARTIAL experiments; no producer script, all untracked |
| `TRANSLATE.md` | — | docs for `paraphrase_translate.py` | stale in one claim (see Caveats) |

`gguf_quant_quality.partial.json` is the per-model checkpoint and only holds the three finished models.
`--resume` skips a model already present there, so the final JSON was written without new generation.
**Delete the partial file to recompute.**

**Inputs.**

| input | content | used |
|---|---|---|
| `flores200_en_it_de_fr.csv` | 2006 rows, 11 cols, UTF-8 (`en,it,de,fr,...`) | first 150 EN/IT pairs (`load_flores`) |
| `flores_en_pl.csv` | 25 rows, `en,pl` (FLORES devtest) | all 25 × 4 paraphrases |
| `models/Qwen3.5-2B-Q4_K_M.gguf` | 1,280,835,840 B | quant run + EN->PL |
| `models/Qwen3.5-2B-Q8_0.gguf` | 2,012,012,800 B | quant run |
| `models/Qwen3.5-2B-BF16.gguf` | 3,775,709,216 B | quant run (reference) |
| `models/qwen2.5-3b-instruct-q4_k_m.gguf` | 2,104,932,768 B | EN->PL comparison |
| `models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1,117,320,736 B | EN->PL, 1.5B baseline |

All three quant GGUFs exist locally. `models/` and `models/*.gguf` are gitignored, so a fresh clone must
fetch them with `tools/fetch_gguf.py`, `tools/fetch_qwen25_gguf.py`, `tools/fetch_qwen25_3b_gguf.py`. No
checksum is recorded for any weight file.

**Protocol.**

- Quant run: `Prompt` from `gguf_translate.py` (system + instruction head, encoded once), the
  `crosslingual` prompt from `paraphrase_prompts.build_messages`, the model chat template, an empty
  ` thinking` block forced (no reasoning), greedy decode until EOS, `n_ctx=2048`, `n_gpu_layers=-1`,
  seed 0. Score with sacrebleu: chrF (char_order 6, word_order 0), chrF++ (word_order 2), corpus BLEU.
- EN->PL: paraphrase ×4 (temp 0.8, top_k 50) then translate each (temp 0), `n_ctx=4096`, full GPU, 25
  rows, `--limit 25`. Score variant-0 output against the human `pl` reference with sacrebleu chrF
  (default). No BLEU is stored for this line (recomputed in the audit).

**Results — quantization (EN->IT, FLORES, 150 sentences).**

| quant | size MB | chrF | chrF++ | BLEU | avg tok/s | chrF/BF16 | BLEU/BF16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 1281 | 53.62 | 50.46 | 22.56 | 66.1 | 0.9927 | 0.9919 |
| Q8_0 | 2012 | 53.80 | 50.60 | 22.53 | 55.7 | 0.9960 | 0.9904 |
| BF16 | 3776 | 54.01 | 50.81 | 22.74 | 42.4 | 1.0000 | 1.0000 |

Throughput scales as expected (Q4_K_M 1.56× BF16, Q8_0 1.31×); size scales 0.34× and 0.53×. Quality is
monotone in precision for chrF and chrF++, not for BLEU (Q8_0 22.53 < Q4_K_M 22.56, a 0.03 point
inversion consistent with decoding noise, not a real ranking).

**0.975 bar.** The bar applies because the ratio is a false-vs-true precision quality ratio with BF16 =
1.000 as the reference. chrF: Q4_K_M 0.9927, Q8_0 0.9960. chrF++: 0.9931 / 0.9960. BLEU: 0.9919 /
0.9904. **Minimum 0.9904 -> PASS (6/6 cells).** Both quants clear the bar by 1.5–2.1 points; the metric
is not the binding constraint at this precision range.

**Results — EN->PL comparison (25 rows, variant 0).**

| model | quant | chrF | BLEU* | out tokens | gen s | tok/s |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B | Q4_K_M | 41.77 | 9.12 | 8620 | 87.5 | 98.48 |
| Qwen2.5-3B-Instruct | Q4_K_M | 37.86 | 6.63 | 9476 | 136.4 | 69.46 |
| Qwen2.5-1.5B-Instruct | Q4_K_M | 35.01† | 6.62† | n/a | n/a | n/a |

\* BLEU is not stored anywhere; recomputed during the audit with `sacrebleu.BLEU()` on the same variant-0
rows (25 sentences). † Nothing is stored for the 1.5B run; chrF and BLEU were recomputed from
`out_qwen25_15b_pl.csv`. The 3B and 1.5B BLEU scores are within 0.002 of each other, so BLEU does not
separate them; chrF does (37.86 vs 35.01).

**0.975 bar (EN->PL): N/A.** No full-precision (BF16) run exists for either model, and the two entries
are different models, not two precisions of one model. A cross-model ratio would be meaningless here.

**Completeness.**

- Quant run: complete (3 models, 150/150 sentences each, checkpoint and final report agree).
- EN->PL 2B/3B: complete rows (25 × 4 = 100 each) but degenerate variants (see Caveats).
- EN->PL 1.5B: 100 rows exist; no timing and no score were kept (`tools/compare_pl.py` scores only the
  2B and 3B files).
- `cmp_*` / `seq*` CSVs: partial probe runs (10–12 inputs, 4 variants). `seq12.csv` and `seq_2507.csv`
  are the only ones with 12/12 distinct variants, i.e. taken after the re-seed fix.
- No run log exists for any of these jobs.

**Verdict.** Q4_K_M is a safe drop-in for BF16 on this task: it loses 0.39 chrF (0.7%) while running
1.56× faster at 34% of the size. Q8_0 adds 0.18 chrF over Q4_K_M for 1.6× the size and 0.84× the speed,
so it is dominated. The EN->PL ranking (2B > 3B > 1.5B on chrF) is a real observation about these
prompts, but it is a 25-sentence, single-seed result.

**Caveats.**

- **Reference encoding bug.** `load_flores` opens `flores200_en_it_de_fr.csv` as **latin-1**
  (`gguf_quant_quality.py:64`) although the file is UTF-8. All 150 Italian references are mojibake
  (`Ã¬`, `Ã¨`, `â€™`). A perfect hypothesis scores at most **97.4 chrF** against those strings, so every
  absolute number above is depressed by roughly 1.4–2.6 points. The ratios are almost unaffected because
  all three quants see the same damaged references.
- **Hypotheses not stored.** `hyps` is `null` in the JSON and only 3 example rows per model are kept. The
  table cannot be recomputed or given CIs from the committed files.
- **Identical variants.** All 25 rows of `out_qwen35_2b_pl.csv`, `out_qwen25_3b_pl.csv` and
  `out_qwen25_15b_pl.csv` have 4 byte-identical variants. `llm.load_state()` restores the sampler RNG, so
  every paraphrase repeated. `paraphrase_translate.py` now re-seeds per call, but these CSVs predate that
  fix and were not regenerated. The per-variant chrF block in `compare_pl.json` is therefore vacuous.
- **Sample size.** 150 sentences (quant) and 25 sentences (EN->PL), one seed, no intervals. A 0.39 chrF
  gap is inside plausible noise; only the direction (more precision -> slightly better) is supported.
- **Timing.** `wall_seconds` for Q4_K_M is 262.2 s against 81.5 s of generation, versus 155 s for Q8_0.
  The first model carries load/warm-up, so wall time is not a throughput measure.
- **No producer for the probe CSVs.** `cmp_*`, `seq*` and `out_*test*` come from ad-hoc commands; the
  `--para-sep` mode that made some of them was later removed.
- **Docs drift.** `TRANSLATE.md` still advertises `--para-sep` (removed in `b89bcef`) and "no max-token
  setting" (the CLI has `--para-max-new`/`--trans-max-new`). `tools/compare_pl.py` names Qwen2.5-1.5B in
  its docstring but scores the 3B file.
- Untracked, so not in a clone: `cmp_*.csv`, `seq12.csv`, `seq_2507.csv`, `out_div_test.csv`,
  `out_sep_test.csv`, `out_smoke_qwen25.csv`, `out_test_para.csv`, `out_test_q35_gguf.csv`,
  `gguf_quant_quality.partial.json` (gitignored by `*.partial.json`).

---

## Data and scripts map

Paths are relative to `multi/` unless marked. "Producing scripts" means the script that writes the
file, not every script that reads it.

| dataset | files | producing scripts | used by experiments |
|---|---|---|---|
| FLORES-200 sentences (en/it/de/fr) | `flores200_en_it_de_fr.csv` (2006 rows; 997 `dev` / 1009 `devtest`); `embeddings/{mean,bos,eos,lead}/emb_{en,it,de,fr}.npy` (2009×768); `embeddings_610/mean/...` (2009×1152); `emb_bf16/flores_sent/{210m,610m}/mean/...` | `extract_embeddings.py` (embeddings); `train_flores12.py` (`prepare_sentence`, `load_sent_embeddings`, splits) | E1/E1b, E3/E6/E7, E12/E13/E26, E14/E14b, E15, E17, E18, E19, E21, E22, E23, E25, E25b, E28, E29, E30, E31, E32, E35, E39, E40, E41, E42 |
| FLORES-agg (URL aggregate) | `flores_agg.csv` (562 rows, 11 kept classes); `embeddings_agg/{mean,...}/emb_*.npy` (562×768); `embeddings_agg_610/mean/...` (562×1152); `emb_bf16/flores_agg/{210m,610m}/mean/...` | `extract_embeddings.py`; URL split derived inside each consumer from `flores200_en_it_de_fr.csv` | E12/E13/E26, E14/E14b, E15, E17, E18, E19, E25, E25b, E28, E29, E33, E35, E39 |
| MultiEURLEX (en/de/fr) | `multieurlex/{en,de,fr}_{train,validation,test}.jsonl`; `multieurlex/emb_mean_{en,de,fr}_{train,test}.npy` (11000/5000×768); `emb_bf16/mel/{210m,610m}/...`; `multieurlex_610/` (present, read by no current script); `multieurlex_chunk/<lang>_<split>_{trunc,mean,max,concat}.npy` | `chunk_pool_mel.py` (chunk features); mel embeddings via `imbalanced_recall.py:285 ensure_mel_embeddings` (per E9 caveat); **no producer for `multieurlex_610/`** | E9/E10/E11/E16, E15, E17, E18, E25, E28, E34 |
| `toxi-text-3M` (raw, shipped splits) | HF cache revision `e1e50df2`; `%TEMP%\opencode\toxi_cache` (now empty) | `toxi_proof.py` (reader; no rebuild) | E36 |
| `toxi_clean` | `data/toxi_clean/{en,it,de,fr,es}_{train,validation,test}.parquet`; `data/toxi_clean/leakage_report.md` | `clean_toxi.py` | E20, E25, E25b, E27, E28, E29, E30, E37 |
| Toxic embeddings | `emb_bf16/toxic/{210m,610m}/emb_<lang>_<split>.npy`; `toxi_emb_210/` (separate, **no producer in repo**) | `extract_embeddings.py` (canonical); unknown for `toxi_emb_210/` | E12/E13/E26, E25, E25b, E27, E28, E29, E30, E37 |
| SST-2 / emotion (EN) | `sst2.csv` (54,577/6,822/6,822); `emotion.csv` (16,000/2,000/2,000); `sst2_emb/`, `emotion_emb/` (`{mean,bos}_{train,val,test}.npy`) | `rebuild_caches.py`; trial logs from `xgboost_sst2.py`, `xgboost_emotion.py` | E2, E4, E28, E40 |
| Translated SST-2 / emotion (it/de/fr) | `translated/{sst2,emotion}_{it,de,fr}_{validation,test}.csv` (12); `translated_emb/{dataset}_{mean,bos}_{lang}_{validation,test}.npy` (24) | `rebuild_caches.py`; `translate_datasets.py` is **missing** | E2, E4 |
| XNLI, PAWS-X, SIB-200, textdetox | HF datasets (`facebook/xnli`, `google-research-datasets/paws-x`, `Davlan/sib200`, textdetox); `%TEMP%\opencode\more_datasets\emb_cache` (60 `.npy`) | `more_datasets.py` (download, embed, cache); HF revisions not pinned | E12/E13/E26, E24, E25 |
| Europarl v7 (de/es/fr/it/nl/pt→en) | `europarl_all_5k.csv` (30,000 rows); `europarl_{xx}_en_5k_embeddings/<mode>/` (**directories empty**) | `fetch_europarl_langs.py` (sampling); `table1_europarl_multilang.py` (retrieval) | E39 |
| EN→PL sample | `flores_en_pl.csv` (25 rows) | `tools/fetch_flores_en_pl.py` | E38 |
| GGUF weights | `models/Qwen3.5-2B-{Q4_K_M,Q8_0,BF16}.gguf`, `models/qwen2.5-{1.5b,3b}-instruct-q4_k_m.gguf` | `tools/fetch_gguf.py`, `fetch_qwen25_gguf.py`, `fetch_qwen25_3b_gguf.py` | E38 |
| Model bundles | `models_toxi/{en,it,de,fr,es}_seed{0,1,2}.ubj` + `manifest.json`; `models/`; `models_registry.json` | `toxi_proof_clean.py`; `train_flores12.py`; `model_store.py` | E20, E27, E29 |
| Diagnostic figures | `pca_plots/*.png` (15); `lang_shap_dims_*.png`; `dim_removal_<dataset>.png` (8) | `pca_plots.py`, `lang_dims.py`, `shap_flores_*.py`; `lang_shap_dims.py`; `dim_removal.py` | E17, E41, E12/E13/E26 |
| MT / embedding caches (outside repo) | `%TEMP%\opencode\{flores_close_cache,translate_test_eval,mtcache_*,tt_cache,toxi_clean_cache,synth_it,synth_toxic_it,gen_compare,en_vs_enit_raw}` | the consuming scripts; no cache builder is tracked for `tt_cache` | E14, E22, E23, E30, E31, E32, E42 |

### Missing scripts and modules — NOT AVAILABLE (UNK)

Cited by the documentation but absent from the working tree and from every commit on every branch.
`git log --all -- <path>` is empty for each; they were untracked and are unrecoverable from git.

| missing script / module | cited by | nearest surviving equivalent |
|---|---|---|
| `langdims_probe.py` | E1/E1b | `lang_metrics.py` (η²) + `lang_shap_dims.py` probe |
| `lang_inlp.py` (+ `lang_inlp*.npz` bases) | E1/E1b | `mel_close.py` INLP block (`lang_inlp_mel.npz`); `dim_removal.py` INLP arm; `lang_neutralize.py` |
| `pc_ablation*.py` | E12/E13/E26 | `dim_removal.py` (consolidated reimplementation) |
| `crosslingual_topic*.py` | E3/E6/E7 | `flores_close.py`, `dim_removal.py`, `proof_mean_alllangs.py` (different protocols) |
| `crosslingual_inlp.py` | E3/E6/E7 | `dim_removal.py` INLP arm; `mel_close.py` INLP |
| `crosslingual_sentiment.py` | E4 | none; only `translated/` + `translated_emb/` survive |
| `translate_datasets.py` | E2 | `rebuild_caches.py` (supersedes it) |
| `multieurlex_*.py` | E9/E10/E11/E16 | `mel_close.py`; `chunk_pool_mel.py`; `imbalanced_recall_mel.py` |
| `agg_pr_methods.py` | `agg_pr_2models.py:313,746` (import) | none; only `--reemit-from-json` works |
| `agg_pr_lang.py` | `agg_pr_2models.py:12` (docstring) | provenance only |
| `pc_ablation_flores.py` | `agg_pr_2models.py:16,441`; `dim_removal.py:13` | `dim_removal.py` has its own copy |
| `crosslingual_topic_agg.py` | `agg_pr_2models.py:21,121,185` | provenance only; it produced the missing `lang_inlp_agg.npz` |

Missing artifacts in the same category: `lang_inlp.npz`, `lang_inlp_mean.npz`, `lang_inlp_agg.npz`,
`lang_inlp_agg_610.npz`, the `multieurlex_*` 610m outputs, `europarl_multilang_table1.md`,
`shap_flores_lang.{json,md}`, `shap_flores_dropped.{json,md}`, `imbalanced_recall_mel.{json,md,png}`
(v1), `translations of the E40 result tables` (archived at `aa3f6f6` only).

Two further gaps are **not** missing scripts but missing data: `toxi_emb_210/` has no producing script,
and `europarl_*_embeddings/` directories exist but are empty.

---

## Master result at 0.975

Read this as "what the audited record supports", not as a new analysis. All counts below are copied
from the entries; no ratio was recomputed at assembly.

### What passes

At the 0.975 bar, against a denominator that actually measures English-model transfer:

1. **Translate-test (E23)** is the broadest pass against the English model: **4/6 macro-F1** and
   **6/6 OVR-macro AUC** cells, and **6/6 + 6/6** against the in-language reference. The two misses
   are 210m it (0.957) and de (0.958).
2. **Translation strategies (E42)** pass. Translate-train E: E/B = **1.0057**, E/C = **0.9994**,
   E/A = 1.0683. Translate-test F: F/E = **1.0227** AUC, F/E macro-F1 = **1.2042**, F/A = 1.0926.
   F is the quality leader; E matches native Italian with no inference MT. D's 3002-row
   1.0586 / 1.0519 / 1.0483 is a **row-count effect**, not evidence that MT beats real Italian.
3. **Gap-closers against an in-language reference (E14/E14b)** pass on the primary micro-F1 metric for
   five methods — kNN (`l2_thenZ`, w10), translate-train, `mlp_align`, `affine_moment`,
   `ens_top3+tt` — 3/3 on both datasets. These are **method/reference** ratios, not target/English
   parity, and the threshold protocol is broken (see caveats).
4. **Synthetic Italian (E30, E31, E32)** passes against a real-Italian reference: E30 synth 0.9888 /
   0.9834 and mixed 1.0263 / 0.9859; E31 SmolLM2 0.994 and Qwen3.5-0.8B 0.980; E32 union 1.145 / 1.151
   (3× volume confound). Toxic does not pass anywhere.
5. **Quantization (E38)** passes 6/6 with a minimum of **0.9904** (BLEU) — a quant-vs-BF16 ratio.
6. **Training-regime and operating-point ratios** that are not transfer parity: E25b all-langs/per-lang
   **6/6**; E15 FLORES EN/native FPR **1.005** (marginal; 12/24 cells individually).

### What fails

- **E18 (out-of-the-box):** 0/8 target cells; best 0.967 (FLORES-agg IT). AUC at the old 0.90 bar was
  8/8; that result does not survive.
- **E24:** 1/13 cells (SIB-200 fr 0.995); 0/4 datasets. Textdetox is the clean failure (0.62–0.81).
- **E25:** 2/16 dataset × backbone rows, both PAWS-X, both near chance (1.024 / 0.994). Every other
  ratio is 0.562–0.954.
- **E20 (clean toxicity):** 0/4 on both target/EN and EN/native; best fr EN/native 0.896.
- **E27 / E37 (toxicity at recall 0.90):** 0/2 backbones (0.708 / 0.771) and 13/14 classifier cells;
  the one nominal pass (GNB 610m 1.0675) rests on the weakest native baseline.
- **E21 (classifiers):** F1 3/34, AUC 5/34; the 2/34 both-metric cells are degenerate QDA. No
  non-degenerate classifier passes both metrics.
- **E22 (MT):** 0/3 bilingual-vs-multilingual BLEU (0.778 / 0.719 / 0.756).
- **E4 (sentiment/emotion):** 0/12 target cells (best 0.819); numbers unverified.
- **E9/E10/E11:** 210m fails against both denominators; 610m passes only against the unverified native
  ceiling (0.996 / 0.993) and fails against English self (0.946 / 0.940).
- **E29 (fine-tune):** target/EN-self 0/20; FT/frozen 20/26 (6 fails).
- **E33 (agg P/R):** derived IT/EN tops at 0.9595; 0/10.
- **E34 (chunk pooling):** 0/24; best 0.9638.
- **E15 (MultiEURLEX recall):** 1.196, 2/16 cells.
- **E19 (bar feasibility):** sent 6/6 and agg 3/6 cells are *infeasible at the bar* — not a transfer
  failure but a finding that the 0.975 target may be above the in-language reference itself.
- **N/A, not passes:** E1/E1b, E2, E3/E6/E7, E12/E13/E26, E17, E28, E35, E39, E40, E41.

### Biggest caveats

1. **Missing scripts.** E1/E1b, E3/E6/E7 and E4 have no surviving script or result artifact. E9/E10/E11
   610m, E2's `translate_datasets.py`, E12's `pc_ablation*.py` and E33's `agg_pr_methods.py` are also
   gone. Any number from those lines is documentation-only.
2. **E9 610m is unverified.** The 0.711 / 0.707 row and the 0.714 / 0.712 native ceiling exist only as
   prose; `lang_matrix` corroborates only the macro-F1 range. The 210m `mel_close` artifact gives
   0.643/0.641 (tuned) or 0.661/0.662 (qm), not the table's 0.688/0.683.
3. **E29's toxic arm is not trained to convergence.** 400 steps = 12,800 samples = 53% of one EN epoch;
   early stopping cannot trigger; the LR is the grid edge. The toxic FT columns are undertrained and
   LR-confounded, and toxic "F1" is macro-F1 at 0.5, not F1(pos).
4. **E21, E28 and E32 omit material results.** E21 shows 5 of 17 models per backbone (KNN and two 610m
   AUC passers are omitted). E28's old table shows 20 of 48 cells. E32's toxic row is 2 of 3 generators,
   and its "same-size ceiling" is false (4503 vs 1501 rows).
5. **E14's threshold protocol is broken.** The code fits on train+val and tunes the threshold on val,
   which is part of the fit set; val F1 is 1.0000, so the tuner returns 0.04. Every XGB row and every
   reference is depressed relative to an honest protocol, and kNN is the only out-of-sample threshold.
6. **E15's mel label bug.** v1 trained the native de/fr models with EN labels on non-parallel train
   splits; v2 fixed it. The v1 outputs are missing and v1 is not reconstructible. v1 and v2 also
   differ in device.
7. **The E27 table is orphaned.** No committed artifact contains 0.324 / 0.448 / 0.276 or 0.359 / 0.460
   / 0.278; the deltas are wrong (EN/EN +0.002 stated, +0.016 actual). Use `toxi_pr_all.json`.
8. **E36 is leaky-split provenance only.** 84.7–99.9% of eval-test texts appear verbatim in the shipped
   train CSV, and the test positive rates contradict the train pool (en 0.625 vs 0.117). Its two 0.975
   passes are artifacts. Quote E20.
9. **Small tests, one seed, no CIs.** FLORES test is 505 (sent) / 139 (agg); MultiEURLEX is 5,000
   parallel documents; SIB-200 is 204; textdetox is 1,000. A 0.975 ratio is often inside the noise band.
10. **The 0.975 bar is applied to at least six different denominators** (target/EN, EN/native,
    target/ceiling, method/reference, all-langs/per-lang, quant/BF16). A pass in one is not a pass in
    another. E19 shows the sent verdict flips completely between train+val and train-only English self.

---

## Known gaps and next steps

Ordered by what blocks a defensible 0.975 claim, not by cost.

1. **Close the missing-script list.** Either recover the eight documented scripts or rewrite them from
   the surviving primitives (`lang_neutralize.py`, `dim_removal.py`, `mel_close.py`,
   `chunk_pool_mel.py`) and re-run. Until then E1/E1b (INLP half), E3/E6/E7 and E4 stay **UNK**. Do not
   present their numbers in a verified table.
2. **Settle E9/E10/E11 610m.** Rebuild `multieurlex_*.py` or delete the 610m row and the native ceiling
   from the entry. The current 0.996 / 0.993 pass is against an unverified denominator. Also state the
   train protocol per row (10k vs 11k) and stop calling the in-language model a "ceiling".
3. **Finish or delete E29's toxic arm.** Train at least one full epoch, record the caps in the JSON and
   the log consistently, and report F1(pos) as well as macro-F1. The current 400-step arm cannot support
   either a pass or a fail.
4. **Repair E14/E14b's threshold protocol.** Tune the threshold on a held-out split, not on the val split
   that is part of the fit set. Re-run the method table; the current numbers are not comparable across
   method families.
5. **Replace the E27 table with `toxi_pr_all.json` values and recompute the deltas.** Label EN/EN as a
   backbone control, not transfer. Keep the per-language-z and raw pipelines in separate columns. Fix
   the `?` delta header.
6. **E21: report all 17 models or justify the 5.** Exclude QDA (or set `reg_param` and log the rank) and
   report EN→IT and IT→EN separately. The current 5-row table hides KNN and two 610m AUC passers.
7. **E28: publish the 48-cell grid.** The omitted 210m rows change the story (FLORES-agg 210m mean is
   best; mel 210m raw is 0.059). Document that INLP is not bit-reproducible and do not rank it.
8. **E32: fix the "same-size ceiling" wording and finish the toxic arm.** The union has 3× the rows; the
   matched-volume control is *below* the best single. SmolLM2 toxic stopped at 4800/8000.
9. **E31/E30: finish the toxic generation.** SmolLM2 toxic is missing; E30's toxic arm is a cap-160
   plumbing run with a 115× size mismatch. Neither supports a 0.975 statement.
10. **E38: fix the latin-1 reference bug and store hypotheses.** All absolute chrF/BLEU numbers are
    depressed by 1.4–2.6 points. The quant ratios survive; the absolute table does not.
11. **E33: restore `agg_pr_methods.py` or archive the re-emit path as the only supported mode.** Record
    that the full rerun is unavailable.
12. **E34: quarantine the smoke outputs.** `chunk_pool_mel_smoke.*` do not reproduce; the trunc/mean
    cells do.
13. **E35: store the agg `list_errors` run.** Only the sent en→it run is in the JSON; the agg/de runs
    left only MT caches.
14. **E36: keep it as provenance, never as evidence.** Add a one-line header to `toxi_proof.md` saying
    it is the leaky-split predecessor of E20.
15. **E40: pin the archive.** Re-runs need the `aa3f6f6` CSV, not `196bbd3` and not the current 2006-row
    file. Without it every entry point aborts at the row-count assert.
16. **E41: re-run the SHAP scripts after fixing row alignment.** The SHAP JSON/MD and the per-dim
    vectors are lost; only Spearman and approximate bar heights survive.
17. **Close the E13/E26 IDs.** They have no artifact and no separate claim. Either merge them into E12
    permanently or mark them retired.
18. **State the denominator on every pass.** Add a `denominator` column to any future summary table.
    E19 is the cautionary case: the same cells are feasible or infeasible depending on which English
    self is used.
