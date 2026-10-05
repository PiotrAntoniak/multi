# EuroBERT-210m cross-lingual embeddings — retrieval, topic classification, inference

Sentence embeddings of **FLORES-200** (`en`, `it`, `de`, `fr`; 2009 aligned sentences) and of
**Europarl v7** (six languages against English: de, es, fr, it, nl, pt; 5,000 aligned pairs each)
computed with **EuroBERT-210m**, plus exact-translation retrieval (foreign-to-English, raw and
`+D`), an XGBoost topic classifier with cross-lingual transfer and a standalone inference tool,
and a token/cosine diagnostic.

Retrieval scope is **foreign-to-English only**: FLORES has 3 directions (`it/de/fr -> en`), and
Europarl `xx -> en` for all six languages; both report **raw and `+D`** for every pooling mode.
Generated artifacts (training reports, Optuna trials, prediction CSVs) are **not stored** in the
repo: the scripts write them next to themselves and they are git-ignored.

## Repo contents

| file | what it is |
|---|---|
| `flores200_en_it_de_fr.csv` | FLORES-200 data: 2006 rows (997 `dev` / 1009 `devtest`), columns `split, id, en, it, de, fr, URL, domain, topic, has_image, has_hyperlink`; topic labels canonicalized (case/duplicate variants merged, one topic per URL) |
| `flores_agg.csv` | FLORES aggregated by URL: 562 rows (one per article), merged `en/it/de/fr` texts in article order + canonical topic, `n_sentences` |
| `europarl_all_5k.csv` | all Europarl data in one file: 30,000 rows (6 languages x 5,000 seeded, deduplicated line-aligned pairs), columns `lang, ...` |
| `sst2.csv` | SST-2 (GLUE) data: 68,221 rows (67,349 train / 872 validation), columns `sentence, label, split` |
| `emotion.csv` | dair-ai/emotion data: 18,000 rows (16,000 train / 2,000 validation), columns `text, label, split` |
| `extract_embeddings.py` | embeds FLORES in the four pooling modes -> `embeddings/<mode>/emb_<lang>.npy` |
| `table1.py` | FLORES exact-translation retrieval, foreign-to-English, raw + `+D` |
| `table1_agg.py` | the same retrieval protocol on the aggregated URLs (`flores_agg.csv`) |
| `fetch_europarl_langs.py` | downloads/extracts the six Europarl xx-en pairs and (re)writes `europarl_all_5k.csv` |
| `table1_europarl_multilang.py` | Europarl retrieval (`xx -> en`, all modes, raw + `+D`); reuses already-computed embeddings |
| `xgboost_topic_optuna.py` | XGBoost topic training: lenient labels, per-tag tuned thresholds, per-language grid, cross-lingual transfer |
| `xgboost_infer.py` | standalone inference: embedded per-mode default params, `--shift`, `--eval`, `--input` |
| `xgboost_sst2.py` | XGBoost binary sentiment on EuroBERT embeddings (`mean`/`bos`), Optuna-tuned; reads `sst2.csv`, caches embeddings to `sst2_emb/` |
| `xgboost_emotion.py` | XGBoost 6-class emotion on EuroBERT embeddings (`mean`/`bos`), Optuna-tuned (val AUC objective) with a per-emotion report; reads `emotion.csv`, caches to `emotion_emb/` |
| `check_tokens_cos_sim.py` | token-norm and cosine-similarity diagnostic |
| `check_bos_diagnostics.py` | per-layer BOS / mean-pooling stats and the BOS ablation (15 en + 15 it FLORES sentences; full column glossary in the script docstring) |
| `watch_progress.py` | minute-refresh progress board for the single-mode XGBoost runs (rewrites `progress_live.txt` every 60 s) |
| `pca_plots.py` | PCA plots of the FLORES embeddings (raw/+D per pooling mode, all-but-the-top variants, topic view) -> `pca_plots/*.png` |
| `lang_dims.py` | per-dimension language separability (eta^2) and the cumulative "how many dims carry the language" curve |
| `lang_metrics.py` | shared helpers: per-dim language eta^2 and top-k dim selection (used by the analysis scripts) |
| `model_store.py` | save/load trained model bundles (`save_bundle`/`load_bundle`/`list_bundles`): native `.ubj` per tag + `manifest.json`; bundles live in `models/`, which is git-ignored |
| `shap_flores_lang.py` | SHAP on the unrestricted FLORES baseline: does the topic model use the language-separable dims? (Spearman, top-50 overlap, SHAP mass vs random) -> `pca_plots/shap_vs_langdims_*.png` |
| `shap_flores_dropped.py` | held-out en/it models: sum |SHAP| of the dropped eta^2 dims vs kept dims (share vs random), on en-test and it-test inputs -> `pca_plots/shap_dropped_vs_kept_*.png` |
| `xgboost_flores_lang.py` | FLORES en/it per-language reference + language-dim ablation (train en with the top language dims zeroed -> eval it) |
| `xgboost_flores_combined.py` | FLORES en/it follow-up: combined en+it training and en->it raw vs `+D` transfer (fixed 0.5 and in-sample tuned thresholds) |
| `xgboost_flores_matrix.py` | FLORES en/it full-row matrix: train full en / full it, cross-evaluate; train en without language dims -> it raw |
| `xgboost_flores_drop_optuna.py` | per-language Optuna protocol on embeddings with the top-50% language dims zeroed (4 cells: en/it x mean/bos) |
| `train_flores12.py` | 12-class single-label FLORES topic driver over BOTH datasets (`flores200_en_it_de_fr.csv` sentence + `flores_agg.csv` URL-agg): full and prune50 embeddings, per-language (en/it) Optuna (10 trials), cross-language transfer, 16 cells; saves bundles to `models/` and updates the tracked `models_registry.json` |
| `models_registry.json` | tracked registry of `train_flores12.py` models (one entry per bundle: dataset, language, mode, condition, k, val/test macro-F1); the `models/` bundles themselves stay git-ignored |

EuroBERT-210m has no dedicated classification token and its tokenizer does not prepend a leading
special token. We therefore materialize **four pooling modes** over the last hidden state:

| mode | definition |
|---|---|
| `mean` | masked mean over the plain tokenization (content + appended end token) |
| `lead` | last hidden state at position 0 of the plain forward (no prepend) |
| `bos`  | position 0 of a forward whose input has the leading special id `128000` manually prepended |
| `eos`  | last non-pad position of the plain forward (the appended end token) |

The retrieval protocol uses a **language-shift vector `D`**: for a directed pair `a -> b`,
`D = mean over dev rows of (emb_a - emb_b)`. Meaning cancels in that difference, leaving the
cross-lingual shift. Retrieval is evaluated on the `devtest` rows: the query is `emb_a`, the
candidate keys are `emb_b` ("raw"); adding `D` to every key ("+D") moves each key toward the query
side. Embeddings are row-normalized so dot products are cosine similarities, and the rank of query
`i` is `1 + #{keys strictly closer than its exact translation key i}` (rank 1 = nearest).

## The shift vector `D` — derivation

For a parallel sentence `i` (same meaning in every language), model each language's embedding as

```
emb_a(i) = lang_a + meaning(i) + eps_a(i)
emb_b(i) = lang_b + meaning(i) + eps_b(i)
```

where `lang_*` is a sentence-independent language offset and `meaning(i)` is the shared content.
Taking the paired difference makes the meaning term cancel:

```
emb_a(i) - emb_b(i) = (lang_a - lang_b) + (eps_a(i) - eps_b(i))
```

and averaging over the 997 `dev` sentences (the residual noise is approximately zero-mean) leaves

```
D_ab = (1/997) * sum_i ( emb_a(i) - emb_b(i) )  ~=  lang_a - lang_b
```

So `D` is a closed-form estimate of the pure language offset: one 768-dimensional vector,
independent of the sentence, obtained with two means and a subtraction (no gradients, no training).
At test time, `key_b + D` moves every language-`b` candidate toward the `a` side, which is why the
"+D" columns improve on "raw". The `eps` term is also exactly why `D` cannot be perfect: it is a
single constant offset while the true per-sentence shift fluctuates, leaving the small rank-1
misses visible in the tables.

## Run end-to-end

```bash
pip install -r requirements.txt
python extract_embeddings.py           # downloads EuroBERT-210m -> embeddings/<mode>/emb_<lang>.npy
python table1.py                       # FLORES retrieval: foreign->EN, raw vs +D
python fetch_europarl_langs.py         # downloads the six Europarl xx-en pairs -> europarl_all_5k.csv
python table1_europarl_multilang.py all   # Europarl retrieval xx->en for all six languages
python xgboost_topic_optuna.py --mode all --labels lenient --thresholds tuned --trials 50
python xgboost_topic_optuna.py --mode perlang --langs en,it,de,fr --labels lenient --thresholds tuned --trials 50
python xgboost_infer.py --mode mean --lang it --shift --eval
python xgboost_sst2.py --modes mean,bos --trials 50       # SST-2 (binary), mean + bos pooling
python xgboost_emotion.py --modes mean,bos --trials 50    # dair-ai/emotion (6-class)
python xgboost_flores_drop_optuna.py       # language-dim ablation: top-50% eta^2 dims zeroed (4 cells)
python check_tokens_cos_sim.py         # token norms + cosine diagnostic
python check_bos_diagnostics.py        # per-layer BOS/mean stats + BOS ablation
```

`table1.py` is standalone (csv + numpy only). `check_tokens_cos_sim.py` re-loads EuroBERT-210m and
inspects the tokenizer/embedding matrix.

## Table 1 — FLORES retrieval (foreign-to-English)

Printed by `python table1.py` (dev=997, devtest N=1012, chance rank 506.5):

| mode | top-1 raw | top-1 +D | MRR +D | median rank +D |
|---|---:|---:|---:|---:|
| mean | 38.5-60.6% | 83.9-90.0% | 0.882-0.931 | 1 |
| lead | 7.3-22.6% | 11.1-25.9% | 0.155-0.325 | 22-128 |
| bos | 2.5-6.2% | 4.4-10.7% | 0.075-0.150 | 116-224 |
| eos | 2.0-2.5% | 10.0-15.8% | 0.165-0.230 | 36-62 |

Every cell is the min–max over the three foreign-to-English directions (`it->en`, `de->en`,
`fr->en`). Setup: 997 dev pairs fit the shift vector `D`; 1012 devtest queries are matched
against the 1012 candidate keys. **top-1** — share of queries whose exact translation is the
single nearest key (chance 0.1%); **+D** — every key shifted by `D`; **MRR** — mean of `1 / rank`
of the exact translation; **median rank** — median rank of the exact translation (chance 506.5;
robust to the tail, unlike the arithmetic mean, which a few badly misplaced sentences dominate).

Reading:

- **Mean pooling dominates.** Raw retrieval is already 38.5–60.6% top-1; after adding `D` it reaches
  **83.9–90.0% top-1** with MRR **0.882–0.931** and mean rank **1.6–4.4**. It is the only pooling
  that produces usable exact-translation retrieval.
- **Lead / bos / eos fail.** `bos` position 0 is dominated by the leading special id, whose input
  embedding is essentially untrained (id `128000` row norm `0.00280`, the 313th-smallest of all
  128,256 embedding rows); `eos` distances saturate; `lead` degenerates (raw ranks ~190–266).
- **`D` works and is not trained.** A single closed-form vector improves every mode's metrics.

### Table 1-agg — the same protocol on the aggregated URLs

`table1_agg.py` reruns the retrieval on the merged per-URL texts (`flores_agg.csv`, 562 rows;
dev = 281, devtest N = 280, chance rank 140.5; the empty orphan row is excluded). Longer texts
sharpen the embeddings:

| mode | top-1 raw | top-1 +D | MRR raw | MRR +D | median rank +D |
|---|---:|---:|---:|---:|---:|
| mean | 92.9-98.9% | 98.2-100.0% | 0.962-0.994 | 0.989-1.000 | 1 |
| bos | 4.3-13.2% | 11.4-21.1% | 0.076-0.201 | 0.158-0.298 | 11-60 |

`mean` reaches near-perfect retrieval even raw (92.9-98.9% top-1) and saturates with `+D`
(de->en 100.0% top-1, MRR 1.000); `bos` stays weak. Cells are min-max over it/de/fr -> en.

## Europarl — foreign-to-English

Second corpus: **Europarl v7** (European Parliament proceedings), shipped as one combined file
(`europarl_all_5k.csv`, `lang` column; dev = 1,000 / devtest = 4,000 per language). All four
pooling modes are embedded; `D` is fitted on dev and added to the keys. `xx -> en`, raw and `+D`,
4,000 queries vs 4,000 keys (chance rank 2,000.5):

| pair | top-1 raw | top-1 +D | MRR raw | MRR +D | mean rank raw | mean rank +D | median rank raw | median rank +D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| de-en | 4.2% | 41.5% | 0.102 | 0.493 | 177.2 | 111.4 | 54 | 3 |
| es-en | 23.6% | 66.9% | 0.355 | 0.723 | 62.0 | 60.2 | 6 | 1 |
| fr-en | 22.6% | 62.3% | 0.341 | 0.686 | 53.6 | 60.1 | 7 | 1 |
| it-en | 19.6% | 45.6% | 0.304 | 0.527 | 100.2 | 118.4 | 10 | 2 |
| nl-en | 6.0% | 45.9% | 0.167 | 0.531 | 109.9 | 115.1 | 14 | 2 |
| pt-en | 21.5% | 50.6% | 0.324 | 0.577 | 98.7 | 101.8 | 8 | 1 |

Other poolings stay near chance, as on FLORES (`+D` `xx->en` top-1: lead ~7–12%, bos ~1–3%,
eos ~3–7%). The same pattern replicates — `D` is the dominant lever (es 23.6% → 66.9%, de 4.2% →
41.5%), Romance pairs (es/fr/pt/it) transfer better than Germanic (de/nl), and **the median rank
collapses (6–54 → 1–3)** even where the arithmetic mean rank ticks up: the mean is dominated by a
small tail of sentences that the constant shift pushes deep, while the typical sentence moves to
the very front.

## XGBoost topic classifier (FLORES)

Multi-label XGBoost over the 8 canonical topic tags, with the pooling embeddings as features.
Split: dev 997 / devtest 506 / test 506 (seed-0 shuffled halves of the original devtest). Every
Optuna trial (TPE, 50 trials, one shared search space) trains on dev (997) and scores macro-F1 on
`devtest`; the best params are retrained on dev+devtest (1503) and evaluated on `test`, and a final
copy trains on all 2009 English rows for cross-lingual transfer.

**Labels: lenient.** Raw comma-tags are mapped onto the 8 canonical tags with curated aliases
implemented in `xgboost_topic_optuna.py` (`LENIENT_EXACT/PREFIX/CONTAINS`: singular/plural,
`crime`/`law` → `crime and law`, `science and technology` → `science`, the misspelling `sciece`,
`tourism`/`accomodation` → `travel`, `disasters and accidents` → `safety`, country/continent tags →
`geography`). This recovers the large fraction of positives that exact-string matching drops and
cuts all-zero rows from 1147/2009 (strict) to ~504/2009. **Thresholds: one per tag**, tuned
in-sample (grid 0.05..0.95) instead of a global 0.5.

Per-language × pooling grid, tuned `test` macro-F1 (`devtest → test`):

| language | mean | bos | eos | lead |
|---|---:|---:|---:|---:|
| en | 0.416 → 0.528 | 0.305 → 0.298 | 0.348 → 0.370 | 0.272 → 0.341 |
| it | 0.344 → 0.464 | 0.226 → 0.352 | 0.314 → 0.352 | 0.171 → 0.222 |
| de | 0.392 → 0.470 | 0.283 → 0.322 | 0.304 → 0.344 | 0.202 → 0.221 |
| fr | 0.411 → 0.503 | 0.248 → 0.251 | 0.281 → 0.365 | 0.235 → 0.246 |

`mean` wins every language — tuned `test` macro-F1 **en 0.528 / it 0.464 / de 0.470 / fr 0.503** —
and is also best on micro-F1 and subset accuracy. Per-topic accuracy: `mean` leads on every topic
in every language except a few ties (e.g. it `health` eos 0.966, de `geography` bos 0.917).

**Transfer (train on all 2009 English rows, evaluate it/de/fr raw and `+D`):** `mean` + `D` is best
(tuned transfer macro-F1 `mean+D` 0.345 / 0.327 / 0.400 for it / de / fr, vs `mean` raw
0.278 / 0.298 / 0.381); the shift gain concentrates on `travel` (+0.187 accuracy for fr, +0.081
  it, +0.053 de) and `science` (+0.020/+0.024/+0.006), while `bos` loses on science/safety.

### Language-dim ablation (Optuna protocol)

Same protocol as the grid above, but the top-k dims carrying 50% of the 4-language eta^2 mass are zeroed in every array (k=237 `mean`, 211 `bos`). After each cell's retrain the driver saves that cell's 8 tag models as a bundle under `models/` (via `model_store.py`, git-ignored) so transfer/SHAP/inference evaluations can load them instead of retraining. Tuned `test` macro-F1:

| language | mean | bos |
|---|---:|---:|
| en | 0.5093 | 0.3585 |
| it | 0.4773 | 0.2911 |

Versus the full-feature grid (en 0.528 / 0.298, it 0.464 / 0.352): deltas -0.019 / +0.061 (en) and +0.013 / -0.061 (it) - no consistent effect. This matches the SHAP result: the language dims carry no topic signal, so removing them is near-neutral.

### Inference

`xgboost_infer.py` is standalone: it retrains the 8 tag models deterministically with the
per-mode default params (the 50-trial lenient Optuna best params, embedded in the script) and
tunes the thresholds in-sample; `--params-json FILE` overrides the params, `--thresholds 0.5`
forces the fixed threshold. It writes `predictions_*.csv` (git-ignored).

```bash
python xgboost_infer.py --mode mean --lang it --shift --eval     # English transfer models, it + D
python xgboost_infer.py --source perlang --lang en --mode mean --eval
python xgboost_infer.py --mode mean --input my_emb.npy --eval    # custom (n, 768) float32 features
```

`--source transfer` (default) trains on all 2009 English rows; `--source perlang` trains on
`dev+devtest` of `--lang`; `--shift` adds `D_lang = mean(emb_en − emb_lang)`. `--eval` prints
per-tag accuracy/F1 and subset accuracy against the gold lenient labels (repo embeddings only).

### Earlier results — strict labels, global 0.5 threshold

Kept for reference; the lenient results above supersede them. Labels are exact canonical strings
and every tag uses a single 0.5 threshold. Cells are **subset accuracy / macro-F1**; each transfer
cell is `raw → +D` for that language.

| pooling | English test base → tuned | it raw → +D | de raw → +D | fr raw → +D |
|---|---:|---:|---:|---:|
| **mean** | 0.672/0.256 → 0.666/0.333 | 0.595/0.048 → 0.634/0.100 | 0.589/0.068 → 0.637/0.081 | 0.642/0.190 → 0.647/0.120 |
| eos | 0.632/0.146 → 0.615/0.229 | 0.586/0.041 → 0.593/0.084 | 0.572/0.003 → 0.594/0.062 | 0.525/0.074 → 0.603/0.098 |
| bos | 0.607/0.105 → 0.597/0.164 | 0.565/0.080 → 0.579/0.040 | 0.578/0.107 → 0.592/0.094 | 0.568/0.078 → 0.577/0.068 |
| lead | 0.607/0.097 → 0.615/0.170 | 0.583/0.056 → 0.581/0.026 | 0.580/0.062 → 0.593/0.043 | 0.592/0.067 → 0.588/0.047 |

Strict-label reading: `mean` leads again — best English test score and best raw transfer — and the
shift lifts it in every language on **subset accuracy** (it 0.595 → 0.634, de 0.589 → 0.637,
fr 0.642 → 0.647) while macro-F1 is mixed (+0.052 it, +0.013 de, −0.070 fr). Rare tags have very
few positives, so macro-F1 is noisy and dominated by `travel`/`sports`. Optuna selects on the same
`test` set it reports (selection-set scores, per protocol).

## SST-2 and emotion (EuroBERT embeddings)

Two more classifiers on the same embeddings (`mean` / `bos`), each tuned with 50 Optuna trials.

SST-2 (binary sentiment, 80/10/10 re-split: 54,577 / 6,822 / 6,822; objective = val accuracy):

| mode | val acc | test acc | test macro-F1 | test AUC |
|---|---:|---:|---:|---:|
| mean | 0.8147 | 0.8080 | 0.8052 | 0.8943 |
| bos | 0.7366 | 0.7382 | 0.7338 | 0.8144 |

dair-ai/emotion (6 classes, 16,000 / 2,000 / 2,000; objective = val AUC, OVR macro):

| mode | val AUC | test acc | test macro-F1 | test AUC |
|---|---:|---:|---:|---:|
| mean | 0.8207 | 0.5635 | 0.3740 | 0.8196 |
| bos | 0.7264 | 0.4780 | 0.2478 | 0.7241 |

`mean` wins both. Emotion is class-imbalanced (surprise 3.6%); the mean model's test per-class AUC stays 0.78-0.85 for all six classes, while rare-class F1 is low (surprise 0.06).

## `check_tokens_cos_sim.py`

This diagnostic prints two things.

**1. Input-embedding row norms.** Special ids and the reserved range behave very differently:

```
id 128000 <|begin_of_text|>: norm 0.00280
id 128001 <|end_of_text|>:   norm 0.34248
id 128002 <|mask|>:          norm 0.48082
median row norm 0.21375 | mean row norm 0.20626
reserved 128003+ median 0.00271 min 0.00256 max 0.00290   (all < 0.01)
```

The `<|mask|>` embedding (`128002`) is the **largest** special-id row and even exceeds the median
row norm; `<|begin_of_text|>` (`128000`) is nearly zero (untrained), as is the entire reserved
range `128003+`. This is direct evidence for why `bos` pooling (which reads position 0 after
prepending `128000`) is uninformative.

**2. Per-pooling cosine on example pairs.** Averages over three unrelated and three related pairs:

| pooling | unrelated | related | behavior |
|---|---|---|---|
| `mean` | +0.391 | +0.752 | orders relatedness correctly (related well above unrelated) |
| `lead` | +0.650 | +0.770 | degenerate |
| `bos`  | +0.379 | +0.310 | inverted |
| `eos`  | +0.773 | +0.769 | saturated |

Only `mean` separates related from unrelated pairs; the same picture as Table 1.
