# EuroBERT-210m cross-lingual sentence embeddings & exact-translation retrieval

Sentence embeddings of **FLORES-200** (`en`, `it`, `de`, `fr`; 2009 aligned sentences) and of
**Europarl v7** (six languages against English: de, es, fr, it, nl, pt; 5,000 pairs each) computed
with **EuroBERT-210m**, plus exact-translation retrieval evaluations (foreign-to-English), an
XGBoost topic-transfer experiment with a deterministic inference tool, and a token/cosine
diagnostic.

Retrieval scope is **foreign-to-English only**: FLORES has 3 directions (`it/de/fr -> en`) and
Europarl `xx -> en`; both report **raw and `+D`** for every pooling mode.

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
cross-lingual shift. Retrieval is evaluated on the `devtest` rows only (1012 queries, 1012
candidate keys): the query is `emb_a`, the keys are `emb_b` ("raw"); adding `D` to every key
("+D") should move each key onto the query side. Embeddings are row-normalized so dot products are
cosine similarities, and the rank of query `i` is `1 + #{keys strictly closer than its exact
translation key i}` (rank 1 = nearest). `D` is a single closed-form dev-fitted vector — it is not
trained by gradient descent.

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
"+D" columns should improve on "raw". The `eps` term is also exactly why `D` cannot be perfect: it
is a single constant offset while the true per-sentence shift fluctuates, leaving the small
rank-1 misses visible in the tables.

## Run end-to-end

```bash
pip install -r requirements.txt
python extract_embeddings.py      # downloads EuroBERT-210m, writes embeddings/<mode>/emb_<lang>.npy
python table1.py                  # prints Table 1 (FLORES foreign-to-English retrieval ranges)
python fetch_europarl_langs.py    # downloads + extracts five Europarl pairs (de/es/fr/nl/pt; it-en is shipped as csv)
python table1_europarl_multilang.py all   # Europarl: 5k pairs/lang, all modes, xx->en raw + D table
python xgboost_topic_optuna.py --mode all --labels lenient --trials 50   # topic classifier: pooled-mode comparison
python xgboost_topic_optuna.py --mode perlang --labels lenient --thresholds tuned   # per-language x pooling grid
python xgboost_infer.py --mode mean --source perlang --lang en --eval   # inference from stored params
python check_tokens_cos_sim.py    # prints input-embedding norms and per-pooling cosine similarities
```

`extract_embeddings.py` writes float32 `(2009, 768)` arrays for the four modes and four languages.
`table1.py` is standalone (csv + numpy only) and depends only on the produced `.npy` files and the
CSV. `check_tokens_cos_sim.py` re-loads EuroBERT-210m and inspects the tokenizer/embedding matrix.

Data: `flores200_en_it_de_fr.csv` (2009 rows: 997 `dev`, 1012 `devtest`) with columns
`split, id, en, it, de, fr, URL, domain, topic, has_image, has_hyperlink`.

## What Table 1 shows

Printed by `python table1.py`:

**Table 1 - devtest protocol** | dev=997 | devtest N=1012 | 3 foreign-to-English directions | chance rank 506.5

| mode | top-1 raw | top-1 +D | MRR raw | MRR +D | mean rank raw | mean rank +D | median rank raw | median rank +D |
|---|---|---|---|---|---|---|---|---|
| mean | 38.5-60.6% | 83.9-90.0% | 0.537-0.763 | 0.882-0.931 | 2.1-8.5 | 1.6-4.4 | 1-2 | 1-1 |
| lead | 7.3-22.6% | 11.1-25.9% | 0.118-0.285 | 0.155-0.325 | 189.7-265.6 | 189.6-267.1 | 33-139 | 22-128 |
| bos | 2.5-6.2% | 4.4-10.7% | 0.044-0.098 | 0.075-0.150 | 280.0-364.6 | 257.0-330.0 | 166-291 | 116-224 |
| eos | 2.0-2.5% | 10.0-15.8% | 0.049-0.068 | 0.165-0.230 | 200.4-259.8 | 154.8-190.3 | 107-164 | 36-62 |

Column by column:

- **top-1** — percentage of the 1012 devtest queries whose exact translation is the single nearest
  of the 1012 candidate keys.
- **MRR** — mean of `1 / rank` over queries.
- **mean rank** — average rank of the exact translation (random chance is `(1012 + 1) / 2 = 506.5`).
- **median rank** — the median rank; robust to the tail, whereas the arithmetic mean is dominated by
  a few badly misplaced sentences.
- **raw vs +D** — before vs after adding the dev-fitted shift `D` to every candidate key.
- **ranges** — min–max over the three **foreign-to-English** directions (`it->en`, `de->en`, `fr->en`).

## What the experiment shows

- **Mean pooling dominates.** Raw foreign-to-English retrieval is already 38.5–60.6% top-1; after
  adding `D` it reaches **83.9–90.0% top-1** with MRR **0.882–0.931** and mean rank **1.6–4.4**
  (from 2.1–8.5 raw). It is the only pooling that produces usable exact-translation retrieval.
- **Lead / bos / eos fail.**
  - `bos` position 0 is dominated by the leading special id, whose input embedding is essentially
    untrained: id `128000` has row norm `0.00280` and is the **313th-smallest of all 128,256
    embedding rows**, so the bos vector carries almost no sentence signal.
  - `eos` distances saturate — rows are extremely close to one another regardless of content.
  - `lead` degenerates; raw retrieval ranks are ~190–266.
- **`D` works and is not trained.** It is a single closed-form vector (`mean(emb_a - emb_b)` over
  997 dev rows), yet adding it improves every mode's metrics, most dramatically for `eos`
  (top-1 1.8–7.5% -> 9.1–21.1%, mean rank 155.7–290.0 -> 105.7–194.9).

## Europarl (parliament corpus) — foreign-to-English

Second corpus: **Europarl v7** (European Parliament proceedings). For each of six pairs against
English (de, es, fr, it, nl, pt): 5,000 seeded, deduplicated line-aligned pairs, dev = 1,000 /
devtest = 4,000. All four pooling modes are embedded; `D` is fitted on dev and added to the keys.
Retrieval is evaluated **foreign-to-English only** (`xx->en`), **raw and `+D`**, 4,000 queries vs
4,000 keys (chance rank 2,000.5).

| pair | top-1 raw | top-1 +D | MRR raw | MRR +D | mean rank raw | mean rank +D | median rank raw | median rank +D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| de-en | 4.2% | 41.5% | 0.102 | 0.493 | 177.2 | 111.4 | 54 | 3 |
| es-en | 23.6% | 66.9% | 0.355 | 0.723 | 62.0 | 60.2 | 6 | 1 |
| fr-en | 22.6% | 62.3% | 0.341 | 0.686 | 53.6 | 60.1 | 7 | 1 |
| it-en | 19.6% | 45.6% | 0.304 | 0.527 | 100.2 | 118.4 | 10 | 2 |
| nl-en | 6.0% | 45.9% | 0.167 | 0.531 | 109.9 | 115.1 | 14 | 2 |
| pt-en | 21.5% | 50.6% | 0.324 | 0.577 | 98.7 | 101.8 | 8 | 1 |

Other poolings stay near chance, as on FLORES (`+D` `xx->en` top-1: lead ~7–12%, bos ~1–3%,
eos ~3–7%). Reading: the same pattern replicates — `D` is the dominant lever (es 23.6% → 66.9%,
de 4.2% → 41.5%), Romance pairs (es/fr/pt/it) transfer better than Germanic (de/nl), and **the
median rank collapses (6–54 → 1–3)** even where the arithmetic mean rank ticks up — the mean is
dominated by a small tail of sentences that the constant shift pushes deep, while the typical
sentence moves to the very front.

## XGBoost topic classifier — English training, cross-lingual transfer

Multi-label XGBoost over the 8 canonical topic tags, with the pooling embeddings as features.
Split: dev 997 / devtest 506 / test 506 (seed-0 shuffled halves of the original devtest). Every
Optuna trial (TPE, 50 trials, one shared search space) trains on dev+devtest (1503 English rows)
and scores macro-F1 on `test` (506); the best params are then retrained on all 2009 English rows
and transferred to it/de/fr — raw and shifted (`emb_L + D_L`, `D_L = mean(emb_en − emb_L)`).

### Final results — lenient labels, per-tag tuned thresholds

The headline run maps the raw comma-tags onto the 8 canonical tags with the curated aliases in
[`audit_labels.md`](audit_labels.md) §1 (**lenient** labels: singular/plural, `crime`/`law` →
`crime and law`, `science and technology` → `science`, the misspelling `sciece`, `tourism`/
`accomodation` → `travel`, `disasters and accidents` → `safety`, country/continent tags →
`geography`). This recovers the large fraction of positives that exact-string matching drops and
cuts all-zero rows from 1147/2009 (strict) to ~504/2009. It also tunes **one threshold per tag**
in-sample (grid 0.05..0.95) instead of a global 0.5; the Optuna search itself is unchanged (still
devtest macro-F1 at the fixed 0.5).

Per-language × pooling grid, tuned `test` macro-F1 (`devtest → test`):

| language | mean | bos | eos | lead |
|---|---:|---:|---:|---:|
| en | 0.416 → 0.528 | 0.305 → 0.298 | 0.348 → 0.370 | 0.272 → 0.341 |
| it | 0.344 → 0.464 | 0.226 → 0.352 | 0.314 → 0.352 | 0.171 → 0.222 |
| de | 0.392 → 0.470 | 0.283 → 0.322 | 0.304 → 0.344 | 0.202 → 0.221 |
| fr | 0.411 → 0.503 | 0.248 → 0.251 | 0.281 → 0.365 | 0.235 → 0.246 |

`mean` wins every language — tuned `test` macro-F1 **en 0.528 / it 0.464 / de 0.470 / fr 0.503**
(language mean 0.491) — and is also best on micro-F1 and subset accuracy. Per-language × pooling
grid: [`xgboost_perlang_lenient.md`](xgboost_perlang_lenient.md); pooled-mode comparison:
[`xgboost_topic_allmodes_lenient.md`](xgboost_topic_allmodes_lenient.md); per-topic breakdown:
[`xgboost_perlang_lenient_pertopic.md`](xgboost_perlang_lenient_pertopic.md).

**Transfer takeaway:** `mean` + `D` is best (tuned transfer macro-F1 `mean+D` 0.345 / 0.327 / 0.400
for it / de / fr, vs `mean` raw 0.278 / 0.298 / 0.381), and the gain concentrates on `travel`
(+0.187 accuracy fr, +0.081 it, +0.053 de); the other poolings do not transfer
([`xgboost_pertopic_transfer.md`](xgboost_pertopic_transfer.md)).

### Inference

`xgboost_infer.py` rebuilds the 8 tag models deterministically from the stored params/thresholds
(no serialized model files) and writes a `predictions_*.csv` (git-ignored).

```bash
python xgboost_infer.py --mode mean --lang it --shift --eval     # transfer models, it + D
python xgboost_infer.py --source perlang --lang en --mode mean --eval
python xgboost_infer.py --source perlang --lang de --mode mean --shift --input my_emb.npy   # custom (n, 768) float32 features
```

`--source transfer` (default) retrains the English tag models on all 2009 English rows and
re-tunes the thresholds in-sample; `--source perlang` uses the per-language models. `--shift` adds
`D_lang = mean(emb_en − emb_lang)`; `--eval` prints per-tag accuracy/F1 and subset accuracy against
the gold lenient labels (repo embeddings only).

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

(Full tables and per-mode reports: `xgboost_topic_allmodes.md`; script: `xgboost_topic_optuna.py`.)

Strict-label reading: `mean` leads again — best English test score and best raw transfer — and the
shift lifts it in every language on **subset accuracy** (it 0.595 → 0.634, de 0.589 → 0.637,
fr 0.642 → 0.647)
while macro-F1 is mixed (+0.052 it, +0.013 de, −0.070 fr). The other poolings transfer poorly and
the shift is mixed for them (only `eos` gains on average). Rare tags (geography, science, safety,
health) have very few positives, so macro-F1 is noisy and dominated by `travel`/`sports`. Optuna
selects on the same `test` set it reports (selection-set scores, per protocol).

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

**2. Per-pooling cosine on example pairs.** Averages over three unrelated and three related pairs
(the script also prints an identical and a cross-lingual control):

| pooling | unrelated | related | behavior |
|---|---|---|---|
| `mean` | +0.391 | +0.752 | orders relatedness correctly (related well above unrelated) |
| `lead` | +0.650 | +0.770 | degenerate |
| `bos`  | +0.379 | +0.310 | inverted |
| `eos`  | +0.773 | +0.769 | saturated |

Only `mean` separates related from unrelated pairs; `lead` is degenerate, `bos` inverted, `eos`
saturated. This mirrors the Table 1 result.
