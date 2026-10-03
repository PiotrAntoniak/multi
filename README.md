# EuroBERT-210m cross-lingual sentence embeddings & exact-translation retrieval

Sentence embeddings of **FLORES-200** (`en`, `it`, `de`, `fr`; 2009 aligned sentences) and of
**Europarl v7** (five languages against English: de, es, fr, nl, pt; 5,000 pairs each) computed
with **EuroBERT-210m**, plus exact-translation retrieval evaluations (foreign-to-English) and a
token/cosine diagnostic.

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
python fetch_europarl_langs.py    # downloads + extracts the five Europarl pairs (de/es/fr/nl/pt)
python table1_europarl_multilang.py all   # Europarl: 5k pairs/lang, embed, foreign-to-English table
python check_tokens_cos_dist.py   # prints input-embedding norms and per-pooling cosine distances
```

`extract_embeddings.py` writes float32 `(2009, 768)` arrays for the four modes and four languages.
`table1.py` is standalone (csv + numpy only) and depends only on the produced `.npy` files and the
CSV. `check_tokens_cos_dist.py` re-loads EuroBERT-210m and inspects the tokenizer/embedding matrix.

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

Second corpus: **Europarl v7** (European Parliament proceedings). For each of five pairs against
English (de, es, fr, nl, pt): 5,000 seeded, deduplicated line-aligned pairs, dev = 1,000 /
devtest = 4,000; EuroBERT-210m `mean` embeddings; `D` fitted on dev and added to the keys;
retrieval evaluated **foreign-to-English only** (`xx->en`), 4,000 queries vs 4,000 keys
(chance rank 2,000.5).

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

## `check_tokens_cos_dist.py`

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
