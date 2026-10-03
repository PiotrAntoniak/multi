# EuroBERT-210m cross-lingual sentence embeddings & exact-translation retrieval

Sentence embeddings of **FLORES-200** (`en`, `it`, `de`, `fr`; 2009 aligned sentences) computed
with **EuroBERT-210m**, plus an exact-translation retrieval evaluation and a token/cosine
diagnostic.

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

## Run end-to-end

```bash
pip install -r requirements.txt
python extract_embeddings.py      # downloads EuroBERT-210m, writes embeddings/<mode>/emb_<lang>.npy
python table1.py                  # prints Table 1 (exact-translation retrieval ranges)
python check_tokens_cos_dist.py   # prints input-embedding norms and per-pooling cosine distances
```

`extract_embeddings.py` writes float32 `(2009, 768)` arrays for the four modes and four languages.
`table1.py` is standalone (csv + numpy only) and depends only on the produced `.npy` files and the
CSV. `check_tokens_cos_dist.py` re-loads EuroBERT-210m and inspects the tokenizer/embedding matrix.

Data: `flores200_en_it_de_fr.csv` (2009 rows: 997 `dev`, 1012 `devtest`) with columns
`split, id, en, it, de, fr, URL, domain, topic, has_image, has_hyperlink`.

## What Table 1 shows

Printed by `python table1.py`:

```
Table 1 - devtest protocol | dev=997 devtest N=1012 | 12 ordered directions | chance rank 506.5
mode    top-1 raw        +D |   MRR raw        +D |  mean rank raw        +D
mean   27.1-80.8%  78.8-93.7%  | 0.485-0.869 0.836-0.961  |   2.1- 13.1   1.2-  6.3
lead    7.3-26.5%  10.5-28.9%  | 0.118-0.333 0.144-0.354  | 181.2-308.7 179.4-316.8
bos     2.5- 8.9%   4.4-11.8%  | 0.044-0.128 0.075-0.173  | 208.4-364.6 180.5-330.0
eos     1.8- 7.5%   9.1-21.1%  | 0.043-0.139 0.154-0.289  | 155.7-290.0 105.7-194.9
```

Column by column:

- **top-1** — percentage of the 1012 devtest queries whose exact translation is the single nearest
  of the 1012 candidate keys.
- **MRR** — mean of `1 / rank` over queries.
- **mean rank** — average rank of the exact translation (random chance is `(1012 + 1) / 2 = 506.5`).
- **raw vs +D** — before vs after adding the dev-fitted shift `D` to every candidate key.
- **ranges** — min–max over the **12 ordered directions** (`en->it`, `en->de`, ..., `fr->de`).

## What the experiment shows

- **Mean pooling dominates.** After adding `D`, mean pooling reaches **78.8–93.7% top-1** and MRR
  **0.836–0.961** across all 12 directions, and its mean rank drops to 1.2–6.3. It is the only
  pooling that produces usable exact-translation retrieval.
- **Lead / bos / eos fail.**
  - `bos` position 0 is dominated by the leading special id, whose input embedding is essentially
    untrained: id `128000` has row norm `0.00280` and is the **313th-smallest of all 128,256
    embedding rows**, so the bos vector carries almost no sentence signal.
  - `eos` distances saturate — rows are extremely close to one another regardless of content.
  - `lead` degenerates: the position-0 hidden state is nearly constant across inputs, so raw
    retrieval ranks are ~181–309.
- **`D` works and is not trained.** It is a single closed-form vector (`mean(emb_a - emb_b)` over
  997 dev rows), yet adding it improves every mode's metrics, most dramatically for `eos`
  (top-1 1.8–7.5% -> 9.1–21.1%, mean rank 155.7–290.0 -> 105.7–194.9).

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
| `lead` | +0.650 | +0.770 | degenerate — unrelated pairs also score ~+0.99 (`+0.9982`, `+0.9924`), i.e. near-constant vectors |
| `bos`  | +0.379 | +0.310 | **inverted** — related pairs score below unrelated |
| `eos`  | +0.773 | +0.769 | saturated — related and unrelated are indistinguishable |

Only `mean` separates related from unrelated pairs; `bos` is inverted, `lead` is degenerate, and
`eos` is saturated. This mirrors the Table 1 result.
