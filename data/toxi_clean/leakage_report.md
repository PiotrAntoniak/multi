# toxi-text-3M -- leakage-corrected 80/10/10 copy

Source: `FredZhang7/toxi-text-3M` (read-only). Shipped splits pooled: `train, validation, test`.

**Method.** For each language: (1) pool ALL shipped splits (train + validation + test); (2) deduplicate GLOBALLY per language on `text`; (3) optionally cap the TOTAL per language (`--total-cap`, before splitting); (4) re-split 80/10/10 with seed `--seed` (ceil on the 10% parts so val and test are each >= 10%). A text is unique per language, so no text can appear in two splits -> 0 cross-split leakage by construction (asserted).

## Per-language counts

| lang | pooled N | dedup drop | pool N (post-cap) | train | val | test | train/val/test frac | pos-rate tr/val/te | cross-split overlap |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|---:|
| en | 2535319 | 1224 | 30000 (capped) | 24000 | 3000 | 3000 | 0.800 / 0.100 / 0.100 | 0.117 / 0.112 / 0.121 | 0 |
| it | 31471 | 8379 | 23092 | 18472 | 2310 | 2310 | 0.800 / 0.100 / 0.100 | 0.283 / 0.275 / 0.294 | 0 |
| de | 12478 | 182 | 12296 | 9836 | 1230 | 1230 | 0.800 / 0.100 / 0.100 | 0.299 / 0.302 / 0.310 | 0 |
| fr | 37073 | 10808 | 26265 | 21011 | 2627 | 2627 | 0.800 / 0.100 / 0.100 | 0.365 / 0.354 / 0.369 | 0 |
| es | 37164 | 7977 | 29187 | 23349 | 2919 | 2919 | 0.800 / 0.100 / 0.100 | 0.313 / 0.318 / 0.313 | 0 |

**Cross-split text overlap is 0 for every language** (asserted at build time; val and test are each >= 10% of the pooled per-language data).

## Caps and size

- `--total-cap` = 30000 per language total (0 = no cap); seed = 0.  A language whose deduped pool is below the cap keeps all of it.

- total size on disk = **16.54 MiB** (17.34 MB decimal; 17344019 bytes) across 15 parquet files (gzip) + this report.

## Files

- `data/toxi_clean/en_train.parquet` (5068.6 KiB)
- `data/toxi_clean/en_validation.parquet` (636.5 KiB)
- `data/toxi_clean/en_test.parquet` (659.2 KiB)
- `data/toxi_clean/it_train.parquet` (2069.0 KiB)
- `data/toxi_clean/it_validation.parquet` (272.3 KiB)
- `data/toxi_clean/it_test.parquet` (276.5 KiB)
- `data/toxi_clean/de_train.parquet` (1030.0 KiB)
- `data/toxi_clean/de_validation.parquet` (136.7 KiB)
- `data/toxi_clean/de_test.parquet` (118.9 KiB)
- `data/toxi_clean/fr_train.parquet` (2880.9 KiB)
- `data/toxi_clean/fr_validation.parquet` (342.1 KiB)
- `data/toxi_clean/fr_test.parquet` (367.0 KiB)
- `data/toxi_clean/es_train.parquet` (2425.6 KiB)
- `data/toxi_clean/es_validation.parquet` (329.5 KiB)
- `data/toxi_clean/es_test.parquet` (324.6 KiB)

## Exact CLI used

```
python clean_toxi.py --langs en,it,de,fr,es --out data/toxi_clean --seed 0 --total-cap 30000
```
