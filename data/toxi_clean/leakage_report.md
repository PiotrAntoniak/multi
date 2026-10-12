# toxi-text-3M -- leakage-corrected copy

Source: `FredZhang7/toxi-text-3M` (read-only). train `train/multilingual-train-deduplicated.csv` = 2,880,667 rows; test `test/multilingual-test.csv` = 63,812 rows.

**Method.** For every language: (1) finalise the test set (cap `--test-cap`); (2) remove every test text from that language's train pool; (3) drop duplicate train texts; (4) cap train to `--cap` (seed `--seed`); (5) verify 0 remaining overlap. Test texts are treated as the leak source, so the post-condition is exactly 0 by construction and asserted.

> EN is **also** leaked (90.1% of its 525 test texts sit in the EN train pool), so the same removal is applied to EN by default. Pass `--en-subsample-only` to instead reproduce the literal 'subsample EN train only' behaviour (leaves EN leakage intact).

## Overlap per language (test texts found in the train pool)

| lang | test rows | train pool (rows / unique) | leaked test texts BEFORE | % BEFORE | removed from train | dup train dropped | train rows AFTER | leaked AFTER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 525 | 2528002 / 2527317 | 473 | 90.1% | 473 | 685 | 20000 | 0 |
| it | 5000 | 20611 / 20589 | 4997 | 99.9% | 4997 | 22 | 15592 | 0 |
| de | 190 | 12255 / 12234 | 161 | 84.7% | 161 | 21 | 12073 | 0 |
| fr | 5000 | 26175 / 26084 | 4997 | 99.9% | 4997 | 91 | 20000 | 0 |
| es | 5000 | 26726 / 26715 | 4992 | 99.8% | 4992 | 11 | 20000 | 0 |

**After correction every language has 0 leaked test texts** (asserted at build time).

## Caps and size

- train cap `--cap` = 20000 per language (0 = no cap); test cap `--test-cap` = 5000 per language (0 = no cap); seed = 0.  
A test language with fewer rows than the cap keeps all of them (e.g. de test, en test).

- total `data/toxi_clean/` size on disk = **13.47 MiB** (14.12 MB decimal; 14119352 bytes total) across 10 parquet files (gzip).

## Files

- `data/toxi_clean/en_train.parquet` (4243.8 KiB)
- `data/toxi_clean/en_test.parquet` (87.2 KiB)
- `data/toxi_clean/it_train.parquet` (1532.0 KiB)
- `data/toxi_clean/it_test.parquet` (712.9 KiB)
- `data/toxi_clean/de_train.parquet` (1238.1 KiB)
- `data/toxi_clean/de_test.parquet` (30.0 KiB)
- `data/toxi_clean/fr_train.parquet` (2631.9 KiB)
- `data/toxi_clean/fr_test.parquet` (773.5 KiB)
- `data/toxi_clean/es_train.parquet` (1867.8 KiB)
- `data/toxi_clean/es_test.parquet` (671.2 KiB)

## Full-dataset reproduction

The size-managed copy above is a cap. To rebuild the FULL cleaned dataset (leakage removal + train dedup, no caps):

```
python clean_toxi.py --cap 0 --test-cap 0 --langs en,it,de,fr,es --out data/toxi_clean_full
```

## Exact CLI used for this copy

```
python clean_toxi.py --cap 20000 --test-cap 5000 --langs en,it,de,fr,es --out data/toxi_clean --seed 0
```
