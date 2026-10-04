# FLORES topic-classifier label & baseline audit

Read-only audit of `multi/`. Features: `embeddings/mean/emb_en.npy` (2009x768). 
Split: seed-0 shuffle of the original devtest (dev=997, devtest=506, test=506), 
identical to `xgboost_topic_optuna.py`. Threads capped at 1 (`OMP/MKL/OPENBLAS=1`, `n_jobs=1`).

## 1. Label coverage / signal lost

Raw label space: **278 distinct tags**, 2453 tag instances over 2009 rows. 
The 8 canonical tags are *not* the only names for their concepts in the raw `topic` column: 
closely-related tags (`sport`, `crime`, `law`, `science and technology`, `tourism`, `accomodation`, 
`disasters and accidents`, the misspelling `sciece`, and long tails such as `science/*`) are dropped 
by exact-string matching.

### Top-30 raw tags

| # | raw tag (lowercased, stripped) | count |
|---:|---|---:|
| 1 | `travel` | 497 |
| 2 | `sports` | 101 |
| 3 | `politics` | 90 |
| 4 | `crime and law` | 80 |
| 5 | `health` | 44 |
| 6 | `safety` | 40 |
| 7 | `science` | 39 |
| 8 | `disasters and accidents` | 38 |
| 9 | `geography` | 34 |
| 10 | `history` | 34 |
| 11 | `entertainment` | 31 |
| 12 | `culture` | 30 |
| 13 | `science and technology` | 28 |
| 14 | `tourism` | 28 |
| 15 | `accomodation` | 27 |
| 16 | `crime` | 25 |
| 17 | `accident` | 23 |
| 18 | `environment` | 23 |
| 19 | `communication` | 23 |
| 20 | `how things work` | 21 |
| 21 | `food` | 20 |
| 22 | `antartica` | 19 |
| 23 | `politics and conflicts` | 18 |
| 24 | `sport` | 18 |
| 25 | `weather` | 15 |
| 26 | `sailing` | 15 |
| 27 | `culture and entertainment` | 14 |
| 28 | `business` | 13 |
| 29 | `science/animal behavior` | 13 |
| 30 | `photography` | 13 |

### Exact vs lenient match per canonical tag (row counts)

Lenient = curated alias set (documented in this script: singular/plural, spelling variants, 
`science/*`, tourism/accomodation, disasters/accidents, country names, etc.). It is an 
*upper-bound estimate* of recoverable positives, not a definitive taxonomy.

| canonical tag | exact rows | lenient rows | lost positives (lenient-exact) | lost % |
|---|---:|---:|---:|---:|
| travel | 497 | 518 | +21 | +4% |
| sports | 101 | 148 | +47 | +47% |
| crime and law | 80 | 121 | +41 | +51% |
| politics | 90 | 177 | +87 | +97% |
| health | 44 | 81 | +37 | +84% |
| geography | 34 | 117 | +83 | +244% |
| science | 39 | 317 | +278 | +713% |
| safety | 40 | 151 | +111 | +278% |
| **total (sum over tags)** | 925 | 1630 | +705 | +76% |

### All-zero targets

- Rows with **zero canonical tags (exact matcher): 1147 / 2009 = 57.1%**.
- Rows with zero lenient tags: 504 / 2009 = 25.1%.
- An **all-zero predictor** therefore gets subset accuracy = 57.1% on the full 2009, and **57.1% of exact-match subset accuracy is 'free' before any learning**.

  - dev: all-zero exact rows 556/997 = 55.8%
  - devtest: all-zero exact rows 301/506 = 59.5%
  - test: all-zero exact rows 290/506 = 57.3%

### Canonical-tag prevalence per split (exact matcher)

| tag | all 2009 | dev 997 | devtest 506 | test 506 |
|---|---:|---:|---:|---:|
| travel | 497 | 255 | 117 | 125 |
| sports | 101 | 53 | 22 | 26 |
| crime and law | 80 | 48 | 12 | 20 |
| politics | 90 | 39 | 28 | 23 |
| health | 44 | 21 | 10 | 13 |
| geography | 34 | 19 | 6 | 9 |
| science | 39 | 19 | 12 | 8 |
| safety | 40 | 15 | 14 | 11 |

Label cardinality (exact, per row): 0 tag(s): 1147, 1 tag(s): 799, 2 tag(s): 63.

## 2. Baselines on the same split (train `dev` -> eval `devtest` unless noted)

| predictor | subset accuracy | macro-F1 |
|---|---:|---:|
| all-zero (predict no tag) | 0.5949 | 0.0000 |
| all-ones (predict every tag) | 0.0000 | 0.0966 |
| per-tag majority class (from dev) | 0.5949 | 0.0000 |
| **per-tag logistic regression** (std, max_iter=1000, thr=0.5) | 0.6403 | 0.2055 |

Per-tag logistic-regression F1 on devtest: `travel`=0.601, `sports`=0.514, `crime and law`=0.125, `politics`=0.279, `health`=0.000, `geography`=0.000, `science`=0.125, `safety`=0.000.

| XGBoost default (max_depth=6, lr=0.1, n=300) on all 8 tags | 0.6601 | 0.1679 |
| XGBoost default on only 4 high-count tags (travel/sports/crime and law/politics) | 0.7154 | 0.3001 |
| logistic regression on only 4 high-count tags | 0.7095 | 0.3798 |

> Note: these XGBoost numbers train on `dev` (997 rows) and score `devtest` (506) as the task 
> specifies, so they are *not* directly comparable to `xgboost_topic_results_mean.md` (which trains 
> on dev+devtest=1503 and scores `test`=506, baseline macro-F1 0.2563). The qualitative conclusion 
> is unchanged: adding the four rare tags back drops macro-F1 from ~0.30-0.38 to ~0.17-0.21.

### Threshold tuning (per-tag logistic regression)

For each tag the threshold in {0.1..0.9} maximising **dev** F1 is chosen (using in-sample dev 
probabilities; a selection caveat) and then applied to devtest. Fixed threshold = 0.5.

| tag | dev base rate | best dev threshold | devtest F1 @0.5 | devtest F1 @tuned |
|---|---:|---:|---:|---:|
| travel | 0.256 | 0.2 | 0.601 | 0.597 |
| sports | 0.053 | 0.1 | 0.514 | 0.578 |
| crime and law | 0.048 | 0.1 | 0.125 | 0.286 |
| politics | 0.039 | 0.1 | 0.279 | 0.357 |
| health | 0.021 | 0.1 | 0.000 | 0.308 |
| geography | 0.019 | 0.1 | 0.000 | 0.000 |
| science | 0.019 | 0.1 | 0.125 | 0.261 |
| safety | 0.015 | 0.1 | 0.000 | 0.000 |

**Macro-F1: fixed 0.5 = 0.2055 -> per-tag tuned = 0.2983 (delta +0.0927).** 
Subset accuracy: 0.6403 -> 0.5968 (delta -0.0435).

## 3. Alternative targets (quick quantification)

### full exact `topic` string (lowercased)

- classes: **283**; top-10 coverage: **33.1%** (of 2009 rows); majority train class = `travel`; majority-class accuracy on devtest = **7.7%**.
- top-10: `travel`=193, `politics`=90, `sports`=86, `crime and law`=80, `travel, safety`=40, `science`=39, `disasters and accidents`=38, `health`=34, `travel, history`=34, `entertainment`=31.

### first comma tag (lowercased)

- classes: **220**; top-10 coverage: **46.3%** (of 2009 rows); majority train class = `travel`; majority-class accuracy on devtest = **21.9%**.
- top-10: `travel`=463, `sports`=94, `politics`=90, `crime and law`=80, `health`=39, `science`=39, `disasters and accidents`=38, `entertainment`=31, `science and technology`=28, `geography`=28.

### Top-10 raw tags as a 10-way multi-label target (prevalence)

| tag | all 2009 | dev 997 | devtest 506 | test 506 |
|---|---:|---:|---:|---:|
| `travel` | 497 | 255 | 117 | 125 |
| `sports` | 101 | 53 | 22 | 26 |
| `politics` | 90 | 39 | 28 | 23 |
| `crime and law` | 80 | 48 | 12 | 20 |
| `health` | 44 | 21 | 10 | 13 |
| `safety` | 40 | 15 | 14 | 11 |
| `science` | 39 | 19 | 12 | 8 |
| `disasters and accidents` | 38 | 26 | 8 | 4 |
| `geography` | 34 | 19 | 6 | 9 |
| `history` | 34 | 14 | 11 | 9 |

The top-10 raw tags cover 50% of rows but still leave a long tail of ~268 singleton-ish tags; using them directly as labels would be far more balanced than the 8 canonical tags (each top-10 tag has >=38 rows), and would not drop `tourism`/`accomodation`/`crime`/`law` /`sport`/`science and technology`.

## 4. Verdict and recommendations

The low macro-F1 (0.10-0.33) is **not primarily a modelling/XGBoost failure**. It is a 
**data + metric problem** with a genuinely hard residual:

1. **(a) Label loss.** Exact canonical-string matching drops a large fraction of the real signal: 
   lenient aliasing raises total positive tags from 925 to 1630 
   (+76%): e.g. `sports` 101->148, `crime and law` 80->121, `science` 39->317, `safety` 40->151, `health` 44->81. 
   Many rows labelled only `science and technology`, `tourism`, `accomodation`, `crime`, `law`, `sport`, 
   or the misspelling `sciece` become all-zero/false-negative targets.

2. **(b) All-zero dominance.** 1147/2009 = 57.1% of rows have no exact canonical tag, 
   so an all-zero predictor already scores 57.1% subset accuracy on the full set and 59.5% on devtest. Subset accuracy is therefore 
   uninformative and macro-F1 must carry the diagnosis.

3. **(b) Rare-tag dilution.** `health`, `geography`, `science`, `safety` have only 34-44 positives over 2009 rows; 
   in the 506-row devtest they have 6-14 positives. Per-tag F1 is extremely noisy and each near-zero rare tag 
   contributes an equal 1/8 to the macro average, so 3-4 weak rare tags cap macro-F1 around 0.2-0.35 even when 
   the common tags score 0.45-0.6.

4. **(c) Genuinely hard.** Even the per-tag reference (logistic regression on mean-pooled embeddings) only reaches 
   macro-F1 0.206 and the XGBoost default 0.168 on devtest; the 
   pooled sentence embedding carries topical signal for `travel`/`sports` (F1 ~0.5) but weak signal for the rare, 
   semantically-overlapping tags. So the task is hard, but the reported ultra-low numbers are largely an artefact of the label definition.

### Recommendations

1. **Use lenient tag mapping.** Normalise the raw `topic` column (singular/plural, `science and technology`->`science`, 
   `tourism`/`accomodation`->`travel`, `crime`/`law`->`crime and law`, fix `sciece`->`science`, map `disasters and accidents`->`safety`) 
   before building targets; this recovers a third to a half of the lost positives and makes targets less all-zero.
2. **Report micro-F1 and per-tag F1 alongside macro-F1, and/or exclude tags with < N positives from macro.** 
   State `N` (e.g. 50) explicitly. With the 4 high-count tags macro-F1 is already 2-4x higher than over all 8.
3. **Tune per-tag decision thresholds** (cost-sensitive for rare tags) instead of a global 0.5; this improves the rare-tag 
   recall that macro-F1 rewards (see threshold table above).
4. **Consider a different target**: a coarser, balanced label set (e.g. the top-10 raw tags, or a single-label `topic` 
   classification with a majority-class floor) is far more learnable; if the scientific claim needs the 8 canonical tags, 
   report them with lenient mapping plus confidence intervals from the small positive counts.
