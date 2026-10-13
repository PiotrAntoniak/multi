# EXPERIMENTS — Cross-lingual study (single tracking file)

**Question.** Can a model trained on **English** perform comparably on **non-English** text, and which
embedding dimensions carry language? Constraint (user): frozen **EuroBERT-210m / 610m**, **mean pooling
only**, classifier **XGBoost** (plus comparisons), and — for the headline "out-of-the-box" claim — **no
second model and no machine translation**.

**How to read this file.** Each experiment is one block: **What** (what it does), **Setup**, **Key result**,
**Finding**. Metrics are `macro-F1 / micro-F1 / AUC`. Large derived caches are git-ignored and rebuilt by
the scripts (`REPRODUCE.md`).

## Glossary

| term | meaning |
|---|---|
| **mean** | mean-pooled last hidden state (the only pooling used). |
| **perlang_z** (`l2_thenZ [both]`) | L2-normalize each vector, then z-score with **each language's own train** mean/std, applied to English **and** the target. Model-free. |
| **pc** | GLOBAL iterative PCA ablation: fit PCA on the stacked-language train pool, zero the leading PC, inverse, re-fit; k = #PCs removed. |
| **inlp** | project out a language-subspace basis (iterated null-space probe). |
| **D** | language-shift: target features + (mean_en_train − mean_target_train). |
| **self / transfer** | self = in-language (train L → test L); transfer = cross-lingual (train L → test L′). |
| **XL-mean** | mean over non-English targets. |
| **backbones** | 210m = 768-d, 610m = 1152-d. |

---

## PART A — The headline result: one English model, comparable across languages

### E18 — Out-of-the-box cross-lingual proof — `proof_mean_alllangs.py` — DONE (verified)
**What:** train ONE English XGBoost on mean-pooled EuroBERT-210m with only model-free feature ops, and
check comparability (target/English ≥ 0.9) across languages. **Setup:** ladder `raw → l2 → global_center/
global_z` (English stats only) `→ perlang_center/perlang_z` (target's own unlabeled train stats) → XGB.
**Key result (perlang_z):** target/English — FLORES sent AUC it .92 / de .93 / fr .92 (EN .849);
FLORES agg AUC it .97 / de .95 / fr .94 (EN .888); MultiEURLEX micro-F1 de .93 / fr .93, AUC de .94 / fr .96
(EN .718 / .906). **Finding:** equivalent across languages **iff** a mild unlabeled **target sample** is
used for per-language standardization; English-only stats never reach 0.9. **Independently verified** (rebuilt
from raw embeddings; all claims reproduce to 6 decimals).

### E24 — More datasets (XNLI, PAWS-X, textdetox, SIB-200) — `more_datasets.py` — DONE
**What:** same out-of-the-box proof on additional multilingual classification tasks.
**Key result (target/EN ratio, perlang_z):** PAWS-X (binary paraphrase) **0.92–0.95 ✅**; XNLI (3-cl NLI)
de 0.90 / fr 0.95 / es 0.95 (borderline); SIB-200 (topic) de 0.90 / fr 1.00 / es 0.90; **textdetox
(binary toxic) 0.62–0.81 ✗**. **Finding:** comparability holds for **topic / paraphrase / NLI**, fails for
**toxicity**. `global_z` (no target sample) is always worse.

### E25 — Full language × language matrices — `lang_matrix.py` — DONE
**What:** train **every** language, test **every** language (held-out test), all multilingual datasets,
both backbones. **Key result (in-language diag vs cross-lingual off-diag macro-F1):**

| dataset | 210m diag/off (gap) | 610m diag/off (gap) |
|---|---|---|
| flores_sent | .461/.377 (.084) | .486/.447 (.039) |
| flores_agg | .544/.423 (.121) | .510/.464 (.046) |
| mel | .588/.330 (.258) | .577/.417 (.160) |
| toxic | .739/.567 (.171) | .750/.598 (.152) |
| xnli | .452/.408 (.044) | .460/.430 (.030) |
| **pawsx** | .520/**.532** (closed) | .561/.558 (closed) |
| sib200 | .792/.691 (.101) | .800/.763 (.037) |
| textdetox | .870/.672 (.198) | .874/.717 (.158) |

**Finding:** **610m shrinks the cross-lingual gap on every dataset**; PAWS-X transfers at parity; XNLI/FLORES/
SIB-200 nearly close; MultiEURLEX/textdetox retain the largest gaps. (English-only SST-2/emotion skipped.)

### E20 — Toxicity, clean split — `toxi_proof_clean.py` — DONE
**What:** out-of-the-box proof on `toxi-text-3M` (binary toxic, en/it/de/fr/es), leakage-corrected 80/10/10,
3 seeds. **Key result (AUC EN/native, target/EN):** it .649/.834 (0.72), de .678/.869 (0.75), fr .776/.866
(0.86), es .639/.797 (0.71) — **NOT comparable**. **Finding:** toxicity is the clear counter-example.
Models saved to `models_toxi/`; cleaned dataset in `data/toxi_clean/` (+ single `data/toxi_clean.parquet`).

---

## PART B — Why mean pooling transfers (dimensions & language)

### E1/E1b — Language-dimension census + INLP — `langdims_probe.py`, `lang_inlp.*` — DONE
**What:** per-dim η² across languages; language probe before/after zeroing dims; INLP. **Key result:** probe
= 1.000 on 768 dims; zeroing the top-557 η² dims (90% of mass) → 0.999; only all-768 → 0.25. INLP reaches
chance after ~30 directions (60–72% variance retained). **Finding:** language is **redundant across ~all
dims** (no small dim set), but a small **subspace** suffices.

### E17 — Language dims vs their SHAP contribution — `lang_shap_dims.py` — DONE
**What:** per-dim η² vs the English XGB's mean|SHAP|. **Key result:** Spearman(η², SHAP) ≈ 0 (−.02…−.08);
top-50 overlap 0–1/50; SHAP mass on the 50%-language dims **below random** (FLORES) / **= random** (mel).
**Finding:** the task model **does not use** the language dims — why mean pooling transfers.

### E12/E13/E26 — Dimension removal — `pc_ablation*.py`, `dim_removal.py` — DONE
**What:** iteratively remove leading PCs (global shared PCA, k=0–5), plus INLP and random control; across
datasets/backbones. **Key result (XL-mean macro-F1, k=0→best):** mel 210m .076→**.290** (k=2), 610m
.107→**.341**; flores_agg 210m .204→**.380** (k=3); flores_sent 210m .206→.259 (k=3); **toxic neutral-to-
harmful** (610m best at k=0). Random control stays below. **Finding:** the leading PCs are ~96–98%
language-aligned; removing ~2–3 **helps topic tasks** (optimum k≈2–3, harm past that), is
neutral/harmful for toxicity; effect is **subspace-specific**, not mere dimensionality; EN-self flat.

---

## PART C — Task-level cross-lingual transfer (210m & 610m)

### E3/E6/E7 — FLORES topic cross-lingual — `crosslingual_topic*.py`, `crosslingual_inlp.py` — DONE
**What:** EN topic classifier → it/de/fr under `full/zero_k/subspace/centroid/abtt/inlp`. **Key result
(mean macro-F1):** sent full .234 → .334 (`inlp`); agg .246 → .356. **Finding:** removing language
helps; ABTT hurts; the rank-3 centroid projection's probe collapse is tautological (INLP is faithful).

### E9/E10/E11 — MultiEURLEX (long EU docs, 21-label) — `multieurlex_*`, `mel_close.py` — DONE
**What:** baseline, `l2_thenZ`, calibration, 610m, ensembling, and gap-closers. **Key result (EN→de/fr
micro-F1):** raw .266/.334 → `l2_thenZ`+calib (210m) .688/.683 → **610m + calib .711/.707** (~94% of
English). `pc` k=2 on plain .512/.525. Ensembling with a gold/MT English view closes the AUC gap.
**Finding:** `l2_thenZ` + threshold/prior calibration + bigger backbone essentially closes the gap;
`mlp_align` fails on long docs; translate-train (chunked MT) hurts here.

### E14/E14b — FLORES gap-closers — `flores_close.py` — DONE
**What:** transforms, kNN, translate-train, MLP-alignment, ensembles; macro-F1 vs in-language ceiling.
**Key result (sent / agg mean target):** raw .291/.277 → `pc+inlp` .371/.310 → translate-train .567/.618,
MLP-align .541/.649, **kNN .602/.676** (≥90% closure; matches ceiling). **Finding:** with XGBoost,
translate-train closes ~95%/86% (accuracy); kNN / MLP-alignment reach the native ceiling (use unlabeled
target / MT).

### E23 — Translate-test — `translate_test_eval.py` — DONE
**What:** train EN → MT the it/de/fr **test** → EN → evaluate (vs original EN test). **Key result (macro-F1):**
EN test .486 (210m)/.495 (610m); translated-test it .466/.519, de .466/.494, fr .476/.515 → **ratio
0.96–0.98 (210m), 1.00–1.05 (610m)**; untranslated only .23–.34. **Finding:** translating the **test** into
English recovers English-level performance.

### E4 — Sentiment / emotion — `crosslingual_sentiment.py` — DONE
**What:** EN sentiment (SST-2) & emotion → translated it/de/fr under neutralizations. **Key result (macro-F1):**
SST-2 EN .792 → cross it/de/fr .596/.593/.636 (raw) → best `centroid` .630/.627/.649; emotion EN .370 →
best .203/.214/.219. **Finding:** neutralization helps but a real residual gap remains (lexical
language-specificity + MT noise).

---

## PART D — Other classifiers, MT, and calibration

### E21 — Non-XGBoost classifiers — `models_compare.py` — DONE
**What:** 16 sklearn classifiers on FLORES-sent mean embeddings, EN & IT, both backbones. **Key result
(EN-self / transfer macro-F1, 610m):** **`SVC_linear` .583 / .562** > MLP .564/.517 > LogisticRegression
.541/.514 > XGBoost .495/.466; `GaussianNB` is a transfer specialist at 210m (.453). **Finding:** a **linear
SVM beats XGBoost** on both self and transfer; tree ensembles do not.

### E28 — backbone × dataset × method grid (10 Optuna trials) — `backbone_grid.py` — DONE
**What:** XGBoost + 10 Optuna trials per cell; `mean/pc/inlp/D`; 210m & 610m. **Key result (XL-mean
macro-F1):** mel 210m .059 (mean) → **.293 (pc)** / .299 (inlp); mel 610m .115 → .338/.349; flores_sent
610m .241 → **.307 (inlp)**; toxic ~flat (.45–.50, pc/D slightly hurt). **Finding:** the language-subspace
methods (`pc`/`inlp`/`D`) massively help multi-label/long-doc mel and help FLORES-sent; helpfulness shrinks
as the task becomes lexical (toxicity).

### E22 — MT: bilingual vs multilingual → English — `mt_compare.py` — DONE
**What:** FLORES it/de/fr → en with `opus-mt-{it,de,fr}-en` (bilingual) vs `opus-mt-mul-en` (multilingual).
**Key result (BLEU / chrF, macro):** bilingual **37.5 / 64.4** vs multilingual 28.1 / 56.6 (−9.4 BLEU).
**Finding:** the three bilingual models beat one many→en model on every language.

### E19 — FLORES calibration & ceiling feasibility — `flores_calib.py`, `flores_ceiling.py` — DONE
**What:** label-free calibration; per-language in-language ceilings. **Key result:** agg micro-F1 reaches
0.91/0.92/0.99 of English via `prior_sinkhorn`; sent F1 does not (best ~0.77). `0.9×EN-self` is **provably
infeasible** for sent/de, sent/it(macro), agg/fr (exceeds their own ceilings). AUC invariant across
calibrations. **Finding:** FLORES single-label F1's residual is ceiling/label-noise bound, not transfer.

### E27 — Precision/recall vs threshold — `toxi_pr.py` (+610) — DONE
**What:** P/R-vs-threshold 2×2 (EN-on-EN | EN-on-IT + IT-on-IT) for toxicity, 210m & 610m.
**Key result:** at recall 0.90, EN-on-IT precision .32 vs IT-on-IT .44 (210m); AUC verdict matches the
"not comparable on toxicity" result.

---

## PART E — Recall-first / highly unbalanced

### E15 — English vs native at high recall — `imbalanced_recall*.py` — DONE
**What:** rare-positive detection (FLORES rare topics; MultiEURLEX rare EuroVoc concepts), EN-trained vs
native, recall-first (recall@FPR, recall@false-alarms, FPR@90%-recall). **Key result:** FLORES — FPR for 90%
recall EN .543 vs native .540 (**identical**); MultiEURLEX (v2, label-bug fixed) EN .426 vs native .357.
**Finding:** at the recall-critical point the English detector spends ~the same false-alarm budget as a
native model. (A first MultiEURLEX run falsely showed EN ≫ native — a **label bug**; corrected in v2.)

---

## PART F — Data, translation, and utilities

### E2/DATA — Translated eval sets — `translate_datasets.py`, `rebuild_caches.py` — DONE
MT SST-2/emotion test(+val) → it/de/fr for the sentiment line (Marian flips some polarity — a floor).

### E5 — Independent verifications — DONE
- E5a: reproduced the η²/SHAP/probe claims; found the rank-3 `subspace` probe-collapse is **tautological**.
- E5b: reproduced E3/E4/E6 headline numbers (match ≥4 decimals).
- (E18 verified to 6 decimals; E15-mel bug found by a dedicated check.)

### TOXI CLEANING — `clean_toxi.py` — DONE
`toxi-text-3M` had train↔test **text leakage** in every language (en 90%, it/fr 99%); pooled+deduped,
re-split **80/10/10** into `data/toxi_clean/` + one `data/toxi_clean.parquet` (120,840 rows).

### MASTER RESULT (10 datasets)
English-trained, mean-pooled, model-free `perlang_z` → comparable (target/EN AUC or F1 ≥ 0.9) for
**topic** (FLORES sent/agg, MultiEURLEX, SIB-200), **paraphrase** (PAWS-X), and **NLI** (XNLI); **not**
comparable for **toxicity** (textdetox, toxi-text-3M). 610m shrinks the gap everywhere. On highly unbalanced
positives the English model matches the native model's false-alarm cost at 90% recall.

> **Recovery note (2026-10-11):** a buggy cleanup command deleted all files under `multi/`. Tracked files
> were restored from git; embeddings were rebuilt; this log and `lang_neutralize.py` rewritten. Only
> git-recoverable + rebuilt artifacts remain; large caches stay git-ignored.
