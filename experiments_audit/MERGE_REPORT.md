# MERGE REPORT — `EXPERIMENTS.draft.md`

**Generated:** 2026-10-20
**Output draft:** `C:\Users\pedro\Documents\random_projects\multi\EXPERIMENTS.draft.md`
**Source directory:** `C:\Users\pedro\AppData\Local\Temp\opencode\multi_experiments_parts\`
**`EXPERIMENTS.md` was not modified.** `git status` in `multi/` shows `EXPERIMENTS.draft.md` as the only new
file from this merge; `git diff -- EXPERIMENTS.md` is empty.

---

## 1. Inputs used

**34 entry files** were read and merged. `COVERAGE.md` was used only to build the index and the
missing-script list; it contributes no body text.

| # | input file | IDs covered | Script line present | Status recorded |
|---:|---|---|---|:--|
| 1 | `E1_E1b.md` | E1, E1b | yes | NOT AVAILABLE (UNK) |
| 2 | `E2.md` | E2 | yes | PARTIAL |
| 3 | `E3_E6_E7.md` | E3, E6, E7 | yes | NOT AVAILABLE (UNK) |
| 4 | `E4.md` | E4 | yes | NOT AVAILABLE (UNK) |
| 5 | `E9_E10_E11.md` | E9, E10, E11 | yes | PARTIAL |
| 6 | `E12_E13_E26.md` | E12, E13, E26 | yes | PARTIAL (E13/E26 NOT AVAILABLE) |
| 7 | `E14_E14b.md` | E14, E14b | yes | REPRODUCIBLE |
| 8 | `E15.md` | E15 | yes | REPRODUCIBLE |
| 9 | `E17.md` | E17 | yes | REPRODUCIBLE |
| 10 | `E18.md` | E18 | yes | REPRODUCIBLE |
| 11 | `E19.md` | E19 | yes | PARTIAL |
| 12 | `E20.md` | E20 | yes | REPRODUCIBLE |
| 13 | `E21.md` | E21 | yes | PARTIAL |
| 14 | `E22.md` | E22 | yes | PARTIAL |
| 15 | `E23.md` | E23 | yes | REPRODUCIBLE |
| 16 | `E24.md` | E24 | yes | PARTIAL |
| 17 | `E25.md` | E25 | yes | REPRODUCIBLE |
| 18 | `E25b.md` | E25b | yes | REPRODUCIBLE |
| 19 | `E27.md` | E27, E37 (split out) | yes | PARTIAL |
| 20 | `E28.md` | E28 | yes | PARTIAL |
| 21 | `E29.md` | E29 | yes | PARTIAL |
| 22 | `E30.md` | E30 | yes | REPRODUCIBLE (FLORES) |
| 23 | `E31.md` | E31 | yes | PARTIAL |
| 24 | `E32.md` | E32 | yes | REPRODUCIBLE (FLORES) |
| 25 | `E33.md` | E33 | yes | PARTIAL / full rerun NOT AVAILABLE |
| 26 | `E34.md` | E34 | yes | PARTIAL |
| 27 | `E35.md` | E35 | yes | REPRODUCIBLE |
| 28 | `E36.md` | E36 | yes | PARTIAL (provenance only) |
| 29 | `E37.md` | E37 | yes | PARTIAL |
| 30 | `E38.md` | E38 | yes | PARTIAL |
| 31 | `E39.md` | E39 | yes | PARTIAL |
| 32 | `E40.md` | E40 | yes | NOT AVAILABLE / SUPERSEDED |
| 33 | `E41.md` | E41 | yes | PARTIAL |
| 34 | `E42.md` | E42 | yes | REPRODUCIBLE |

**Excluded from the body, as instructed:** all 33 `*.audit.md` files (E1_E1b, E2, E3_E6_E7, E4,
E9_E10_E11, E12_E13_E26, E14_E14b, E15, E17, E19, E20, E21, E22, E23, E24, E25, E25b, E27, E28, E29,
E30, E31, E32, E33, E34, E35, E36, E37, E38, E39, E40, E41, E42 — there is no `E18.audit.md`), plus
`COVERAGE.md` (index only) and the non-part scratch files `e34_*.py`, `e4_*.py`, `e4_probe_out.json`.

---

## 2. IDs included

**43 IDs** are in the coverage index and have a corresponding place in the body:

E1, E1b, E2, E3, E4, E6, E7, E9, E10, E11, E12, E13, E14, E14b, E15, E16, E17, E18, E19, E20, E21,
E22, E23, E24, E25, E25b, E26, E27, E28, E29, E30, E31, E32, E33, E34, E35, E36, E37, E38, E39, E40,
E41, E42.

Notes on IDs that do not have their own body section:

- **E5, E8** — no content anywhere in `EXPERIMENTS.md`, `REPORT.md`, `REPRODUCE.md` or the parts. They
  are ID gaps and are stated as free IDs in the index note.
- **E16** — alias only. It has its own `### E16 (alias)` note pointing to E9/E10/E11 and the missing
  `multieurlex_*.py`.
- **E6, E7** — merged into the E3/E6/E7 section; no separate claim or artifact exists.
- **E10, E11** — merged into the E9/E10/E11 section; no separate artifact survives.
- **E13, E26** — merged into the E12/E13/E26 section; no separate artifact survives.
- **E14b** — merged into the E14/E14b section; same artifact, macro-F1 view.

---

## 3. Script-line audit

**No part was missing a `Script to reproduce:` line.** All 34 parts contained one.

**24 parts carried the line above the heading; it was moved directly under the heading** (the required
placement): E2, E3_E6_E7, E4, E9_E10_E11, E12_E13_E26, E14_E14b, E15, E19, E21, E22, E23, E27, E29,
E30, E31, E32, E33, E35, E36, E37, E38, E39, E40, E41.

**10 parts already had it under/at the heading:** E1_E1b, E17, E18, E20, E24, E25, E25b, E28, E34, E42.

**Duplicate script lines removed or collapsed:**

- `E27.md` and `E37.md` both carried `toxi_pr_models.py` / `toxi_pr_all.py` lines for the same wildcard.
  The draft gives E27 the two-pipeline XGBoost line and E37 the seven-classifier line; no line is
  repeated.
- `E9_E10_E11.md` and the E16 alias both point at `mel_close.py`. The E16 note now says "same script as
  E9/E10/E11" instead of restating a full duplicate command.
- Repeated headings (`E27`, `E37`, `E16`) were reduced to one heading per ID/alias.

The draft contains **35 entry-level script lines** (34 entries + the E16 alias) plus one instructional
mention in "How to read" (matched by a naive text search, not an entry).

---

## 4. Conflicting numbers between parts

These are real disagreements in the sources. The draft keeps the audited value, states the conflict, and
does not average or silently pick.

| # | conflict | values | resolution in the draft |
|---:|---|---|---|
| 1 | **E27 table vs committed JSON** | E27 table: EN/IT 0.324 (210m), 0.359 (610m); IT/IT 0.448 / 0.460; EN/EN 0.276 / 0.278. `toxi_pr_all.json`: 0.318395 / 0.360047; 0.449595 / 0.467125; 0.270920 / 0.287346. | Both shown side by side in the E27 table. The entry states the E27 cells are orphaned (no committed artifact reproduces them) and that the JSON values are authoritative. Deltas: E27 +0.035 / +0.012 / +0.002 vs JSON +0.0417 / +0.0175 / +0.0164. |
| 2 | **E27/E37 EN/EN delta** | E27 table says +0.002; the JSON difference is +0.016427. | Stated explicitly in E27: "EN/EN is +0.016, not +0.002". |
| 3 | **E9 210m calibrated row** | E9 table: 0.688 / 0.683. `mel_close.json`: 0.661 / 0.662 (qm) and 0.643 / 0.641 (tuned). Independent E18: 0.670 / 0.667 (qm) and 0.651 / 0.652. | E9 results table keeps all rows with a "NOT REPRODUCED" flag on 0.688/0.683. Verdict says the committed script does not support it. |
| 4 | **E9 610m EN self** | 0.752 in the E9 ratio table; E18 mel EN self is 0.718 (perlang_z) / 0.717 (raw, qm). | Both numbers appear in the E9 bar table and the E18 table. The difference is a protocol/threshold mismatch (E9 uses micro-F1 with its own EN-val threshold; E18 uses quantile-matched micro-F1), noted in E9 and in "How to read". |
| 5 | **E9 210m raw row** | E9 table: de 0.266 / fr 0.334 at fixed 0.5; `mel_close.json`: 0.343 / 0.464 at tuned t=0.34; E18 raw@0.5: 0.256 / 0.341. | All three shown with their threshold; E9 says the 0.266/0.334 row is the fixed-0.5 number. |
| 6 | **E9 native "ceiling"** | 0.714 / 0.712 (E9/E16 prose, unverified) vs `mel_close.json` 0.437 / 0.460 (buggy, trained on EN labels). | Both shown; the 0.437/0.460 row is marked **BUGGY**; the 0.714/0.712 row is marked **UNVERIFIED**. The 0.975 pass against 0.714/0.712 is therefore qualified "numerator unverified". |
| 7 | **E24 vs E25 PAWS-X EN-self** | E24: EN-self AUC 0.602 (train-only). E25: 0.530 (210m) / 0.628 (610m) (train+val). | Both shown; E25 explains the train+val sensitivity (adding 2,000 val rows lowers AUC 0.581 -> 0.528 in a 250-tree reproduction) and says the rows are not comparable. |
| 8 | **E42 D/B vs D′/B** | D(IT)/B(IT) = 1.0586 (3002 vs 1994 rows) vs D′(IT)/B = 1.0057 (matched 1994 rows). | Both shown; the entry says the 1.0586 is confounded with row count and that D′ is the fair comparison. Same for D/C 1.0519 vs 0.9994. |
| 9 | **E36 vs E20 toxicity** | E36 target/EN: it 0.9702, de 0.7956, fr 0.9538, es 0.9862. E20 target/EN: 0.720 / 0.752 / 0.861 / 0.709. | E36 is labelled provenance only; the delta table shows every E36 ratio is higher by 0.017–0.282. The draft says E20 supersedes E36 and E36 must not be quoted. |
| 10 | **E30 / E20 native Italian AUC** | E30 `real_it` AUC 0.8420 (210m) / 0.8430 (610m), seed 0, 18,472 rows. E20 native it 0.834 ± 0.002 (bootstrap seeds), same data. E29 in-language FT it 0.8413. | Values are kept in their own entries. They differ by pipeline (E30 raw mean + XGB, E20 perlang_z + XGB bootstrap, E29 fine-tuned), not by a contradiction; no cross-entry equality is claimed. |
| 11 | **E14 vs E3 raw macro-F1** | E3 claims 0.234 (sent) / 0.246 (agg). E14 `flores_close.py` gives 0.2908 / 0.2774 OVR micro-F1 (macro means 0.2503 / 0.2623). | E3 keeps the claims and marks them UNVERIFIED; its "nearest equivalents" table shows the E14 numbers are a different protocol and do not reproduce them. |
| 12 | **E15 FLORES ratio** | 0.543/0.540 = 1.006 from rounded values; exact unrounded 1.0047. | Both stated; the verdict uses PASS either way. |
| 13 | **E25 gap direction** | "610m shrinks the gap on every dataset" is true for \|gap\| (8/8) but false for the signed gap (7/8; PAWS-X rises by +0.0157). | The entry states both and the verdict calls the sentence false for the signed gap. |
| 14 | **E32 union ratio** | 1.145 from 4-dp table values; 1.144472 full precision; 1.1445 at 4 dp. | All three stated; no verdict change. |
| 15 | **E40 split convention** | Legacy 997 / 506 / 506 vs current 997 / 504 / 505. | E40 states both and says numbers are not interchangeable. |
| 16 | **E21 `transfer` mean** | E21's `transfer` is the mean of EN→IT and IT→EN; the audit's 0.975 bar uses EN→IT only. | Both directions are shown separately; the entry says the mean hides asymmetry. |

No conflict was resolved by inventing a number. Where the sources disagree and neither is verifiable
(E9 610m, E27 table, E3/E6/E7, E4), the draft keeps both values and marks the unverified one.

---

## 5. Sections that could not be assembled as verified content

| item | reason | treatment in the draft |
|---|---|---|
| E1/E1b INLP claim (claim 4) | No FLORES language basis, no accuracy curve, no stopping log. | Claims 1–3 reproduced; claim 4 marked UNVERIFIED; entry status PARTIAL/UNK. |
| E3/E6/E7 | No script, no artifact, no per-target rows, no English-self reference. | Documentation-only claims table, all marked UNVERIFIED; no 0.975 ratio constructed. |
| E4 | No script, no result artifact; only two 4-dp values quoted in temp diagnostics. | Claims table marked UNVERIFIED; the 0.975 table uses only those unverified numbers and says so. |
| E9/E10/E11 610m | Text-only row from the initial commit; no JSON/MD/PNG. | Kept as "UNVERIFIED / NOT AVAILABLE"; only the 210m `mel_close` artifact is verified. |
| E13, E26 | No separate artifact or claim. | Merged into E12; stated that E12 is the only evidence for all three IDs. |
| E16 | Not a separate experiment. | Short alias section pointing to E9/E10/E11 and the missing scripts. |
| E29 toxic fine-tuned arm | 400 steps = 53% of one epoch, early stopping cannot trigger, caps disagree across JSON/log/default. | Reported with the caveat; no pass claimed. |
| E31 toxic SmolLM2 | Generation stopped at 4800/8000. | Table row marked "skipped"; toxic arm marked NOT DONE. |
| E32 toxic union | 2 of 3 generators; MD labels it "ALL generators". | Reported as partial; the draft explicitly says not to cite it as all-generator. |
| E36 | Leaky splits; no embedding cache; no saved models. | Provenance-only entry with "DO NOT QUOTE"; deltas vs E20 given. |
| E40 result tables | Deleted from the worktree; only the `aa3f6f6` archive survives. | Entry is marked SUPERSEDED/UNK and records which cells were verified by re-fit and which were not. |
| E41 SHAP JSON/MD | Never tracked; deleted. | Only the PNG-derived Spearman and approximate shares are reported; deleted numbers marked UNVERIFIED. |
| E39 Europarl | No embeddings, no md/JSON/log; directories empty. | Six README rows quoted and marked "NOT re-run / unverified". |
| E2 `translate_datasets.py` | Never in git. | E2 status PARTIAL; `rebuild_caches.py` documented as the surviving path. |
| E33 full rerun | `agg_pr_methods` missing; script dies at import. | Full rerun marked NOT AVAILABLE; only `--reemit-from-json` is supported. |
| E5, E8 | No content anywhere. | Index note only; not invented. |

**Audit-only material deliberately omitted from the body:** the `Verified`, `Discrepancies`,
`Commands run`, `Bugs/risks`, `Missing scripts`, and `Recommended changes` sections of `E27.md`; the
line-level `Verified` notes of `E23.md`/`E37.md` were folded into one-line provenance statements; the
`E37` bug list was reduced to the caveats that affect the numbers. All of it remains in the `*.audit.md`
files.

---

## 6. Assembly checks on the output

| check | result |
|---|---|
| File exists and is non-empty | yes — 4,490 lines, ~298 KB |
| `EXPERIMENTS.md` unmodified | yes — `git diff -- EXPERIMENTS.md` empty |
| Part markers / HTML comments left behind | none |
| Entry headings | 35 (34 entries + E16 alias) |
| Coverage-index rows | 43 |
| `Script to reproduce:` lines | 35 entry lines (+1 instructional mention) |
| Duplicate entry headings | none |
| Duplicate script lines | none (E27/E37 split; E16 collapsed to "same script") |
| UTF-8 | yes — em dashes, `×`, `η²`, `→`, `≥`, `Δ`, `†`, `±` all present |
| Arrows / comparisons | `->` and `>=` used in prose and tables; UTF-8 arrows kept only in a few heading titles and in the E42 title |
| Numbers re-audited during merge | none — every value is copied from its part |

**One residual formatting note:** the PART banners are H1 (`# PART A — ...`) while the document title is
also H1 and entries are H2. This is readable but not a strict single-H1 outline. It was left as-is to
avoid a mass heading-level rewrite of 35 entries and the E21 sub-tables.
