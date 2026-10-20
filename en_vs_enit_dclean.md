### E42 sidecar -- D_clean (arm E): leak-free translate-train -- DONE

**Question.** With A/B/C/D already stored, what does translate-train score when the Optuna objective is measured on a validation split that no training row touches -- and how much of D's 0.8860 was inflation?

**Script to reproduce:** `en_vs_enit_dclean.py` -- `python en_vs_enit_dclean.py --smoke` then `python en_vs_enit_dclean.py --launch`. Reproduction status: REPRODUCIBLE (deterministic seed 0, fresh study).

**Setup.** Train = 1994 rows (997 EN **train** + 997 MT-IT of those same rows). Tune = 10 fresh Optuna TPE trials (seed 0), objective = OVR-macro AUC on the **unseen** real IT val (504). Test = real IT test (505) and EN test (505). Features raw mean-pooled EuroBERT-210m float32; no L2, z-score, per-lang z, PCA/INLP, and **no test-time MT**.

**Held-out check.** split 997 / 504 / 505, pairwise disjoint = True; val rows inside the training matrix = **0**; MT rows used = 997 (train slice only), val MT rows used = **0**.

| cell | accuracy | micro-F1 | macro-F1 | OVR-macro AUC |
|---|---:|---:|---:|---:|
| E: EN + MT-IT (1994) -> IT test | 0.5109 | 0.5109 | 0.3547 | **0.8418** |
| E: EN + MT-IT (1994) -> EN test | 0.5406 | 0.5406 | 0.4243 | **0.8827** |

| ratio | value | >= 0.975 | reading |
|---|---:|:--:|---|
| E(IT) / B(IT) | **1.0057** | yes | MT matches real Italian training |
| E(IT) / C(IT) | **0.9994** | yes | at the Italian ceiling |
| E(EN) / A(EN) | **1.0101** | yes | English retained |
| E(IT) / D(IT) | **0.9501** | no | D is the 3002-row final fit (val rows included) |
| E(IT) / D'(IT) | **1.0000** | yes | D' = D's params on the clean 1994 rows |

**Reference numbers (unchanged, read from `en_vs_enit_raw.json`).**

| arm | rows | tuned on | IT test AUC | EN test AUC |
|---|---:|---|---:|---:|
| A (EN only) | 997 | EN val | 0.7880 | 0.8739 |
| B (EN+IT) | 1994 | IT val | 0.8370 | 0.8729 |
| C (IT only) | 997 | IT val | 0.8423 | - |
| D (EN+MT-IT, train+val) | 3002 | IT val (train-only fit) | 0.8860 | 0.9161 |
| D' (D's params, 1994 rows) | 1994 | inherited | 0.8418 | 0.8827 |
| **E (clean, 1994 rows)** | 1994 | **held-out IT val** | **0.8418** | **0.8827** |

**Verdict.** Clean translate-train scores **0.8418** AUC on real Italian vs **0.8370** for B (real Italian train) and **0.8423** for the Italian-only ceiling C: E/B = **1.0057** (above the 0.975 bar) and E/C = **0.9994** (above). On English it scores **0.8827** vs A's **0.8739** (E/A = **1.0101**, within the bar). So with an honest held-out objective and matched rows, MT Italian is a genuine substitute for real Italian training data. D's raw 0.8860 is **not comparable**, but not because of in-sample tuning: D's Optuna objective was scored on a 1994-row train-only fit, exactly as E does here, which is why E's 10 trials and best params reproduce D's bitwise. The gap is a final-fit row-count effect -- D refits on 3002 rows (1501 EN train+val + MT-IT of those same 1501, i.e. 1.5x B and 3x A, and the 504 IT-val rows are inside it), while E refits on the 1994 clean rows. That is +0.0442 AUC, and it is why D/C = 1.0519 was misleading.

**Cross-check against D (why the gap is not a tuning leak).** E's 10 trial values are bit-identical to D's stored 10 (True); E's best params equal D's (True); E's test numbers equal D's 1994-row variant to 1e-12 (True). D therefore already tuned on a train-only 1994-row fit, so its Optuna objective was held-out. The 0.0442 gap to D's headline number is entirely D's 3002-row final fit, which contains the 504 val rows.

**Caveats.**
- Single seed (0); 505-row test. AUC SE ~0.01-0.02, so E/B and E/C differences under ~0.02 AUC are not resolved.
- E trains on 1994 rows, the same size as B, so E/B and E/C are size-matched -- unlike D/B (3002 vs 1994) and D/C (3002 vs 997).
- Only EN train rows were translated; the 504 val and 505 test texts are the original human Italian and are never translated or trained on.
- MT output is synthetic; a different MT system would move E. The number is specific to Helsinki-NLP/opus-mt-en-it.
- 10 Optuna trials: enough to beat defaults, not to exhaust the space. E's params happen to coincide with D's because both tuned the same 1994-row matrix with the same seed -- a cross-check, not a copy.
- The parent E42 JSON is read-only for reference numbers and asserted against expected values, so E cannot silently inherit a drifted baseline.
