### E42 sidecar -- arm F: raw translate-test vs translate-train -- DONE

**Question.** On the same 505 Italian documents, with the same raw features and the same model class, does translating the **test** text beat translating the **training** text?

**Script to reproduce:** `en_vs_enit_ttest.py` -- `python en_vs_enit_ttest.py --smoke` then `python en_vs_enit_ttest.py --launch`. Reproduction status: REPRODUCIBLE (deterministic seed 0; cache md5-verified).

**Setup.** F = arm A's protocol: 997 raw EN train rows, 10 Optuna TPE trials (seed 0), objective = OVR-macro AUC on raw EN val (504). Evaluated on the MT Italian test translated to English (`Helsinki-NLP/opus-mt-it-en`, E23 cache) with raw EuroBERT-210m mean features -- no L2, no z-score, no per-language z. Also scored on the raw EN test and the raw real-IT test for reference. **No test labels are used.**

**Cache provenance.** md5 of the Italian test texts in `test_idx` order = `b649a1e26ee0` (matches the MT file tag), md5 of the 505 translations = `9e50ebbf6a1d` (matches the npy tag); a fresh mean-pool of 4 translated rows reproduces the cache at cos >= 0.999999940. Shape [505, 768]. So the vectors are exactly these 505 documents, in this order.

| F evaluation set | accuracy | micro-F1 | macro-F1 | OVR-macro AUC |
|---|---:|---:|---:|---:|
| MT Italian test -> EN (translate-test) | 0.5386 | 0.5386 | 0.4272 | **0.8609** |
| raw EN test (reference) | 0.5446 | 0.5446 | 0.4149 | **0.8739** |
| raw real-IT test (reference, = A) | 0.3901 | 0.3901 | 0.1799 | **0.7880** |

**Controlled 3-way comparison on the same 505 documents (raw features throughout).**

| arm | strategy | input at test | train rows | IT AUC | IT macro-F1 |
|---|---|---|---:|---:|---:|
| A | none | real Italian | 997 EN | 0.7880 | 0.1799 |
| E | translate-train | real Italian | 1994 (EN + MT-IT) | 0.8418 | 0.3547 |
| **F** | **translate-test** | **MT English** | 997 EN | **0.8609** | **0.4272** |

| ratio | value | >= 0.975 | reading |
|---|---:|:--:|---|
| F(IT) / E(IT) | **1.0227** | yes | translate-test beats translate-train on AUC |
| F(IT) / A(IT) | **1.0926** | yes | translating the test beats no translation |
| E(IT) / A(IT) | **1.0683** | yes | translating the train beats no translation |

**Verdict.** On the same 505 Italian documents with identical raw features, **translate-test wins on both metrics -- narrowly on AUC (1.9 pp, about 1-2 SE) and clearly on macro-F1 (+20%)**: F = 0.8609 AUC / 0.4272 macro-F1 vs E (translate-train) = 0.8418 / 0.3547 and A (no translation) = 0.7880 / 0.1799. F/E = **1.0227**, F/A = **1.0926**, E/A = **1.0683** on AUC; on macro-F1 F/E = **1.2042**. The winner on AUC is **F (translate-test)** and on macro-F1 **F (translate-test)**. The price is operational: translate-test runs MT at inference on every test item (505 translations here, re-run for every new item), while translate-train pays MT once on 997 training items and serves with no MT in the loop. Translate-test buys roughly 1.9 AUC points over translate-train for that inference-time cost.

**Operational cost.**
- translate-test: MT runs **at inference, once per test item** (505 translations here), and must re-run for every new item at serving time; latency and MT compute scale with traffic.
- translate-train: MT runs **once over the 997 training items**; serving is pure encoder + XGBoost with **no MT in the loop**, and the model can be retrained without re-translating test data.

**Caveats.**
- Translate-test changes the test-time representation: the documents are the same 505, the text the model sees is MT English. Label noise is unchanged (labels are human), but MT artifacts are part of the input.
- F is not a new model: its training protocol equals A's, and with seed 0 it selects the same hyperparameters, so F's raw EN/IT test numbers reproduce A's exactly. The comparison is therefore a pure test-representation swap.
- E (translate-train) trains on 1994 rows vs F's 997, so F wins despite having half the training rows.
- Single seed, 505-row test; AUC SE ~0.01-0.02. The F/E AUC gap is only 1.9 pp (about 1-2 SE), so the AUC win is a weak margin; the macro-F1 gap is 7.3 pp (+20% relative) and is the decisive effect. F/A and E/A (7-9 pp AUC) are well outside the noise band.
- MT is opus-mt-it-en for the test and opus-mt-en-it for the training arm, i.e. different directions/models; a weaker MT system on either side would move that arm.
- Inference-time MT also needs the encoder run over 505 translated strings, so serving cost is MT + encoder + XGBoost rather than encoder + XGBoost.
- E23's translate-test cell used this same cache but 1501 train+val rows and fixed params, scoring 0.8969 vs F's 0.8609. The gap is training rows (1501 vs 997), not the cache: F's raw EN test (0.8739) also sits below E23's EN test (0.9038).
