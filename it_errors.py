"""FLORES EN-trained vs IT-trained model error analysis on the ITALIAN test set.

For each dataset that has Italian (``sent`` 12-class, ``agg`` 11-class) this trains TWO
XGBoost models and evaluates both on the SAME Italian test rows:

* **EN-trained**: fit on the EN *train* rows, scored on the IT test rows (cross-lingual transfer).
* **IT-trained**: fit on the IT *train* rows, scored on the IT test rows (in-language).

Features match the project recipe (``flores_calib`` / ``proof_mean_alllangs``):
EuroBERT-210m **mean** embeddings + ``perlang_z [both]`` = L2-normalize then per-language
OWN *train* z-score, applied to both the model side and the Italian test side.  Data come from
``train_flores12`` (``prepare_sentence`` + ``load_sent_embeddings('mean', kept, 2009)``;
``prepare_agg`` + ``load_agg_embeddings('mean', 562)``).  XGBoost params = ``flores_calib.XGB_COMMON``
(softprob, hist, max_bin=64, cpu, n_jobs=2, seed=0, depth=8, min_child_weight=5, lr=0.1, n=400)
plus ``num_class`` = 12 (sent) / 11 (agg); both models are trained on the *train* split only.

Buckets over the Italian test items (label = gold):
* ``both_wrong``     : EN wrong AND IT wrong   ("hard" items)
* ``transfer_gap``   : EN wrong AND IT right   (transfer-specific misses)
* ``it_only_wrong``  : EN right AND IT wrong
* ``both_right``

Outputs (create-only): ``it_errors.py`` / ``it_errors.json`` / ``it_errors.md``.

Usage
-----
    python it_errors.py
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import json
import sys
import time

import numpy as np
import xgboost as xgb

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import train_flores12 as T                       # noqa: E402  data prep + alignment
import flores_calib as FC                        # noqa: E402  perlang_z + XGB_COMMON

OUT_JSON = os.path.join(REPO, "it_errors.json")
OUT_MD = os.path.join(REPO, "it_errors.md")

LANGS = ["en", "it", "de", "fr"]
SEED = 0
MAX_LIST = 30
SNIPPET = 200

_T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - _T0, msg), flush=True)


def _jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError("not JSON serializable: %r" % (type(o),))


def _clean_snippet(text, n=SNIPPET):
    """Single-line snippet of <= n chars, utf-8 safe (replaces undecodable bytes)."""
    s = str(text).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split())
    if len(s) > n:
        s = s[:n] + "..."
    return s.encode("utf-8", "replace").decode("utf-8", "replace")


def build(name):
    """Return the raw (untransformed) dataset dict for ``sent`` or ``agg``."""
    sent = T.prepare_sentence()
    classes = sent["classes"]
    agg = T.prepare_agg(sent["cur"], classes)
    if name == "sent":
        X = T.load_sent_embeddings("mean", sent["kept"], len(sent["git"]))
        y = sent["y"]
        tr, va, te = sent["train_idx"], sent["val_idx"], sent["test_idx"]
        C = 12
        cls = classes
        texts = sent["cur"]["it"].tolist()
    else:
        X = T.load_agg_embeddings("mean", len(agg["agg"]))
        y = agg["y"]
        tr, va, te = agg["train_idx"], agg["val_idx"], agg["test_idx"]
        C = agg["n_classes"]
        cls = agg["classes"]
        texts = agg["agg"]["it"].tolist()
    X = {lg: np.asarray(X[lg], dtype=np.float64) for lg in LANGS}
    return {"name": name, "X": X, "y": np.asarray(y, dtype=int),
            "train": tr, "val": va, "test": te, "n_classes": C, "classes": cls,
            "texts": texts}


def confusion(y, pred, C):
    """Full CxC confusion matrix; rows = gold, cols = predicted."""
    M = np.zeros((C, C), dtype=int)
    for t, j in zip(y, pred):
        M[int(t), int(j)] += 1
    return M


def run_dataset(D):
    name, C = D["name"], D["n_classes"]
    y, tr, te = D["y"], D["train"], D["test"]
    log("==== DATASET %s (C=%d, train=%d val=%d test=%d) ===="
        % (name, C, len(tr), len(D["val"]), len(te)))

    # perlang_z [both]: each language's own TRAIN stats (L2-normalize + z-score).
    Xt = FC.perlang_z(D["X"], tr)
    log("  perlang_z done (en/it/de/fr, stats from train rows)")

    params = dict(FC.XGB_COMMON)
    params["num_class"] = C

    en_model = xgb.XGBClassifier(**params)
    en_model.fit(Xt["en"][tr], y[tr])
    it_model = xgb.XGBClassifier(**params)
    it_model.fit(Xt["it"][tr], y[tr])
    log("  EN + IT models fitted")

    en_pred = np.asarray(en_model.predict(Xt["it"][te]), dtype=int)   # EN model -> IT test
    it_pred = np.asarray(it_model.predict(Xt["it"][te]), dtype=int)   # IT model -> IT test
    gold = np.asarray(y[te], dtype=int)

    en_wrong = en_pred != gold
    it_wrong = it_pred != gold
    both_wrong = en_wrong & it_wrong
    transfer_gap = en_wrong & ~it_wrong
    it_only_wrong = ~en_wrong & it_wrong
    both_right = ~en_wrong & ~it_wrong

    en_acc = float((~en_wrong).mean())
    it_acc = float((~it_wrong).mean())

    log("  EN acc=%.4f (errors=%d) | IT acc=%.4f (errors=%d)"
        % (en_acc, int(en_wrong.sum()), it_acc, int(it_wrong.sum())))
    log("  both_wrong=%d transfer_gap=%d it_only_wrong=%d both_right=%d"
        % (int(both_wrong.sum()), int(transfer_gap.sum()),
           int(it_only_wrong.sum()), int(both_right.sum())))

    cls = D["classes"]
    texts = D["texts"]

    def rec(i):
        # ``i`` is a position in the test arrays; ``te[i]`` is the CSV row it came from.
        row = int(te[i])
        assert texts[row] is not None
        return {"index": row, "test_pos": int(i), "text": _clean_snippet(texts[row]),
                "gold": int(gold[i]), "gold_label": cls[int(gold[i])],
                "en_pred": int(en_pred[i]), "en_pred_label": cls[int(en_pred[i])],
                "it_pred": int(it_pred[i]), "it_pred_label": cls[int(it_pred[i])]}

    def collect(mask):
        return [rec(int(i)) for i in np.where(mask)[0]]

    bw = collect(both_wrong)
    tg = collect(transfer_gap)

    M = confusion(gold, en_pred, C)
    error_pairs = []
    for g in range(C):
        for p in range(C):
            if g != p and M[g, p] > 0:
                error_pairs.append({"gold": g, "gold_label": cls[g],
                                    "en_pred": p, "en_pred_label": cls[p],
                                    "count": int(M[g, p])})
    error_pairs.sort(key=lambda r: (-r["count"], r["gold"], r["en_pred"]))

    return {
        "name": name, "n_classes": C, "classes": cls,
        "sizes": {"train": int(len(tr)), "val": int(len(D["val"])), "test": int(len(te))},
        "feature": "perlang_z [both] (L2-norm + per-language TRAIN z-score)",
        "xgb": params,
        "trained_on": "train split only (EN model: EN train; IT model: IT train)",
        "evaluated_on": "Italian (it) test rows",
        "bucket_counts": {
            "both_wrong": int(both_wrong.sum()),
            "transfer_gap": int(transfer_gap.sum()),
            "it_only_wrong": int(it_only_wrong.sum()),
            "both_right": int(both_right.sum()),
        },
        "en_accuracy": en_acc,
        "it_accuracy": it_acc,
        "en_errors_total": int(en_wrong.sum()),
        "it_errors_total": int(it_wrong.sum()),
        "confusion_en_rows_gold_cols_pred": M.tolist(),
        "en_error_pairs": error_pairs,
        "both_wrong_total": len(bw),
        "transfer_gap_total": len(tg),
        "both_wrong_listed": min(MAX_LIST, len(bw)),
        "transfer_gap_listed": min(MAX_LIST, len(tg)),
        "both_wrong": bw[:MAX_LIST],
        "transfer_gap": tg[:MAX_LIST],
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def md_dataset(res):
    L = []
    C = res["n_classes"]
    cls = res["classes"]
    L.append("\n## %s (%d classes)\n" % (res["name"], C))
    L.append("- Splits: train %d / val %d / test %d (Italian test = %d rows)."
             % (res["sizes"]["train"], res["sizes"]["val"], res["sizes"]["test"],
                res["sizes"]["test"]))
    L.append("- Features: `%s`; XGBoost `%s`; %s. %s."
             % (res["feature"], "softprob depth8/mcw5/lr0.1/n400",
                res["trained_on"], "Evaluated on " + res["evaluated_on"] + "."))
    L.append("- Classes: %s.\n" % ", ".join("`%s`" % c for c in cls))

    b = res["bucket_counts"]
    L.append("**Buckets (Italian test):** both_wrong (EN wrong & IT wrong) **%d**, "
             "transfer_gap (EN wrong & IT right) **%d**, it_only_wrong (EN right & IT wrong) "
             "**%d**, both_right **%d**. (total %d)\n"
             % (b["both_wrong"], b["transfer_gap"], b["it_only_wrong"], b["both_right"],
                sum(b.values())))
    L.append("**Accuracy on Italian test:** EN-trained **%.4f** (%d errors) | "
             "IT-trained **%.4f** (%d errors).\n"
             % (res["en_accuracy"], res["en_errors_total"],
                res["it_accuracy"], res["it_errors_total"]))

    L.append("**Full confusion of EN-trained errors on Italian test** "
             "(rows = gold, cols = EN pred; diagonal omitted):\n")
    L.append("| gold \\ EN pred | count |")
    L.append("|---|---:|")
    for r in res["en_error_pairs"]:
        L.append("| `%s` -> `%s` | %d |" % (r["gold_label"], r["en_pred_label"], r["count"]))

    L.append("\n### both_wrong (hard: EN wrong & IT wrong) — %d total, showing %d\n"
             % (res["both_wrong_total"], res["both_wrong_listed"]))
    if res["both_wrong"]:
        L.append("| CSV row | text | gold | EN pred | IT pred |")
        L.append("|---:|---|---|---|---|")
        for r in res["both_wrong"]:
            L.append("| %d | %s | %s | %s | %s |"
                     % (r["index"], r["text"], r["gold_label"], r["en_pred_label"],
                        r["it_pred_label"]))
    else:
        L.append("_none_")

    L.append("\n### transfer_gap (EN wrong & IT right) — %d total, showing %d\n"
             % (res["transfer_gap_total"], res["transfer_gap_listed"]))
    if res["transfer_gap"]:
        L.append("| CSV row | text | gold | EN pred | IT pred |")
        L.append("|---:|---|---|---|---|")
        for r in res["transfer_gap"]:
            L.append("| %d | %s | %s | %s | %s |"
                     % (r["index"], r["text"], r["gold_label"], r["en_pred_label"],
                        r["it_pred_label"]))
    else:
        L.append("_none_")
    return L


def build_md(datasets):
    md = []
    md.append("# FLORES IT-test error analysis: EN-trained vs IT-trained model "
              "(EuroBERT-210m mean + `perlang_z [both]`)\n")
    md.append("Two XGBoost models per dataset, both evaluated on the **Italian test rows**: "
              "**EN-trained** (fit on EN train) and **IT-trained** (fit on IT train). Features = "
              "`perlang_z [both]` (L2-normalize + per-language OWN train z-score, applied to both "
              "sides). Data via `train_flores12` (`prepare_sentence` + "
              "`load_sent_embeddings('mean', kept, 2009)`; `prepare_agg` + "
              "`load_agg_embeddings('mean', 562)`). XGBoost = `flores_calib.XGB_COMMON` + "
              "`num_class` (softprob, hist, max_bin=64, cpu, n_jobs=2, seed=0, depth=8, "
              "min_child_weight=5, lr=0.1, n=400). Both models train on the **train** split; "
              "evaluated on the **Italian test** rows.\n")
    md.append("Buckets: **both_wrong** = EN wrong AND IT wrong (\"hard\"); **transfer_gap** = "
              "EN wrong AND IT right; **it_only_wrong** = EN right AND IT wrong; "
              "**both_right**. Text is the Italian (`it`) column.\n")
    for ds in ("sent", "agg"):
        md.extend(md_dataset(datasets[ds]))
        md.append("")
    md.append("\n## Summary\n")
    md.append("| dataset | C | EN acc | IT acc | both_wrong | transfer_gap | "
              "it_only_wrong | both_right |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for ds in ("sent", "agg"):
        r = datasets[ds]
        b = r["bucket_counts"]
        md.append("| %s | %d | %.4f | %.4f | %d | %d | %d | %d |"
                  % (ds, r["n_classes"], r["en_accuracy"], r["it_accuracy"],
                     b["both_wrong"], b["transfer_gap"], b["it_only_wrong"], b["both_right"]))
    md.append("")
    return md


def main():
    datasets = {}
    for ds in ("sent", "agg"):
        datasets[ds] = run_dataset(build(ds))

    payload = {
        "config": {
            "emb": "EuroBERT-210m mean",
            "feature": "perlang_z [both]",
            "xgb": FC.XGB_COMMON,
            "trained_on": "train split only",
            "evaluated_on": "Italian (it) test rows",
            "seed": SEED,
            "max_list": MAX_LIST, "snippet_chars": SNIPPET,
            "python": sys.version.split()[0],
            "xgboost": xgb.__version__,
        },
        "datasets": datasets,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_jd)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(build_md(datasets)) + "\n")
    log("wrote %s , %s" % (OUT_JSON, OUT_MD))
    for ds, r in datasets.items():
        b = r["bucket_counts"]
        log("%s: EN acc=%.4f IT acc=%.4f | both_wrong=%d transfer_gap=%d it_only_wrong=%d "
            "both_right=%d"
            % (ds, r["en_accuracy"], r["it_accuracy"], b["both_wrong"], b["transfer_gap"],
               b["it_only_wrong"], b["both_right"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
