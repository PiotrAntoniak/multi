"""List where a SOURCE-language-trained model misclassifies TARGET-language test text.

Generalises ``it_errors.py`` to any (dataset, source, target) combination over the FLORES
languages available in this repo (``en``, ``it``, ``de``, ``fr``).

For the chosen dataset (``sent`` 12-class or ``agg`` 11-class) this trains TWO XGBoost
models and evaluates BOTH on the SAME **target-language test rows**:

* **source-trained**: fit on the SOURCE-language *train* embeddings.
* **target-trained**: fit on the TARGET-language *train* embeddings.

Both models are scored on the TARGET-language test embeddings.  Features match the project
recipe (``flores_calib`` / ``proof_mean_alllangs``): EuroBERT-210m **mean** embeddings +
``perlang_z [both]`` = L2-normalize then per-language OWN train z-score.  Data come from
``train_flores12`` (``prepare_sentence`` + ``load_sent_embeddings('mean', kept, 2009)``;
``prepare_agg`` + ``load_agg_embeddings('mean', 562)``).  XGBoost params = ``flores_calib.
XGB_COMMON`` (softprob, hist, max_bin=64, cpu, n_jobs=2, seed=0, depth=8,
min_child_weight=5, lr=0.1, n=400) plus ``num_class``.

Buckets over the target test items (label = gold):
* ``both_wrong``        : source wrong AND target wrong   ("hard" items)
* ``transfer_gap``      : source wrong AND target right   (transfer-specific misses)
* ``target_only_wrong`` : source right AND target wrong
* ``both_right``

Each listed item shows the text in THREE forms (dynamic keys ``<target>_text`` /
``<source>_gold`` / ``<target>2<source>_mt``; for en->it: ``it_text`` / ``en_gold`` /
``it2en_mt``):
1. target text (FLORES ``<target>`` column);
2. aligned gold source text (FLORES ``<source>`` column -- parallel human translation);
3. target->source machine translation, ``Helsinki-NLP/opus-mt-<target>-<source>``
   (transformers; downloaded on demand, fp16 greedy batched on GPU when available; the
   output is cached in ``mt_cache_<target>2<source>.json``).

Each listed item also carries the predicted-class and gold-class probabilities of BOTH models
(keys ``<source>_pred_prob`` / ``<source>_gold_prob`` / ``<target>_pred_prob`` /
``<target>_gold_prob``; for en->it: ``en_pred_prob`` / ``en_gold_prob`` / ``it_pred_prob`` /
``it_gold_prob``), from ``predict_proba`` (max softmax prob = argmax class).

Outputs (create-only): ``list_errors.md`` / ``list_errors.json`` (+ the MT cache).  The repo
has an existing ``it_errors.py`` (hard-coded en->it) that this script reproduces with
``python list_errors.py --dataset sent`` (and ``--dataset agg``).

Usage
-----
    python list_errors.py --dataset sent|agg --source en --target it [--show 30]
                          [--train-on train|trainval] [--no-calib]

Defaults: ``--dataset sent --source en --target it --show 30 --train-on train``.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
    os.environ[_v] = "2" if _v != "TOKENIZERS_PARALLELISM" else "false"

import argparse
import json
import sys
import time

import numpy as np
import torch
import xgboost as xgb

torch.set_num_threads(2)

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import train_flores12 as T                       # noqa: E402  data prep + alignment
import flores_calib as FC                        # noqa: E402  perlang_z + XGB_COMMON

OUT_JSON = os.path.join(REPO, "list_errors.json")
OUT_MD = os.path.join(REPO, "list_errors.md")

LANGS = ["en", "it", "de", "fr"]
DATASETS = ["sent", "agg"]
SEED = 0
SNIPPET = 200

# --- Machine translation (opus-mt, greedy, batched) ---
MT_BATCH = 64
MT_MAX_LEN = 512
_MT_CACHE = {}


def mt_model_id(target, source):
    return "Helsinki-NLP/opus-mt-%s-%s" % (target, source)


def mt_cache_path(target, source):
    return os.path.join(REPO, "mt_cache_%s2%s.json" % (target, source))


def _translate_uncached(texts, model_id, device):
    """Batch greedy translation with opus-mt (fp16 on GPU), returns list[str]."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    log("  [MT] loading %s (downloads on demand) ..." % model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    model = model.half().eval() if device == "cuda" else model.eval()
    out = []
    n = len(texts)
    nb = (n + MT_BATCH - 1) // MT_BATCH
    with torch.no_grad():
        for b in range(nb):
            chunk = texts[b * MT_BATCH:(b + 1) * MT_BATCH]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=MT_MAX_LEN).to(device)
            gen = model.generate(**enc, max_length=MT_MAX_LEN, num_beams=1, do_sample=False)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def translate_cached(texts, target, source, device):
    """Translate TARGET texts into SOURCE with opus-mt-<target>-<source>.

    Persistent cache ``mt_cache_<target>2<source>.json`` (text -> translation); only the
    missing strings are translated.  Returns ``{text: translation_or_None}``.
    """
    uniq = sorted({t for t in texts if t is not None and str(t) != ""})
    key = (target, source)
    if key not in _MT_CACHE:
        path = mt_cache_path(target, source)
        cache = {}
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                cache = json.load(f)
        _MT_CACHE[key] = cache
    cache = _MT_CACHE[key]
    missing = [t for t in uniq if t not in cache]
    if missing:
        model_id = mt_model_id(target, source)
        log("  [MT] %s: %d/%d strings cached; translating %d new"
            % (model_id, len(uniq) - len(missing), len(uniq), len(missing)))
        try:
            trans = _translate_uncached(missing, model_id, device)
        except Exception as e:  # noqa: BLE001  (no opus-mt for this pair, offline, ...)
            log("  [MT] UNAVAILABLE for %s->%s (%s: %s); leaving MT blank"
                % (target, source, type(e).__name__, e))
            return {t: cache.get(t) for t in texts}
        for t, u in zip(missing, trans):
            cache[t] = u
        path = mt_cache_path(target, source)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        log("  [MT] cached %d strings -> %s" % (len(cache), os.path.basename(path)))
    return {t: cache.get(t) for t in texts}

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


def build(name, source, target):
    """Return the raw (untransformed) dataset dict for ``sent`` or ``agg``.

    ``texts`` are the TARGET-language strings and ``gold_texts`` the SOURCE-language strings
    (FLORES is parallel), both in CSV row order.
    """
    sent = T.prepare_sentence()
    classes = sent["classes"]
    agg = T.prepare_agg(sent["cur"], classes)
    if name == "sent":
        X = T.load_sent_embeddings("mean", sent["kept"], len(sent["git"]))
        y = sent["y"]
        tr, va, te = sent["train_idx"], sent["val_idx"], sent["test_idx"]
        C = 12
        cls = classes
        cur = sent["cur"]
    else:
        X = T.load_agg_embeddings("mean", len(agg["agg"]))
        y = agg["y"]
        tr, va, te = agg["train_idx"], agg["val_idx"], agg["test_idx"]
        C = agg["n_classes"]
        cls = agg["classes"]
        cur = agg["agg"]
    texts = cur[target].tolist()
    gold_texts = cur[source].tolist()
    X = {lg: np.asarray(X[lg], dtype=np.float64) for lg in LANGS}
    return {"name": name, "X": X, "y": np.asarray(y, dtype=int),
            "train": tr, "val": va, "test": te, "n_classes": C, "classes": cls,
            "texts": texts, "gold_texts": gold_texts}


def confusion(y, pred, C):
    """Full CxC confusion matrix; rows = gold, cols = predicted."""
    M = np.zeros((C, C), dtype=int)
    for t, j in zip(y, pred):
        M[int(t), int(j)] += 1
    return M


def run_dataset(D, source, target, train_on):
    name, C = D["name"], D["n_classes"]
    y, tr, va, te = D["y"], D["train"], D["val"], D["test"]
    fit_rows = tr if train_on == "train" else np.concatenate([tr, va])
    log("==== DATASET %s (C=%d) | source=%s -> target=%s | train-on=%s ===="
        % (name, C, source, target, train_on))
    log("  splits: fit=%d (train %d + val %d) test=%d"
        % (len(fit_rows), len(tr), 0 if train_on == "train" else len(va), len(te)))

    # perlang_z [both]: each language's own FIT-row stats (L2-normalize + z-score).
    Xt = FC.perlang_z(D["X"], fit_rows)
    log("  perlang_z done (en/it/de/fr, stats from fit rows)")

    params = dict(FC.XGB_COMMON)
    params["num_class"] = C

    src_model = xgb.XGBClassifier(**params)
    src_model.fit(Xt[source][fit_rows], y[fit_rows])
    tgt_model = xgb.XGBClassifier(**params)
    tgt_model.fit(Xt[target][fit_rows], y[fit_rows])
    log("  %s-trained + %s-trained models fitted" % (source.upper(), target.upper()))

    # Both models score the TARGET-language test embeddings (argmax(proba) == predict()).
    src_proba = np.asarray(src_model.predict_proba(Xt[target][te]), dtype=np.float64)
    tgt_proba = np.asarray(tgt_model.predict_proba(Xt[target][te]), dtype=np.float64)
    src_pred = src_proba.argmax(1)
    tgt_pred = tgt_proba.argmax(1)
    gold = np.asarray(y[te], dtype=int)

    src_wrong = src_pred != gold
    tgt_wrong = tgt_pred != gold
    both_wrong = src_wrong & tgt_wrong
    transfer_gap = src_wrong & ~tgt_wrong
    target_only_wrong = ~src_wrong & tgt_wrong
    both_right = ~src_wrong & ~tgt_wrong

    src_acc = float((~src_wrong).mean())
    tgt_acc = float((~tgt_wrong).mean())

    log("  %s acc=%.4f (errors=%d) | %s acc=%.4f (errors=%d)"
        % (source.upper(), src_acc, int(src_wrong.sum()),
           target.upper(), tgt_acc, int(tgt_wrong.sum())))
    log("  both_wrong=%d transfer_gap=%d target_only_wrong=%d both_right=%d"
        % (int(both_wrong.sum()), int(transfer_gap.sum()),
           int(target_only_wrong.sum()), int(both_right.sum())))

    cls = D["classes"]
    texts = D["texts"]
    gold_texts = D["gold_texts"]
    SRC_UP, TGT_UP = source.upper(), target.upper()
    t_key = "%s_text" % target
    s_key = "%s_gold" % source
    mt_key = "%s2%s_mt" % (target, source)
    sp_key = "%s_pred_prob" % source      # source model's predicted-class probability
    sg_key = "%s_gold_prob" % source      # source model's probability of the gold class
    tp_key = "%s_pred_prob" % target      # target model's predicted-class probability
    tg_key = "%s_gold_prob" % target      # target model's probability of the gold class

    def rec(i):
        # ``i`` is a position in the test arrays; ``te[i]`` is the CSV row it came from.
        row = int(te[i])
        assert texts[row] is not None
        return {"index": row, "test_pos": int(i), "text": _clean_snippet(texts[row]),
                t_key: _clean_snippet(texts[row]),
                s_key: _clean_snippet(gold_texts[row]),
                mt_key: None,
                "gold": int(gold[i]), "gold_label": cls[int(gold[i])],
                "source_pred": int(src_pred[i]), "source_pred_label": cls[int(src_pred[i])],
                "target_pred": int(tgt_pred[i]), "target_pred_label": cls[int(tgt_pred[i])],
                sp_key: round(float(src_proba[i, src_pred[i]]), 6),
                sg_key: round(float(src_proba[i, gold[i]]), 6),
                tp_key: round(float(tgt_proba[i, tgt_pred[i]]), 6),
                tg_key: round(float(tgt_proba[i, gold[i]]), 6)}

    def collect(mask):
        out = []
        for i in np.where(mask)[0]:
            i = int(i)
            out.append((rec(i), str(texts[int(te[i])])))
        return out

    bw_pairs = collect(both_wrong)
    tg_pairs = collect(transfer_gap)
    tow_total = int(target_only_wrong.sum())

    # Machine-translate the listed TARGET texts into the SOURCE language (opus-mt).
    listed = ([full for _, full in bw_pairs[:MAX_SHOW]]
              + [full for _, full in tg_pairs[:MAX_SHOW]])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("  [MT] device=%s; %d listed strings to translate (%s->%s)"
        % (device, len(set(listed)), target, source))
    mt_map = translate_cached(listed, target, source, device) if listed else {}
    mt_available = bool(listed) and any(mt_map.get(f) for f in listed)

    def attach(pairs):
        out = []
        for rec_, full in pairs[:MAX_SHOW]:
            tr = mt_map.get(full)
            rec_[mt_key] = _clean_snippet(tr) if tr else None
            out.append(rec_)
        return out

    bw = attach(bw_pairs)
    tg = attach(tg_pairs)

    M = confusion(gold, src_pred, C)
    error_pairs = []
    for g in range(C):
        for p in range(C):
            if g != p and M[g, p] > 0:
                error_pairs.append({"gold": g, "gold_label": cls[g],
                                    "source_pred": p, "source_pred_label": cls[p],
                                    "count": int(M[g, p])})
    error_pairs.sort(key=lambda r: (-r["count"], r["gold"], r["source_pred"]))

    return {
        "name": name, "n_classes": C, "classes": cls,
        "source": source, "target": target, "train_on": train_on,
        "sizes": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
                  "fit": int(len(fit_rows))},
        "feature": "perlang_z [both] (L2-norm + per-language FIT z-score)",
        "xgb": params,
        "trained_on": "%s split only (%s model: %s; %s model: %s)"
                      % (train_on, SRC_UP, source, TGT_UP, target),
        "evaluated_on": "%s test rows" % TGT_UP,
        "bucket_counts": {
            "both_wrong": int(both_wrong.sum()),
            "transfer_gap": int(transfer_gap.sum()),
            "target_only_wrong": int(target_only_wrong.sum()),
            "both_right": int(both_right.sum()),
        },
        "source_accuracy": src_acc,
        "target_accuracy": tgt_acc,
        "source_errors_total": int(src_wrong.sum()),
        "target_errors_total": int(tgt_wrong.sum()),
        "confusion_source_rows_gold_cols_pred": M.tolist(),
        "source_error_pairs": error_pairs,
        "target_text_key": t_key, "source_gold_key": s_key, "mt_key": mt_key,
        "source_pred_prob_key": sp_key, "source_gold_prob_key": sg_key,
        "target_pred_prob_key": tp_key, "target_gold_prob_key": tg_key,
        "mt_model": mt_model_id(target, source), "mt_available": mt_available,
        "both_wrong_total": len(bw_pairs),
        "transfer_gap_total": len(tg_pairs),
        "target_only_wrong_total": tow_total,
        "both_wrong_listed": len(bw),
        "transfer_gap_listed": len(tg),
        "both_wrong": bw,
        "transfer_gap": tg,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def md_dataset(res, max_show):
    L = []
    C = res["n_classes"]
    cls = res["classes"]
    source, target = res["source"], res["target"]
    SRC_UP, TGT_UP = source.upper(), target.upper()
    L.append("\n## %s (%d classes) | %s -> %s\n" % (res["name"], C, source, target))
    L.append("- Splits: train %d / val %d / test %d (fit %d = train-on `%s`; "
             "%s test = %d rows)."
             % (res["sizes"]["train"], res["sizes"]["val"], res["sizes"]["test"],
                res["sizes"]["fit"], res["train_on"], TGT_UP, res["sizes"]["test"]))
    L.append("- Features: `%s`; XGBoost `%s`; %s. Evaluated on %s."
             % (res["feature"], "softprob depth8/mcw5/lr0.1/n400",
                res["trained_on"], res["evaluated_on"] + "."))
    L.append("- Classes: %s.\n" % ", ".join("`%s`" % c for c in cls))

    b = res["bucket_counts"]
    L.append("**Buckets (%s test):** both_wrong (%s wrong & %s wrong) **%d**, "
             "transfer_gap (%s wrong & %s right) **%d**, target_only_wrong (%s right & %s wrong) "
             "**%d**, both_right **%d**. (total %d)\n"
             % (TGT_UP, SRC_UP, TGT_UP, b["both_wrong"], SRC_UP, TGT_UP,
                b["transfer_gap"], SRC_UP, TGT_UP, b["target_only_wrong"], b["both_right"],
                sum(b.values())))
    L.append("**Accuracy on %s test:** %s-trained **%.4f** (%d errors) | "
             "%s-trained **%.4f** (%d errors).\n"
             % (TGT_UP, SRC_UP, res["source_accuracy"], res["source_errors_total"],
                TGT_UP, res["target_accuracy"], res["target_errors_total"]))

    L.append("**Full confusion of %s-trained errors on %s test** "
             "(rows = gold, cols = %s pred; diagonal omitted):\n"
             % (SRC_UP, TGT_UP, SRC_UP))
    L.append("| gold \\ %s pred | count |" % SRC_UP)
    L.append("|---|---:|")
    for r in res["source_error_pairs"]:
        L.append("| `%s` -> `%s` | %d |"
                 % (r["gold_label"], r["source_pred_label"], r["count"]))

    t_key, s_key, mt_key = res["target_text_key"], res["source_gold_key"], res["mt_key"]
    sp_key, sg_key = res["source_pred_prob_key"], res["source_gold_prob_key"]
    tp_key, tg_key = res["target_pred_prob_key"], res["target_gold_prob_key"]

    def _table(items):
        rows = []
        rows.append("| CSV row | %s (`%s`) | %s (`%s`) | %s (`%s`->`%s`) | gold | %s pred | %s pred "
                    "| %s | %s | %s | %s |"
                    % (t_key, target, s_key, source, mt_key, target, source, SRC_UP, TGT_UP,
                       sp_key, sg_key, tp_key, tg_key))
        rows.append("|---:|---|---|---|---|---|---|---|---|---|---|")
        for r in items:
            mt = r.get(mt_key)
            rows.append("| %d | %s | %s | %s | %s | %s | %s | %.4f | %.4f | %.4f | %.4f |"
                        % (r["index"], r[t_key], r[s_key], "" if mt is None else mt,
                           r["gold_label"], r["source_pred_label"], r["target_pred_label"],
                           r[sp_key], r[sg_key], r[tp_key], r[tg_key]))
        return rows

    L.append("\n### transfer_gap (%s wrong & %s right) — %d total, showing %d\n"
             % (SRC_UP, TGT_UP, res["transfer_gap_total"], res["transfer_gap_listed"]))
    if res["transfer_gap"]:
        L.extend(_table(res["transfer_gap"]))
    else:
        L.append("_none_")

    L.append("\n### both_wrong (hard: %s wrong & %s wrong) — %d total, showing %d\n"
             % (SRC_UP, TGT_UP, res["both_wrong_total"], res["both_wrong_listed"]))
    if res["both_wrong"]:
        L.extend(_table(res["both_wrong"]))
    else:
        L.append("_none_")
    return L


def build_md(res, max_show):
    source, target = res["source"], res["target"]
    SRC_UP, TGT_UP = source.upper(), target.upper()
    md = []
    md.append("# FLORES %s-test error analysis: %s-trained vs %s-trained model "
              "(EuroBERT-210m mean + `perlang_z [both]`)\n"
              % (TGT_UP, SRC_UP, TGT_UP))
    md.append("Two XGBoost models, both evaluated on the **%s test rows**: **%s-trained** "
              "(fit on %s) and **%s-trained** (fit on %s). Features = `perlang_z [both]` "
              "(L2-normalize + per-language OWN fit z-score, applied to both sides). Data via "
              "`train_flores12` (`prepare_sentence` + "
              "`load_sent_embeddings('mean', kept, 2009)`; `prepare_agg` + "
              "`load_agg_embeddings('mean', 562)`). XGBoost = `flores_calib.XGB_COMMON` + "
              "`num_class` (softprob, hist, max_bin=64, cpu, n_jobs=2, seed=0, depth=8, "
              "min_child_weight=5, lr=0.1, n=400). Both models train on the `%s` split; "
              "evaluated on the **%s test** rows.\n"
              % (TGT_UP, SRC_UP, source, TGT_UP, target, res["train_on"], TGT_UP))
    md.append("Buckets: **both_wrong** = %s wrong AND %s wrong (\"hard\"); **transfer_gap** = "
              "%s wrong AND %s right; **target_only_wrong** = %s right AND %s wrong; "
              "**both_right**. Each listed item shows three text forms: `%s_text` = FLORES "
              "`%s` (target); `%s_gold` = FLORES `%s` (aligned gold source); `%s2%s_mt` = "
              "`Helsinki-NLP/opus-mt-%s-%s` translation of the target text (fp16 greedy, "
              "cached).\n"
              % (SRC_UP, TGT_UP, SRC_UP, TGT_UP, SRC_UP, TGT_UP, target, target,
                 source, source, target, source, target, source))
    md.extend(md_dataset(res, max_show))
    md.append("")
    return md


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="List source->target transfer errors on FLORES test text.")
    ap.add_argument("--dataset", choices=DATASETS, default="sent",
                    help="sentence 12-class (sent) or URL-aggregated 11-class (agg).")
    ap.add_argument("--source", choices=LANGS, default="en",
                    help="language the SOURCE model is trained on.")
    ap.add_argument("--target", choices=LANGS, default="it",
                    help="language of the test text (and TARGET model).")
    ap.add_argument("--show", type=int, default=30,
                    help="max items to list per bucket (default 30).")
    ap.add_argument("--train-on", choices=["train", "trainval"], default="train",
                    help="fit both models on train only, or train+val (default train).")
    ap.add_argument("--no-calib", action="store_true",
                    help="Score raw model argmax (uncalibrated; this is the default and the "
                         "reproduction path).")
    args = ap.parse_args(argv)
    if args.source == args.target:
        ap.error("--source and --target must differ (got %s==%s)" % (args.source, args.target))
    return args


def main(argv=None):
    try:
        from gpu_limit import limit_gpu; limit_gpu()
    except Exception:
        pass
    args = parse_args(argv)
    global MAX_SHOW
    MAX_SHOW = max(0, int(args.show))

    D = build(args.dataset, args.source, args.target)
    res = run_dataset(D, args.source, args.target, args.train_on)

    payload = {
        "config": {
            "dataset": args.dataset,
            "source": args.source, "target": args.target,
            "train_on": args.train_on,
            "emb": "EuroBERT-210m mean",
            "feature": "perlang_z [both]",
            "calibrated": False,
            "xgb": FC.XGB_COMMON,
            "seed": SEED,
            "max_show": MAX_SHOW, "snippet_chars": SNIPPET,
            "mt_model": mt_model_id(args.target, args.source),
            "mt_cache": os.path.basename(mt_cache_path(args.target, args.source)),
            "mt_available": res["mt_available"],
            "python": sys.version.split()[0],
            "xgboost": xgb.__version__,
            "torch": torch.__version__,
        },
        "result": res,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_jd)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(build_md(res, MAX_SHOW)) + "\n")
    log("wrote %s , %s" % (OUT_JSON, OUT_MD))

    b = res["bucket_counts"]
    log("%s %s->%s: %s acc=%.4f %s acc=%.4f | both_wrong=%d transfer_gap=%d "
        "target_only_wrong=%d both_right=%d"
        % (res["name"], args.source, args.target, args.source.upper(), res["source_accuracy"],
           args.target.upper(), res["target_accuracy"], b["both_wrong"], b["transfer_gap"],
           b["target_only_wrong"], b["both_right"]))
    return 0


MAX_SHOW = 30


if __name__ == "__main__":
    sys.exit(main())
