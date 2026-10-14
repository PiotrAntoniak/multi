"""all_paraphrases.py -- train the downstream classifier on the UNION of ALL generators'
synthetic Italian paraphrases (pooled), and compare against each single generator, the
real-Italian "ceiling", and the English-transfer "floor".

Idea
----
``gen_compare.py`` trains the SAME classifier once per generator.  Here we instead POOL the
synthetic Italian of every available generator into ONE training set and ask whether the
union beats the best single generator and how close it gets to the real-Italian ceiling.

Pipeline (matches ``gen_compare.py`` exactly)
---------------------------------------------
* Synthetic Italian per generator is read from the ``gen_compare`` cache
  (``%TEMP%\\opencode\\gen_compare\\gen_<gen>_<dataset>_<n>_<hash>.json``); nothing is generated.
* EuroBERT-**210m** **mean** embeddings (no per-language z-scoring -- gen_compare uses none).
  FLORES: fp32 forward, plain tokenization truncated to 511.  Toxic: bf16 forward, max_len 512.
* XGBoost, hist / max_bin 64 / cpu / n_jobs 2 / seed 0 / depth 8 / mcw 5 / lr 0.1 / n 400;
  FLORES ``multi:softprob`` (12-class), toxic ``binary:logistic``.
* Evaluate on the **real Italian** test.

Union features (no GPU needed)
------------------------------
EuroBERT mean pooling is **per-text**, so the union's feature matrix is exactly the vertical
stack of the per-generator mean embeddings (with the same row labels repeated).  We therefore
build ``X_union = vstack(emb_g for g in generators)`` from the cached per-generator matrices,
which is identical to embedding the pooled text -- and needs no embedder at all.  A GPU is
used only if a per-generator synthetic embedding cache is missing (toxic, when those bytes
land) and the embedder must actually run.

Outputs (create-only): ``all_paraphrases.json`` / ``.md`` / ``.png``.

Usage
-----
    python all_paraphrases.py                    # FLORES now (+ toxic if all gens cached)
    python all_paraphrases.py --datasets flores
    python all_paraphrases.py --from-json        # rebuild .md/.png from all_paraphrases.json
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import glob
import json
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import gen_compare as GC            # reuse the exact pipeline (cache, embed, XGB, metrics)
import train_flores12 as T          # FLORES sentence splits

OUT_JSON = os.path.join(REPO, "all_paraphrases.json")
OUT_MD = os.path.join(REPO, "all_paraphrases.md")
OUT_PNG = os.path.join(REPO, "all_paraphrases.png")

BB = "210m"                         # EuroBERT-210m mean (primary backbone in gen_compare)
GENERATORS = list(GC.GENERATORS)
GEN_SHORT = dict(GC.GEN_SHORT)
COMBINED = "all_combined"
COMBINED_LABEL = "ALL generators"

_T0 = time.time()
_SMOKE = False


def log(msg):
    print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


# ------------------------------------------------------------------ data structures
def flores_struct():
    """Rebuild the exact FLORES structure gen_compare uses (EN train+val 997+504=1501)."""
    sent = T.prepare_sentence()
    y = np.asarray(sent["y"], dtype=int)
    tr, va, te = sent["train_idx"], sent["val_idx"], sent["test_idx"]
    trva = np.concatenate([tr, va])
    cur = sent["cur"]
    en_src = cur.iloc[trva]["en"].astype(str).tolist()
    it_ref = cur.iloc[trva]["it"].astype(str).tolist()
    return {"y": y, "trva": trva, "te": te, "en_src": en_src, "it_ref": it_ref,
            "sent": sent, "n_train": int(len(tr)), "n_val": int(len(va)),
            "n_test": int(len(te)), "classes": sent["classes"]}


def toxic_struct(en_cap, seed):
    """Rebuild the exact toxic structure gen_compare uses (EN train capped to en_cap, seed)."""
    import pandas as pd
    DATA_DIR = GC.DATA_DIR

    def read_split(lang, split):
        df = pd.read_parquet(os.path.join(DATA_DIR, "%s_%s.parquet" % (lang, split)))
        return df["text"].astype(str).tolist(), df["is_toxic"].astype(int).to_numpy()

    en_tr_text, y_en_all = read_split("en", "train")
    _, y_it_tr = read_split("it", "train")
    _, y_it_te = read_split("it", "test")
    n_en = len(y_en_all)
    cap = min(en_cap, n_en)
    rng = np.random.default_rng(seed)
    rows_used = np.sort(rng.choice(n_en, size=cap, replace=False))
    en_use = [en_tr_text[i] for i in rows_used]
    return {"en_src": en_use, "rows_used": rows_used, "y_en_all": y_en_all,
            "y_it_tr": y_it_tr, "y_it_te": y_it_te, "n_en": n_en,
            "en_cap": int(len(rows_used)), "n_it_train": int(len(y_it_tr)),
            "n_it_test": int(len(y_it_te))}


# ------------------------------------------------------------------ cache / embeddings
def gen_cache_path(generator, dataset, texts):
    """Exact path gen_compare.generate_italian would use for these (generator, dataset)."""
    return os.path.join(GC.CACHE, "gen_%s_%s_%d_%s.json" % (
        generator.replace("/", "__"), dataset, len(texts),
        GC._hash([generator, dataset] + list(texts))))


def load_gen(generator, dataset, texts):
    p = gen_cache_path(generator, dataset, texts)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def synth_emb(texts, ds):
    """Per-generator synthetic-Italian mean embedding (cache hit -> no GPU / no model load)."""
    model_id, dtype, max_len = GC.BACKBONES[BB]["model"], None, None
    if ds == "flores":
        dtype, max_len = "float32", GC.FLORES_EMB_MAX_LEN
    else:
        dtype, max_len = "bf16", GC.TOXIC_EMB_MAX_LEN
    return GC.ensure_synth_emb(texts, BB, ds, model_id, dtype, max_len)


def collect_generations(ds, en_src, generators):
    """Return {gen: [texts]} for the generators whose cache is present, else None entries."""
    out = {}
    for g in generators:
        cached = load_gen(g, ds, en_src)
        if cached is not None and len(cached) != len(en_src):
            log("  WARN %s/%s cache length %d != %d; ignoring"
                % (GEN_SHORT.get(g, g), ds, len(cached), len(en_src)))
            cached = None
        out[g] = cached
    return out


# ------------------------------------------------------------------ evaluation (one dataset)
def run_flores(data_ds, generators):
    f = GC.flores_data_for(BB, data_ds)
    y_tr = f["y_tr"]
    gens = collect_generations("flores", data_ds["en_src"], generators)
    available = [g for g in generators if gens[g] is not None]
    missing = [g for g in generators if gens[g] is None]
    if missing:
        log("  FLORES: generators with NO cached generations (skipped): %s"
            % ", ".join(GEN_SHORT.get(g, g) for g in missing))

    arms = {"order": [], "gaps": [GEN_SHORT.get(g, g) for g in missing]}
    arms["real_it"] = GC._eval_flores(BB, f["X_it_tr"], y_tr, f["X_it_te"], f["y_te"])
    arms["real_it"]["n_train"] = int(len(y_tr))
    arms["en_transfer"] = GC._eval_flores(BB, f["X_en_tr"], y_tr, f["X_it_te"], f["y_te"])
    arms["en_transfer"]["n_train"] = int(len(y_tr))

    embs = {}
    for g in available:
        log("  embedding/cached synthetic IT: %s (flores)" % GEN_SHORT.get(g, g))
        X = synth_emb(gens[g], "flores")
        embs[g] = X
        m = GC._eval_flores(BB, X, y_tr, f["X_it_te"], f["y_te"])
        m["n_train"] = int(len(y_tr))
        arms[g] = m
        log("    %-14s acc=%.4f macro_f1=%.4f auc=%.4f"
            % (GEN_SHORT.get(g, g), m["accuracy"], m["macro_f1"], m["auc"]))

    # ---- UNION: vstack of per-generator mean embeddings (= embedding the pooled text) ----
    if len(available) >= 1:
        Xcomb = np.vstack([embs[g] for g in available])
        ycomb = np.concatenate([y_tr for _ in available])
        log("  UNION (%s): X=%s y=%s (train)"
            % ("+".join(GEN_SHORT.get(g, g) for g in available), Xcomb.shape, ycomb.shape))
        mc = GC._eval_flores(BB, Xcomb, ycomb, f["X_it_te"], f["y_te"])
        mc["n_train"] = int(len(ycomb))
        mc["generators"] = list(available)
        arms[COMBINED] = mc
        log("    %-14s acc=%.4f macro_f1=%.4f auc=%.4f"
            % (COMBINED_LABEL, mc["accuracy"], mc["macro_f1"], mc["auc"]))
    arms["order"] = list(available) + [COMBINED]
    return arms


def run_toxic(data_ds, generators):
    f = GC.toxic_data_for(BB, data_ds)
    y_en = f["y_en_tr"]
    gens = collect_generations("toxic", data_ds["en_src"], generators)
    available = [g for g in generators if gens[g] is not None]
    missing = [g for g in generators if gens[g] is None]
    if missing:
        log("  TOXIC: generators with NO cached generations (skipped): %s"
            % ", ".join(GEN_SHORT.get(g, g) for g in missing))

    arms = {"order": [], "gaps": [GEN_SHORT.get(g, g) for g in missing]}
    arms["real_it"] = GC._eval_toxic(f["X_it_tr"], f["y_it_tr"], f["X_it_te"], f["y_it_te"])
    arms["real_it"]["n_train"] = int(len(f["y_it_tr"]))
    arms["en_transfer"] = GC._eval_toxic(f["X_en_tr"], f["y_en_tr"], f["X_it_te"], f["y_it_te"])
    arms["en_transfer"]["n_train"] = int(len(f["y_en_tr"]))

    embs = {}
    for g in available:
        log("  embedding/cached synthetic IT: %s (toxic)" % GEN_SHORT.get(g, g))
        X = synth_emb(gens[g], "toxic")
        embs[g] = X
        m = GC._eval_toxic(X, y_en, f["X_it_te"], f["y_it_te"])
        m["n_train"] = int(len(y_en))
        arms[g] = m
        log("    %-14s acc=%.4f f1=%.4f auc=%.4f"
            % (GEN_SHORT.get(g, g), m["accuracy"], m["f1"], m["auc"]))

    if len(available) >= 1:
        Xcomb = np.vstack([embs[g] for g in available])
        ycomb = np.concatenate([y_en for _ in available])
        log("  UNION (%s): X=%s y=%s (train)"
            % ("+".join(GEN_SHORT.get(g, g) for g in available), Xcomb.shape, ycomb.shape))
        mc = GC._eval_toxic(Xcomb, ycomb, f["X_it_te"], f["y_it_te"])
        mc["n_train"] = int(len(ycomb))
        mc["generators"] = list(available)
        arms[COMBINED] = mc
        log("    %-14s acc=%.4f f1=%.4f auc=%.4f"
            % (COMBINED_LABEL, mc["accuracy"], mc["f1"], mc["auc"]))
    arms["order"] = list(available) + [COMBINED]
    return arms


# ------------------------------------------------------------------ verdict
def verdict_for(ds, arms):
    pk = "macro_f1" if ds == "flores" else "f1"
    plabel = "macro-F1" if ds == "flores" else "F1"
    singles = {g: arms[g][pk] for g in arms["order"] if g != COMBINED and g in arms}
    best_g = max(singles, key=singles.get) if singles else None
    best_v = singles.get(best_g, float("nan"))
    comb = arms[COMBINED][pk] if COMBINED in arms else float("nan")
    ceil = arms["real_it"][pk]
    floor = arms["en_transfer"][pk]
    beats = comb > best_v + 1e-9
    pct_ceil = (100.0 * comb / ceil) if ceil else float("nan")
    txt = ("UNION of %d generator(s) reaches %.4f %s vs best single **%s** %.4f "
           "(%s) and real-IT ceiling %.4f (%.0f%% of ceiling, %+.4f vs ceiling); EN-transfer "
           "floor %.4f."
           % (len(arms.get(COMBINED, {}).get("generators", [])), comb, plabel,
              GEN_SHORT.get(best_g, best_g), best_v,
              "UNION WINS" if beats else "single wins", ceil, pct_ceil,
              (comb - ceil), floor))
    if arms.get("gaps"):
        txt += " (Missing generators: %s.)" % ", ".join(arms["gaps"])
    return {"text": txt, "beats_best_single": bool(beats), "best_single": best_g,
            "best_single_value": float(best_v), "combined_value": float(comb),
            "ceiling": float(ceil), "floor": float(floor), "pct_of_ceiling": float(pct_ceil)}


# ------------------------------------------------------------------ report / plot
def _jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def arm_label(a):
    if a == COMBINED:
        return COMBINED_LABEL + " [union]"
    if a == "real_it":
        return "real_it [ceiling]"
    if a == "en_transfer":
        return "en_transfer [floor]"
    return GEN_SHORT.get(a, a)


def build_md(payload):
    cfg = payload["config"]
    L = []
    L.append("# All-paraphrases: union of every generator's synthetic Italian")
    L.append("")
    L.append("We pool the synthetic Italian of **all available generators** into ONE training "
             "set and train the SAME fixed classifier (EuroBERT-**%s** mean -> XGBoost) used by "
             "`gen_compare.py`, then compare against each single generator, the **real Italian "
             "ceiling** and the **EN-transfer floor**, all evaluated on the **real Italian test**."
             % BB)
    L.append("")
    L.append("- Generators: %s." % ", ".join("`%s`" % g for g in payload["generators"]))
    L.append("- Union features = vertical stack of the per-generator mean embeddings "
             "(EuroBERT mean pooling is per-text, so this equals embedding the pooled text).")
    L.append("- XGBoost (CPU, fixed): hist, max_bin 64, cpu, n_jobs 2, seed 0, depth 8, mcw 5, "
             "lr 0.1, n 400 (FLORES `multi:softprob`; toxic `binary:logistic`).")
    L.append("- Generation: read from the `gen_compare` cache; **no generation was run**.")
    L.append("")

    for ds in payload["completed_datasets"]:
        arms = payload["results"][ds]
        pk = "macro_f1" if ds == "flores" else "f1"
        plabel = "macro-F1" if ds == "flores" else "F1"
        L.append("## Dataset: %s (real Italian test)" % ds)
        L.append("")
        L.append("| arm | accuracy | %s | AUC | n_train |" % plabel)
        L.append("|---|---|---|---|---|")
        if COMBINED in arms:
            m = arms[COMBINED]
            L.append("| **%s** | %.4f | **%.4f** | %.4f | %d |"
                     % (COMBINED_LABEL, m["accuracy"], m[pk], m["auc"], m["n_train"]))
        for g in arms["order"]:
            if g == COMBINED:
                continue
            m = arms[g]
            L.append("| %s | %.4f | %.4f | %.4f | %d |"
                     % (GEN_SHORT.get(g, g), m["accuracy"], m[pk], m["auc"], m["n_train"]))
        for b in ("real_it", "en_transfer"):
            m = arms[b]
            L.append("| %s (%s) | %.4f | %.4f | %.4f | %d |"
                     % (b, "ceiling" if b == "real_it" else "floor",
                        m["accuracy"], m[pk], m["auc"], m["n_train"]))
        L.append("")
        v = payload["verdicts"][ds]
        L.append("- **Verdict:** %s" % v["text"])
        L.append("- Combining all generators **%s** the best single generator."
                 % ("beats" if v["beats_best_single"] else "does NOT beat"))
        L.append("")

    if payload.get("pending"):
        L.append("## Pending")
        L.append("")
        for ds, why in payload["pending"].items():
            L.append("- **%s**: %s" % (ds, why))
        L.append("")

    L.append("![all_paraphrases](all_paraphrases.png)")
    L.append("")
    return "\n".join(L)


def plot(payload, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dss = [d for d in payload["completed_datasets"] if COMBINED in payload["results"][d]]
    if not dss:
        dss = payload["completed_datasets"]
    fig, axes = plt.subplots(1, len(dss), figsize=(8.4 * len(dss), 5.4), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        arms = payload["results"][ds]
        order = list(arms["order"])
        labels = [arm_label(a) for a in order]
        if ds == "flores":
            metrics = [("accuracy", "accuracy", "#4c72b0"),
                       ("macro_f1", "macro-F1", "#dd8452"),
                       ("auc", "AUC", "#55a868")]
        else:
            metrics = [("accuracy", "accuracy", "#4c72b0"),
                       ("f1", "F1 (pos)", "#dd8452"),
                       ("auc", "AUC", "#55a868")]
        x = np.arange(len(order))
        w = 0.25
        colors = ["#c44e52" if a == COMBINED else ("#8172b3" if a in ("real_it", "en_transfer")
                  else "#4c72b0") for a in order]
        for j, (mk, lab, col) in enumerate(metrics):
            vals = [arms[a][mk] for a in order]
            bars = ax.bar(x + (j - 1) * w, vals, w, label=lab, color=col)
            for r in bars:
                ax.annotate("%.3f" % r.get_height(),
                            (r.get_x() + r.get_width() / 2, r.get_height()),
                            ha="center", va="bottom", fontsize=6)
        for mk, lab, col in metrics:
            ax.axhline(arms["real_it"][mk], color=col, linestyle="--", linewidth=1.0, alpha=0.8)
            ax.axhline(arms["en_transfer"][mk], color=col, linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("score")
        ax.set_title("%s (EuroBERT-%s mean, real IT test)\ndashed = real-IT ceiling, "
                     "dotted = EN-transfer floor" % (ds, BB), fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Train on the UNION of all generators' synthetic Italian vs single "
                 "generators vs real-Italian ceiling", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    log("wrote %s" % out_png)


def write_outputs(payload):
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_jd, ensure_ascii=False)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(build_md(payload))
    plot(payload, OUT_PNG)
    log("wrote %s, %s, %s" % (os.path.basename(OUT_JSON), os.path.basename(OUT_MD),
                              os.path.basename(OUT_PNG)))


def toxic_ready(generators):
    """Which generators already have a cached toxic-8000 generation file."""
    present = set()
    for f in glob.glob(os.path.join(GC.CACHE, "gen_*_toxic_8000_*.json")):
        name = os.path.basename(f)
        for g in generators:
            if name.startswith("gen_" + g.replace("/", "__") + "_toxic_8000_"):
                present.add(g)
    return present


# ------------------------------------------------------------------ main
def main(argv=None):
    global _SMOKE, BB
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="Union-of-all-generators synthetic-IT comparison.")
    ap.add_argument("--generators", default=",".join(GENERATORS))
    ap.add_argument("--datasets", default="flores,toxic",
                    help="comma list from {flores,toxic}; toxic is auto-skipped to 'pending' "
                         "unless every generator's toxic cache is present")
    ap.add_argument("--backbone", default=BB, choices=list(GC.BACKBONES))
    ap.add_argument("--en-cap", type=int, default=GC.DEFAULT_EN_CAP)
    ap.add_argument("--seed", type=int, default=GC.DEFAULT_SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--from-json", action="store_true")
    args = ap.parse_args(argv)

    BB = args.backbone

    if args.from_json:
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            payload = json.load(f)
        with open(OUT_MD, "w", encoding="utf-8") as f:
            f.write(build_md(payload))
        plot(payload, OUT_PNG)
        print("rebuilt %s and %s from %s"
              % (os.path.basename(OUT_MD), os.path.basename(OUT_PNG), os.path.basename(OUT_JSON)))
        return 0

    _SMOKE = args.smoke
    GC._SMOKE = args.smoke
    generators = [g.strip() for g in args.generators.split(",") if g.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # CPU-only XGB by construction; device is irrelevant unless a synthetic-embed cache misses,
    # in which case gen_compare's embedder uses cuda when available.
    import torch
    GC._DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    log("device=%s backbone=%s generators=%s datasets=%s"
        % (GC._DEVICE, BB, generators, datasets))

    payload = {
        "config": {"generators": generators, "datasets": datasets, "backbone": BB,
                   "xgb": dict(GC.XGB_PARAMS), "src_lang": GC.SRC_LANG, "tgt_lang": GC.TGT_LANG,
                   "embed": "EuroBERT-%s mean (per-text)" % BB,
                   "flores_emb": {"dtype": "float32", "max_len": GC.FLORES_EMB_MAX_LEN},
                   "toxic_emb": {"dtype": "bf16", "max_len": GC.TOXIC_EMB_MAX_LEN},
                   "seed": int(args.seed), "toxic_en_cap": int(args.en_cap),
                   "python": sys.version.split()[0]},
        "generators": generators, "results": {}, "verdicts": {}, "completed_datasets": [],
        "pending": {},
    }

    if "flores" in datasets:
        log("######## FLORES ########")
        struct = flores_struct()
        payload["config"]["flores_n_test"] = struct["n_test"]
        arms = run_flores(struct, generators)
        payload["results"]["flores"] = arms
        payload["verdicts"]["flores"] = verdict_for("flores", arms)
        payload["completed_datasets"].append("flores")
        write_outputs(payload)

    if "toxic" in datasets:
        log("######## TOXIC ########")
        ready = toxic_ready(generators)
        available = [g for g in generators if g in ready]
        if not available:
            payload["pending"]["toxic"] = (
                "not run: no toxic generations cached for {%s}."
                % ", ".join(GC.GEN_SHORT.get(g, g) for g in generators))
            log("TOXIC pending: %s" % payload["pending"]["toxic"])
        else:
            missing = sorted(set(generators) - ready)
            if missing:
                log("TOXIC: running with the %d available generator(s) %s; SKIPPING %s"
                    % (len(available),
                       ", ".join(GC.GEN_SHORT.get(g, g) for g in available),
                       ", ".join(GC.GEN_SHORT.get(g, g) for g in missing)))
            struct = toxic_struct(args.en_cap, args.seed)
            payload["config"]["toxic_n_test"] = struct["n_it_test"]
            arms = run_toxic(struct, generators)
            payload["results"]["toxic"] = arms
            payload["verdicts"]["toxic"] = verdict_for("toxic", arms)
            payload["completed_datasets"].append("toxic")
            write_outputs(payload)

    write_outputs(payload)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
