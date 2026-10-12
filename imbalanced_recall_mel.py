"""imbalanced_recall_mel.py -- MultiEURLEX (`mel`) arm of the high-recall EN-vs-native comparison.

Question.  On HIGHLY UNBALANCED detection of RARE EuroVoc level-1 concepts in LONG, non-English
EU legal documents, how much worse is an ENGLISH-trained detector than one trained natively in the
target language (de / fr), when we explicitly TOLERATE many false alarms (recall-oriented)?

Recipe (identical to imbalanced_recall.py; mean pooling only, no chunking):
  * representation : EuroBERT-210m MEAN pool, 512-token truncation, batch 64 (ir.ensure_mel_embeddings).
  * transform      : ``l2_thenZ [both]`` -- L2-normalize then per-language TRAIN z-score (each language
                     uses its own train split stats).  (ir.l2_thenZ_both)
  * model          : one-vs-rest XGBClassifier(binary:logistic, hist, max_bin=64, device=cpu, n_jobs=2,
                     seed=0, max_depth=8, min_child_weight=5, lr=0.1, n_estimators=400).  (ir.run_cell)
  * English-trained: fit on the ENGLISH train split (positives = concept c), score the TARGET test.
  * native-trained : fit on the TARGET-language train split (positives = concept c), score TARGET test.

This driver only reuses the helpers in ``imbalanced_recall.py``.  It does NOT touch or clobber the
sent/agg outputs (``imbalanced_recall.json`` / ``.md`` / ``imbalanced_pr_sent|agg.png``).  All new
files are create-only; nothing is ever deleted or moved.

New outputs (repo root; v2 keeps the buggy v1 files untouched):
    imbalanced_recall_mel_v2.json        full metrics payload
    imbalanced_recall_mel_v2.md          per-concept EN-vs-native recall tables + headline
    imbalanced_pr_mel_v2.png             PR curves (rare concepts x de/fr)
    imbalanced_recall_mel_v2.pid         pid file
    imbalanced_recall_mel_v2_status.json progressive status
    %TEMP%\\opencode\\imbalanced_recall_mel_v2.log   (when launched detached)

v2 label fix.  MultiEURLEX TRAIN splits are NOT parallel across languages (an EN train document
is not the DE/FR one), so the v1 ``load_mel`` training the native de/fr models with the ENGLISH
labels produced a near-random native column.  v2 loads each language's OWN train labels; the TEST
split is parallel, so its labels are shared across languages.

Usage:
    python imbalanced_recall_mel.py --n-rare 8 --device auto
    python imbalanced_recall_mel.py --launch --n-rare 8 --device auto
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import subprocess
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

import imbalanced_recall as ir

REPO = ir.REPO
# This arm is en -> {de, fr}: patch the module globals so the reused helpers operate on this scope.
ir.LANGS = ["en", "de", "fr"]
ir.TARGETS = ["de", "fr"]

# v2: fixes the v1 label bug (v1 trained the native de/fr models with ENGLISH labels even though
# MultiEURLEX TRAIN splits are not parallel).  Outputs are versioned so the buggy v1 files survive.
VERSION = "v2"
OUT_JSON = os.path.join(REPO, "imbalanced_recall_mel_%s.json" % VERSION)
OUT_MD = os.path.join(REPO, "imbalanced_recall_mel_%s.md" % VERSION)
OUT_PNG = os.path.join(REPO, "imbalanced_pr_mel_%s.png" % VERSION)
PID_PATH = os.path.join(REPO, "imbalanced_recall_mel_%s.pid" % VERSION)
STATUS_PATH = os.path.join(REPO, "imbalanced_recall_mel_%s_status.json" % VERSION)
LOG_NAME = "imbalanced_recall_mel_%s.log" % VERSION

# MultiEURLEX level-1 EuroVoc concept ids, in ClassLabel order (config nlpaueb/multi_eurlex).
LEVEL1_IDS = ["100149", "100160", "100148", "100147", "100152", "100143", "100156", "100158",
              "100154", "100153", "100142", "100145", "100150", "100162", "100159", "100144",
              "100151", "100157", "100161", "100146", "100155"]
# English descriptors (data/eurovoc_descriptors.json, nlpaueb/multi-eurlex).
LEVEL1_EN = ["social questions", "industry", "finance", "trade", "business and competition",
             "international relations", "agriculture, forestry and fisheries",
             "production, technology and research", "transport", "employment and working conditions",
             "politics", "law", "education and communications", "international organisations",
             "energy", "european union", "science", "agri-foodstuffs", "geography", "economics",
             "environment"]

FPR_GRID = [0.01, 0.05, 0.10]
FA_GRID = [5, 20, 50, 100]
HL_KEYS = ["recall_at_fpr_10", "recall_at_fa_50", "recall_at_fa_100"]
METRIC_ORDER = [
    ("average_precision", "AP"),
    ("recall_at_fpr_1", "R@FPR1%"),
    ("recall_at_fpr_5", "R@FPR5%"),
    ("recall_at_fpr_10", "R@FPR10%"),
    ("recall_at_fa_5", "R@FA5"),
    ("recall_at_fa_20", "R@FA20"),
    ("recall_at_fa_50", "R@FA50"),
    ("recall_at_fa_100", "R@FA100"),
    ("fpr_for_recall_90", "FPR@R90"),
    ("max_f2", "maxF2"),
    ("max_f3", "maxF3"),
]

_T0 = time.time()


def log(msg):
    line = "[%8.1fs] %s" % (time.time() - _T0, msg)
    print(line, flush=True)


def write_status(phase, extra=None):
    doc = {"phase": phase, "elapsed_s": round(time.time() - _T0, 1),
           "pid": os.getpid(), "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        doc.update(extra)
    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
    except Exception as e:                                        # never fatal
        log("WARN: status write failed: %r" % (e,))


def _fmt(v):
    return "n/a" if v is None else "%.3f" % v


# --------------------------------------------------------------------------- run
def run(n_rare, device, margin):
    write_status("load mel")
    log("loading mel embeddings (device=%s); mean pool, 512 tokens, batch %d"
        % (device, ir.MEL_BATCH))
    m = ir.load_mel(device)                                        # builds missing caches (create-only)
    log("mel loaded: train=%d test=%d langs=%s"
        % (m["Y_train"].shape[0], m["Y_test"].shape[0], ir.LANGS))

    # rare concepts = lowest TRAIN frequency among concepts present in TEST (ties: lowest index)
    freq = m["Y_train"].sum(0)
    test_present = np.where(m["Y_test"].sum(0) > 0)[0]
    order = test_present[np.argsort(freq[test_present], kind="stable")]
    rare_idx = [int(i) for i in order[:n_rare]]
    log("rare concepts = %s" % [(i, LEVEL1_EN[i], int(freq[i])) for i in rare_idx])

    X_all = {lg: np.concatenate([m["X"][lg]["train"], m["X"][lg]["test"]], axis=0)
             for lg in ir.LANGS}
    ntr = m["Y_train"].shape[0]
    train_idx = np.arange(ntr)
    test_idx = np.arange(ntr, ntr + m["Y_test"].shape[0])
    # Per-language labels: each native model trains on its OWN train labels (mel TRAIN splits
    # are not parallel across languages); the parallel TEST labels are shared for scoring.
    Y_all = {lg: np.concatenate([m["Y_train_by_lang"][lg], m["Y_test_by_lang"][lg]], axis=0)
             for lg in ir.LANGS}

    write_status("eval mel", {"rare_indices": rare_idx})
    classes = [str(i) for i in range(21)]
    res, curves = ir.evaluate_dataset(
        "mel", X_all, lambda c: {lg: Y_all[lg][:, c] for lg in ir.LANGS},
        train_idx, test_idx, classes, [str(i) for i in rare_idx])

    # enrich results with concept metadata
    for i in rare_idx:
        res[str(i)]["concept_name"] = LEVEL1_EN[i]
        res[str(i)]["eurovoc_id"] = LEVEL1_IDS[i]

    plot_pr_named(res, curves, rare_idx, OUT_PNG)

    payload = {
        "config": {
            "dataset": "nlpaueb/multi_eurlex (en/de/fr; 11000/1000/5000)",
            "version": VERSION,
            "label_fix": ("per-language train labels; v1 used EN labels for all languages"),
            "arm": "mel", "pooling": "mean", "max_len": ir.MEL_MAX_LEN, "batch": ir.MEL_BATCH,
            "langs": ir.LANGS, "targets": ir.TARGETS, "source": "en",
            "xgb_params": ir.XGB_PARAMS, "transform": "l2_thenZ [both]",
            "fpr_grid": FPR_GRID, "fa_grid": FA_GRID, "recall_target": ir.RECALL_TARGET,
            "n_rare": n_rare, "margin": margin, "device": device,
            "concept_ids": LEVEL1_IDS, "concept_names_en": LEVEL1_EN,
        },
        "datasets": {"mel": {"results": res, "n_concepts": 21, "rare_indices": rare_idx,
                             "rare_names": [LEVEL1_EN[i] for i in rare_idx],
                             "n_train": int(m["Y_train"].shape[0]),
                             "n_test": int(m["Y_test"].shape[0])}},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:               # create-only
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log("wrote %s" % OUT_JSON)

    md = build_md(payload, margin)
    with open(OUT_MD, "w", encoding="utf-8") as f:                 # create-only
        f.write(md)
    log("wrote %s" % OUT_MD)
    write_status("done", {"rare_indices": rare_idx, "outputs": [os.path.basename(OUT_JSON),
                                                                os.path.basename(OUT_MD),
                                                                os.path.basename(OUT_PNG)]})
    return payload


# --------------------------------------------------------------------------- plot
def plot_pr_named(results, curves, rare_idx, out_png):
    nrow, ncol = len(rare_idx), len(ir.TARGETS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.3 * nrow), squeeze=False)
    for i, ci in enumerate(rare_idx):
        cname = str(ci)
        for j, L in enumerate(ir.TARGETS):
            ax = axes[i][j]
            cd = curves[(cname, L)]
            y = np.asarray(cd["y"])
            for key, color, lbl in (("en", "#1f77b4", "English-trained"),
                                    ("native", "#ff7f0e", "%s-trained" % L)):
                s = cd[key]
                ap = cd["ap_en"] if key == "en" else cd["ap_native"]
                if s is None or y.sum() == 0 or y.sum() == len(y):
                    continue
                prec, rec, _ = precision_recall_curve(y, np.asarray(s))
                ax.plot(rec, prec, color=color, lw=1.6,
                        label="%s (AP=%.2f)" % (lbl, ap if ap is not None else float("nan")))
            base = y.mean() if len(y) else 0.0
            ax.axhline(base, color="grey", ls=":", lw=1)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title("%s [%d] | %s" % (LEVEL1_EN[ci], ci, L), fontsize=9)
            ax.set_xlabel("recall", fontsize=8)
            ax.set_ylabel("precision", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("imbalanced_recall_mel: PR curves, English-trained vs native-trained "
                 "(MultiEURLEX, rare EuroVoc level-1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    log("wrote %s" % out_png)


# --------------------------------------------------------------------------- report
def build_md(payload, margin):
    d = payload["datasets"]["mel"]
    res = d["results"]
    rare_idx = d["rare_indices"]
    lines = []
    lines.append("# imbalanced_recall_mel_%s -- MultiEURLEX: English-trained vs native-trained at "
                 "HIGH RECALL" % VERSION)
    lines.append("")
    lines.append("> **v1 label bug (fixed in %s):** MultiEURLEX TRAIN splits are NOT parallel across "
                 "languages, but the v1 `load_mel` trained the native de/fr models with the ENGLISH "
                 "train labels, so the native column in `imbalanced_recall_mel.md` was near-random. "
                 "%s loads each language's OWN train labels (the TEST split is parallel, so its "
                 "labels are shared)." % (VERSION, VERSION))
    lines.append("")
    lines.append("Long, non-English EU legal documents (`nlpaueb/multi_eurlex`, en/de/fr; "
                 "train/test = %d/%d).  Recipe: EuroBERT-210m **mean** pool, 512-token truncation, "
                 "batch %d -> `l2_thenZ [both]` -> OVR `XGBClassifier(binary:logistic, hist, "
                 "max_bin=64, device=cpu, n_jobs=2, seed=0, max_depth=8, min_child_weight=5, lr=0.1, "
                 "n_estimators=400)`.  **English-trained** = fit on EN train, score target test; "
                 "**native** = fit on the target-language train, score target test."
                 % (d["n_train"], d["n_test"], payload["config"]["batch"]))
    lines.append("")
    lines.append("Rare concepts = the %d lowest-TRAIN-frequency EuroVoc level-1 concepts present in "
                 "test:" % len(rare_idx))
    lines.append("")
    lines.append("| idx | EuroVoc id | concept | train_pos | test_pos |")
    lines.append("|---|---|---|---|---|")
    for ci in rare_idx:
        c = res[str(ci)]
        lines.append("| %d | %s | %s | %d | %d |" % (ci, c["eurovoc_id"], c["concept_name"],
                                                     c["n_pos_train"], c["n_pos_test"]))
    lines.append("")
    lines.append("Lower is better only for `FPR@R90`; every other column is **higher is better**.  "
                 "Cells read `EN / native`.  Comparable tolerance (headline): EN within %.3f of native."
                 % margin)
    lines.append("")

    n_cells = n_cells_hi = n_comp_ap = n_comp_hi = 0
    fa100_en, fa100_nat, fa50_en, fa50_nat, fpr90_en, fpr90_nat = [], [], [], [], [], []

    for ci in rare_idx:
        c = res[str(ci)]
        lines.append("## `%d` %s  (train_pos=%d, test_pos=%d, EuroVoc %s)"
                     % (ci, c["concept_name"], c["n_pos_train"], c["n_pos_test"], c["eurovoc_id"]))
        lines.append("")
        hdr = "| metric | " + " | ".join("%s EN / native" % L for L in ir.TARGETS) + " |"
        lines.append(hdr)
        lines.append("|" + "---|" * (len(ir.TARGETS) + 1))
        for key, label in METRIC_ORDER:
            row = [label]
            for L in ir.TARGETS:
                cell = c["per_target"][L]
                en = cell["en"][key] if cell["en"] else None
                nat = cell["native"][key] if cell["native"] else None
                row.append("%s / %s" % (_fmt(en), _fmt(nat)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        for L in ir.TARGETS:
            cell = c["per_target"][L]
            if not cell["en"] or not cell["native"]:
                continue
            n_cells += 1
            e_ap, n_ap = cell["en"]["average_precision"], cell["native"]["average_precision"]
            if e_ap is not None and n_ap is not None and e_ap >= n_ap - margin:
                n_comp_ap += 1
            h_ok = h_tot = 0
            for rk in HL_KEYS:
                ev, nv = cell["en"][rk], cell["native"][rk]
                if ev is None or nv is None:
                    continue
                h_tot += 1
                if ev >= nv - margin:
                    h_ok += 1
            if h_tot:
                n_cells_hi += 1
                if h_ok == h_tot:
                    n_comp_hi += 1
            if cell["en"].get("recall_at_fa_100") is not None and \
                    cell["native"].get("recall_at_fa_100") is not None:
                fa100_en.append(cell["en"]["recall_at_fa_100"])
                fa100_nat.append(cell["native"]["recall_at_fa_100"])
            if cell["en"].get("recall_at_fa_50") is not None and \
                    cell["native"].get("recall_at_fa_50") is not None:
                fa50_en.append(cell["en"]["recall_at_fa_50"])
                fa50_nat.append(cell["native"]["recall_at_fa_50"])
            if cell["en"].get("fpr_for_recall_90") is not None and \
                    cell["native"].get("fpr_for_recall_90") is not None:
                fpr90_en.append(cell["en"]["fpr_for_recall_90"])
                fpr90_nat.append(cell["native"]["fpr_for_recall_90"])

    # compact headline table: high-recall operating points per concept x target
    lines.append("## Headline EN-vs-native recall table (high-recall regime)")
    lines.append("")
    lines.append("| concept | target | R@FA50 EN/nat | R@FA100 EN/nat | R@FPR10% EN/nat | "
                 "FPR@R90 EN/nat | AP EN/nat |")
    lines.append("|---|---|---|---|---|---|---|")
    for ci in rare_idx:
        c = res[str(ci)]
        for L in ir.TARGETS:
            cell = c["per_target"][L]
            if not cell["en"] or not cell["native"]:
                continue
            e, n = cell["en"], cell["native"]
            lines.append("| %s [%d] | %s | %s / %s | %s / %s | %s / %s | %s / %s | %s / %s |" % (
                c["concept_name"], ci, L,
                _fmt(e["recall_at_fa_50"]), _fmt(n["recall_at_fa_50"]),
                _fmt(e["recall_at_fa_100"]), _fmt(n["recall_at_fa_100"]),
                _fmt(e["recall_at_fpr_10"]), _fmt(n["recall_at_fpr_10"]),
                _fmt(e["fpr_for_recall_90"]), _fmt(n["fpr_for_recall_90"]),
                _fmt(e["average_precision"]), _fmt(n["average_precision"])))
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append("- Cells evaluated (rare concept x target-language): **%d**." % n_cells)
    if n_cells:
        lines.append("- English-trained AP within %.2f of native in **%d/%d** cells (%.0f%%)."
                     % (margin, n_comp_ap, n_cells, 100 * n_comp_ap / n_cells))
    if n_cells_hi:
        lines.append("- **High-recall regime** (R@FPR10%% + R@FA50 + R@FA100 all within %.2f of "
                     "native): **%d/%d** cells (%.0f%%).  This is the regime the task cares about."
                     % (margin, n_comp_hi, n_cells_hi, 100 * n_comp_hi / n_cells_hi))
    lines.append("")
    lines.append("### False-alarm cost at high recall")
    lines.append("")
    if fa100_en:
        lines.append("- Mean recall at 100 false alarms: EN **%.3f** vs native **%.3f** "
                     "(across %d cells)." % (float(np.mean(fa100_en)), float(np.mean(fa100_nat)),
                                             len(fa100_en)))
    if fa50_en:
        lines.append("- Mean recall at 50 false alarms: EN **%.3f** vs native **%.3f** "
                     "(across %d cells)." % (float(np.mean(fa50_en)), float(np.mean(fa50_nat)),
                                             len(fa50_en)))
    if fpr90_en:
        lines.append("- Mean FPR needed to reach recall=0.90: EN **%.3f** vs native **%.3f** "
                     "(across %d cells).  Smaller is better; the difference is the extra false-alarm "
                     "rate the English model spends for the same 90%% recall."
                     % (float(np.mean(fpr90_en)), float(np.mean(fpr90_nat)), len(fpr90_en)))
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-rare", type=int, default=8, help="number of rarest concepts (default 8)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--margin", type=float, default=0.05, help="comparable tolerance (default 0.05)")
    ap.add_argument("--launch", action="store_true", help="relaunch detached hidden")
    args = ap.parse_args()

    if args.launch:
        passthrough = [sys.executable, os.path.abspath(__file__),
                       "--n-rare", str(args.n_rare), "--device", args.device,
                       "--margin", str(args.margin)]
        logpath = os.path.join(os.environ.get("TEMP", r"C:\Users\pedro\AppData\Local\Temp"),
                               "opencode", LOG_NAME)
        os.makedirs(os.path.dirname(logpath), exist_ok=True)
        vbs = os.path.join(REPO, "launch_hidden.vbs")
        cmd = subprocess.list2cmdline(passthrough)
        subprocess.Popen(["wscript", "//B", "//Nologo", vbs, cmd, logpath, REPO], cwd=REPO)
        print("launched detached; log=%s" % logpath)
        return

    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    dev = args.device
    if dev == "auto":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("start n_rare=%d device=%s pid=%d" % (args.n_rare, dev, os.getpid()))
    write_status("start", {"device": dev, "n_rare": args.n_rare})
    run(args.n_rare, dev, args.margin)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        write_status("error")
        raise
