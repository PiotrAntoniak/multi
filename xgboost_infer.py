"""XGBoost topic inference (lenient labels) — deterministic rebuild from stored params.

Models are NOT persisted: this script deterministically rebuilds the 8 binary tag
models (and their per-tag thresholds) from the params/thresholds stored in the
repository artifacts, then predicts and writes a CSV.

Two sources:

* ``--source transfer`` (default): the Optuna-best params are parsed from
  ``xgboost_topic_results_lenient_{mode}.md`` (via
  ``pertopic_accuracy_transfer.parse_best_params``). The 8 English tag models are
  retrained on ALL 2009 English rows and the transfer thresholds are re-tuned
  in-sample exactly like the pipeline (``X.tune_thresholds`` on the all-2009
  English train probabilities). Target features are the repo embeddings
  ``load_mode_embeddings(mode)[lang]`` (or a user ``--input`` npy); ``--shift``
  adds ``D_lang = mean(emb_en - emb_lang)`` over all 2009 rows.
* ``--source perlang``: params + thresholds are read from
  ``xgboost_perlang_lenient_{lang}.json``
  (``modes[mode].tuned.best_params`` and
  ``modes[mode].tuned.tuned.thresholds.test``). The 8 tag models are trained on
  the dev+devtest embeddings of ``lang`` and predict the test split (or
  ``--input``).

Output: ``predictions_{source}_{mode}_{lang}[_shift].csv`` in ``--outdir`` with
columns ``row_index, id, prob_<tag> x8, tags`` (``id`` only when predicting repo
embeddings; taken from the FLORES CSV row ids). Console reports the row count and
the per-tag predicted-positive counts. ``--eval`` additionally prints per-tag
accuracy + F1 against the gold lenient labels and the overall subset accuracy
(only possible when predicting repo embeddings: transfer evaluates all 2009 rows,
perlang evaluates its test split).

Usage:
  python xgboost_infer.py --mode {mean,bos,eos,lead} [--source transfer|perlang]
      [--lang {en,it,de,fr}] [--shift] [--input FILE.npy] [--outdir .] [--eval]
"""
import os

# Must be set before numpy / xgboost import to take effect.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import xgboost_topic_optuna as X
import pertopic_accuracy_transfer as PT


def load_input_npy(path):
    """Load a user feature matrix (n, 768) float32."""
    arr = np.load(path)
    assert arr.ndim == 2 and arr.shape[1] == 768, f"--input must be (n, 768); got {arr.shape}"
    assert arr.dtype == np.float32, f"--input must be float32; got {arr.dtype}"
    return arr


def train_models(params, X_train, Y_train):
    """Train one binary XGBClassifier per tag; return (models, train_probs)."""
    models, train_probs = [], np.zeros((X_train.shape[0], len(X.TAGS)))
    for j in range(len(X.TAGS)):
        model = X.xgb.XGBClassifier(**X.make_params(params))
        model.fit(X_train, Y_train[:, j])
        models.append(model)
        train_probs[:, j] = model.predict_proba(X_train)[:, 1]
    return models, train_probs


def predict(models, Xt):
    return np.column_stack([m.predict_proba(Xt)[:, 1] for m in models])


def tag_strings(pred):
    return [";".join(tag for j, tag in enumerate(X.TAGS) if pred[i, j]) for i in range(len(pred))]


def write_predictions(path, probs, pred, row_index, ids):
    data = {"row_index": row_index}
    if ids is not None:
        data["id"] = ids
    for j, tag in enumerate(X.TAGS):
        data[f"prob_{tag}"] = probs[:, j]
    data["tags"] = tag_strings(pred)
    pd.DataFrame(data).to_csv(path, index=False)


def report_predicted(probs, pred):
    pos = pred.sum(axis=0)
    print(f"[infer] predicted positives: "
          + " ".join(f"{tag}={int(pos[j])}" for j, tag in enumerate(X.TAGS)), flush=True)


def eval_gold(Y_true, probs, pred, label):
    acc = (pred == Y_true).mean(axis=0)
    f1 = f1_score(Y_true, pred, average=None, zero_division=0)
    subset = accuracy_score(Y_true, pred)
    print(f"[eval] {label}: n={len(Y_true)} subset_accuracy={subset:.4f}", flush=True)
    print("[eval] per-tag accuracy: "
          + " ".join(f"{tag}={acc[j]:.3f}" for j, tag in enumerate(X.TAGS)), flush=True)
    print("[eval] per-tag F1: "
          + " ".join(f"{tag}={f1[j]:.3f}" for j, tag in enumerate(X.TAGS)), flush=True)


def run_transfer(args, df, Y, masks):
    md_path = os.path.join(args.outdir, f"xgboost_topic_results_lenient_{args.mode}.md")
    md_text = open(md_path, encoding="utf-8").read()
    params, best_trial = PT.parse_best_params(md_text)
    if params is None:
        raise SystemExit(f"no 'Best Optuna parameters' block in {md_path}")
    print(f"[infer] source=transfer mode={args.mode} lang={args.lang} shift={args.shift} "
          f"best_trial={best_trial}", flush=True)

    Xd = X.load_mode_embeddings(args.mode)
    models, train_probs = train_models(params, Xd["en"], Y)
    th = X.tune_thresholds(Y, train_probs)
    print("[infer] transfer thresholds (all-2009 English in-sample): "
          + " ".join(f"{tag}={th[j]:.2f}" for j, tag in enumerate(X.TAGS)), flush=True)

    shift_vec = None
    if args.shift:
        if args.lang == "en":
            print("[infer] --shift ignored for lang=en", flush=True)
        else:
            shift_vec = (Xd["en"] - Xd[args.lang]).mean(axis=0)

    if args.input:
        Xt = load_input_npy(args.input)
        if shift_vec is not None:
            Xt = Xt + shift_vec
        ids = None
        row_index = np.arange(len(Xt))
        gold = None
    else:
        Xt = Xd[args.lang]
        if shift_vec is not None:
            Xt = Xt + shift_vec
        ids = df["id"].values
        row_index = np.arange(len(Xt))
        gold = (Y, "all 2009 rows")

    probs = predict(models, Xt)
    pred = (probs >= th[None, :]).astype(int)
    print(f"[infer] rows={len(pred)}", flush=True)
    report_predicted(probs, pred)

    name = f"predictions_transfer_{args.mode}_{args.lang}" + ("_shift" if args.shift else "") + ".csv"
    path = os.path.join(args.outdir, name)
    write_predictions(path, probs, pred, row_index, ids)
    print(f"[infer] wrote {path}", flush=True)

    if args.eval:
        if gold is None:
            print("[eval] skipped: --input has no gold labels", flush=True)
        else:
            eval_gold(gold[0], probs, pred, gold[1])


def run_perlang(args, df, Y, masks):
    path = os.path.join(args.outdir, f"xgboost_perlang_lenient_{args.lang}.json")
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    if args.mode not in j["modes"]:
        raise SystemExit(f"mode {args.mode} absent in {path}")
    mr = j["modes"][args.mode]
    params = mr["tuned"]["best_params"]
    th = np.array(mr["tuned"]["tuned"]["thresholds"]["test"], dtype=float)
    print(f"[infer] source=perlang lang={args.lang} mode={args.mode} shift={args.shift} "
          f"best_trial={mr['tuned']['best_trial']}", flush=True)

    Xf = X.load_one_embedding(args.mode, args.lang)
    models, _ = train_models(params, Xf[masks["dev_devtest"]], Y[masks["dev_devtest"]])
    print("[infer] perlang thresholds (stored test): "
          + " ".join(f"{tag}={th[j]:.2f}" for j, tag in enumerate(X.TAGS)), flush=True)

    shift_vec = None
    if args.shift:
        if args.lang == "en":
            print("[infer] --shift ignored for lang=en", flush=True)
        else:
            shift_vec = (X.load_one_embedding(args.mode, "en")
                         - X.load_one_embedding(args.mode, args.lang)).mean(axis=0)

    if args.input:
        Xt = load_input_npy(args.input)
        if shift_vec is not None:
            Xt = Xt + shift_vec
        ids = None
        row_index = np.arange(len(Xt))
        gold = None
    else:
        Xt = Xf[masks["test"]]
        if shift_vec is not None:
            Xt = Xt + shift_vec
        ids = df["id"].values[masks["test"]]
        row_index = np.where(masks["test"])[0]
        gold = (Y[masks["test"]], "test split (506)")

    probs = predict(models, Xt)
    pred = (probs >= th[None, :]).astype(int)
    print(f"[infer] rows={len(pred)}", flush=True)
    report_predicted(probs, pred)

    name = f"predictions_perlang_{args.mode}_{args.lang}" + ("_shift" if args.shift else "") + ".csv"
    out_path = os.path.join(args.outdir, name)
    write_predictions(out_path, probs, pred, row_index, ids)
    print(f"[infer] wrote {out_path}", flush=True)

    if args.eval:
        if gold is None:
            print("[eval] skipped: --input has no gold labels", flush=True)
        else:
            eval_gold(gold[0], probs, pred, gold[1])


def main():
    ap = argparse.ArgumentParser(description="XGBoost topic inference (lenient labels).")
    ap.add_argument("--mode", required=True, choices=X.MODES)
    ap.add_argument("--source", default="transfer", choices=["transfer", "perlang"])
    ap.add_argument("--lang", default="en", choices=X.LANGS)
    ap.add_argument("--shift", action="store_true")
    ap.add_argument("--input", default=None)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    df, Y, masks, counts = X.load_data("lenient")
    if args.source == "transfer":
        run_transfer(args, df, Y, masks)
    else:
        run_perlang(args, df, Y, masks)


if __name__ == "__main__":
    main()
