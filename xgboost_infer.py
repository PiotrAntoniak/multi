"""XGBoost topic inference (lenient labels) — standalone, no stored artifacts.

The 8 binary tag models (one per canonical topic) and their per-tag decision
thresholds are NOT persisted.  This script deterministically rebuilds them from
the 50-trial lenient Optuna best params embedded below, then predicts and writes
a CSV.

Sources:

* ``--source transfer`` (default): train the 8 English tag models on ALL 2009
  English rows using the mode's embedded best params; the per-tag thresholds are
  re-tuned in-sample (grid 0.05..0.95) unless ``--thresholds`` forces a fixed
  value.  Target features are the repo embeddings
  ``load_mode_embeddings(mode)[lang]`` (or a user ``--input`` npy).  ``--shift``
  adds ``D_lang = mean(emb_en - emb_lang)`` computed over all 2009 rows.
* ``--source perlang``: train the 8 tag models on the dev+devtest embeddings of
  ``lang`` using the mode's embedded best params, and tune the thresholds
  in-sample (dev+devtest) unless ``--thresholds`` forces a fixed value.  Predict
  the test split (or ``--input``).

The embedded params are the 50-trial lenient Optuna best params used for the
README results.  ``--params-json FILE`` overrides the mode's params with an
8-key JSON dict (``max_depth, learning_rate, n_estimators, subsample,
colsample_bytree, min_child_weight, reg_lambda, gamma``).

Output: ``predictions_{source}_{mode}_{lang}[_shift].csv`` in ``--outdir`` with
columns ``row_index, id, prob_<tag> x8, tags`` (``id`` only when predicting repo
embeddings: taken from the FLORES CSV row ids).  Console reports the row count
and the per-tag predicted-positive counts.  ``--eval`` additionally prints
per-tag accuracy + F1 against the gold lenient labels and the overall subset
accuracy (only possible when predicting repo embeddings: transfer evaluates all
2009 rows, perlang evaluates its test split).

Usage:
  python xgboost_infer.py --mode {mean,bos,eos,lead} [--source transfer|perlang]
      [--lang {en,it,de,fr}] [--shift] [--input FILE.npy] [--outdir .] [--eval]
      [--thresholds 0.5] [--params-json FILE]
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

# 50-trial lenient Optuna best params (shared across all 8 labels, one set per
# pooling mode) used to produce the README results.  These are the values the
# reports were generated with; nothing is read from disk.
BEST_PARAMS = {
    "mean": dict(max_depth=7, learning_rate=0.19081198418161657,
                 n_estimators=628, subsample=0.7736546560469247,
                 colsample_bytree=0.6280618526352608, min_child_weight=8,
                 reg_lambda=0.028953812200939687, gamma=0.2817481092827589),
    "bos": dict(max_depth=7, learning_rate=0.18524575730482706,
                n_estimators=618, subsample=0.7310109496863255,
                colsample_bytree=0.4559809851959609, min_child_weight=10,
                reg_lambda=0.010003809333198692, gamma=2.390027137593299),
    "eos": dict(max_depth=5, learning_rate=0.25247341609360224,
                n_estimators=494, subsample=0.7180369539220692,
                colsample_bytree=0.32665699233832174, min_child_weight=9,
                reg_lambda=0.023696324760734095, gamma=2.310292782299694),
    "lead": dict(max_depth=7, learning_rate=0.19908924391090863,
                 n_estimators=196, subsample=0.6369671093094386,
                 colsample_bytree=0.6638217541091802, min_child_weight=10,
                 reg_lambda=0.002237873237141299, gamma=0.18821319411836626),
}
PARAM_KEYS = ["max_depth", "learning_rate", "n_estimators", "subsample",
              "colsample_bytree", "min_child_weight", "reg_lambda", "gamma"]


def resolve_params(args):
    """Mode's embedded best params, or an 8-key JSON dict from --params-json."""
    if args.params_json:
        with open(args.params_json, encoding="utf-8") as f:
            params = json.load(f)
        missing = [k for k in PARAM_KEYS if k not in params]
        if missing:
            raise SystemExit(f"--params-json missing keys: {missing}")
        return params
    return dict(BEST_PARAMS[args.mode])


def resolve_thresholds(args, Y_train, train_probs):
    """Fixed scalar from --thresholds, else per-tag in-sample tuning on train."""
    if args.thresholds is not None:
        return np.full(len(X.TAGS), float(args.thresholds), dtype=float)
    return X.tune_thresholds(Y_train, train_probs)


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


def report_predicted(pred):
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


def run_transfer(args, df, Y):
    params = resolve_params(args)
    print(f"[infer] source=transfer mode={args.mode} lang={args.lang} shift={args.shift} "
          f"params={params}", flush=True)

    Xd = X.load_mode_embeddings(args.mode)
    models, train_probs = train_models(params, Xd["en"], Y)
    th = resolve_thresholds(args, Y, train_probs)
    how = "fixed" if args.thresholds is not None else "all-2009 English in-sample"
    print(f"[infer] transfer thresholds ({how}): "
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
    report_predicted(pred)

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
    params = resolve_params(args)
    print(f"[infer] source=perlang lang={args.lang} mode={args.mode} shift={args.shift} "
          f"params={params}", flush=True)

    Xf = X.load_one_embedding(args.mode, args.lang)
    X_train = Xf[masks["dev_devtest"]]
    Y_train = Y[masks["dev_devtest"]]
    models, train_probs = train_models(params, X_train, Y_train)
    th = resolve_thresholds(args, Y_train, train_probs)
    how = "fixed" if args.thresholds is not None else "dev+devtest in-sample"
    print(f"[infer] perlang thresholds ({how}): "
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
    report_predicted(pred)

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
    ap.add_argument("--thresholds", type=float, default=None,
                    help="force this fixed per-tag threshold (default: tune in-sample)")
    ap.add_argument("--params-json", default=None,
                    help="JSON file with the 8 hyperparameters to override the mode's best params")
    args = ap.parse_args()

    df, Y, masks, _ = X.load_data("lenient")
    if args.source == "transfer":
        run_transfer(args, df, Y)
    else:
        run_perlang(args, df, Y, masks)


if __name__ == "__main__":
    main()
