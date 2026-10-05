"""12-class single-label FLORES topic classifier over BOTH FLORES datasets.

Datasets
--------
* sentence : ``flores200_en_it_de_fr.csv`` (2006 rows, one 12-way ``topic`` per row).
  train = the 997 ``dev`` rows; the 1009 ``devtest`` rows are permuted with seed 0 and
  split into val (first 504) / test (last 505).
* agg      : ``flores_agg.csv`` (562 URL-aggregated rows, one 12-way ``topic`` per URL).
  train = URLs whose sentence rows are ``dev``; the remaining URLs (incl. the one orphan
  whose sentence rows were deleted, hence no ``dev`` rows -> treated as devtest) are
  permuted with seed 0 and split into val (first 140) / test (last 141).

Labels are the 12 sorted topic strings -> class indices 0..11; the class list is stored in
every manifest and registry entry.

Embeddings and alignment
------------------------
Sentence embeddings ``embeddings/{mode}/emb_{lang}.npy`` still have the historical 2009 rows
while the CSV now has 2006.  The 3 deleted rows are recovered by matching the current CSV
against ``git show 196bbd3:flores200_en_it_de_fr.csv`` on ``(split, id, en)``; the embeddings
are sliced to the kept indices (verify: length + spot rows).  Agg embeddings are read as-is
from ``embeddings_agg/{mode}/emb_{lang}.npy``.

Conditions
----------
``full``    : all 768 dims.
``prune50`` : zero the top-k dims carrying 50% of the 4-language eta^2 mass, per dataset+mode.
              Sentence k comes from ``lang_metrics.top_dims(mode, 0.5)``; agg k is computed
              with ``lang_metrics.eta2`` on the agg embeddings.  The same dims are zeroed in
              train/val/test arrays for self and transfer evaluation.

Protocol (16 cells = dataset {sent,agg} x mode {mean,bos} x condition {full,prune50} x train
language {en,it})
--------------------------------------------------------------------------------------------------
* ``XGBClassifier(objective="multi:softprob", num_class=12, tree_method="hist", device="cuda",
  n_jobs=2)`` + ``xgboost_topic_optuna._optuna_search_space`` params (the search space does not
  set ``device``, so every Optuna trial inherits ``device="cuda"`` from ``BASE_PARAMS``).
* Optuna TPE seed 0, 10 trials; objective = validation AUC (OVR macro).
* Best params retrained on train+val; evaluated on the model's own language test (self) and on
  the OTHER language test (transfer), same condition/pruning.  Metrics: accuracy, macro-F1,
  per-class F1 and OVR AUC (macro + per-class).
* Each final model is saved with ``model_store.save_bundle`` as
  ``flores12_{sent|agg}_{lang}_{mode}_{full|prune50}_optuna10`` and appended to the tracked
  ``models_registry.json`` at the repo root (``models/`` stays git-ignored).

Driver
------
Cells run sequentially with 2 threads and write ``train_flores12.log``, ``train_flores12.pid``
and ``train_flores12_status.json`` (cell / trial / best-so-far, rewritten after every trial).
Launch detached with the repo's hidden launcher::

    python train_flores12.py --launch                 # spawns itself via launch_hidden.vbs
    # or explicitly:
    wscript //B //Nologo launch_hidden.vbs "python train_flores12.py"

Sharding
--------
Optuna search space does not set ``device``, so every Optuna trial inherits ``device="cuda"`` from
``BASE_PARAMS`` (see the smoke-tested GPU support in the Protocol section above).

``--shard I/N`` (default ``0/1``) takes the 16 cells round-robin, so N parallel shards split the
work evenly (``--shard 0/4 .. 3/4`` => 4 cells each).  With GPU execution, run ONE process (the
default ``0/1``): multiple shards would contend for the single GPU.  Each shard keeps 2 threads
(``n_jobs=2`` and OMP/MKL/OPENBLAS=2).  For ``N > 1`` a shard writes
``train_flores12_shard{i}.log`` / ``.pid`` / ``_status.json`` and its own
``models_registry_shard{i}.json`` (the shared ``models_registry.json`` is NOT touched).  ``0/1``
keeps the legacy shared filenames/registry.  Merge the shard registries afterwards with::

    python train_flores12.py --merge-registry 4
"""
import os

# Must be set before numpy / xgboost import to take effect (2-thread runtime budget).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import argparse
import io
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from lang_metrics import eta2, top_dims
from model_store import save_bundle
from xgboost_topic_optuna import _optuna_search_space

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
REPO = os.path.dirname(os.path.abspath(__file__))
SENT_CSV = os.path.join(REPO, "flores200_en_it_de_fr.csv")
AGG_CSV = os.path.join(REPO, "flores_agg.csv")
EMB_SENT_DIR = os.path.join(REPO, "embeddings")
EMB_AGG_DIR = os.path.join(REPO, "embeddings_agg")
MODELS_ROOT = os.path.join(REPO, "models")
REGISTRY_PATH = os.path.join(REPO, "models_registry.json")
LOG_PATH = os.path.join(REPO, "train_flores12.log")
PID_PATH = os.path.join(REPO, "train_flores12.pid")
STATUS_PATH = os.path.join(REPO, "train_flores12_status.json")

LANGS = ["en", "it", "de", "fr"]          # all four, for the language eta^2
TRAIN_LANGS = ["en", "it"]                # the two train/eval languages
MODES = ["mean", "bos"]
CONDITIONS = ["full", "prune50"]
DATASETS = ["sent", "agg"]

N_CLASSES = 12
N_TRIALS = 10
SEED = 0
PRUNE_FRAC = 0.50
GIT_REF = "196bbd3:flores200_en_it_de_fr.csv"

# Fixed multiclass base params (Optuna supplies the tuned ones on top).
BASE_PARAMS = dict(
    objective="multi:softprob",
    num_class=N_CLASSES,
    tree_method="hist",
    device="cuda",
    max_bin=64,
    n_jobs=2,
    seed=SEED,
)


# ----------------------------------------------------------------------------
# Logging / pid / status / registry
# ----------------------------------------------------------------------------
class _Tee:
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _open_log():
    f = open(LOG_PATH, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    return f


def write_pid(pid=None):
    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid() if pid is None else pid))


def write_status(obj):
    obj["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, STATUS_PATH)


def update_registry(entry):
    """Append (or replace by name) one entry in the tracked models_registry.json."""
    entries = []
    if os.path.isfile(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data if isinstance(data, list) else [data]
    entries = [e for e in entries if e.get("name") != entry["name"]]
    entries.append(entry)
    entries.sort(key=lambda e: e.get("name", ""))
    tmp = REGISTRY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    os.replace(tmp, REGISTRY_PATH)


def set_shard_paths(shard_i, shard_n):
    """Point the log/pid/status/registry globals at this shard's files.

    A single shard (``n == 1``) keeps the original shared paths, so ``--shard 0/1`` behaves
    exactly like the legacy single-process run.  ``n > 1`` uses ``train_flores12_shard{i}.*``
    files and a per-shard registry, so parallel shards never write the shared
    ``models_registry.json`` (merge them later with ``--merge-registry N``).
    """
    global LOG_PATH, PID_PATH, STATUS_PATH, REGISTRY_PATH
    if shard_n <= 1:
        return
    stem = os.path.join(REPO, f"train_flores12_shard{shard_i}")
    LOG_PATH = stem + ".log"
    PID_PATH = stem + ".pid"
    STATUS_PATH = stem + "_status.json"
    REGISTRY_PATH = os.path.join(REPO, f"models_registry_shard{shard_i}.json")


def merge_registries(shard_n):
    """Merge the per-shard registries into the shared models_registry.json (dedup by name)."""
    merged = {}
    for i in range(shard_n):
        path = os.path.join(REPO, f"models_registry_shard{i}.json")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for e in (data if isinstance(data, list) else [data]):
            merged[e["name"]] = e
    entries = sorted(merged.values(), key=lambda e: e.get("name", ""))
    tmp = REGISTRY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    os.replace(tmp, REGISTRY_PATH)
    return len(entries)


# ----------------------------------------------------------------------------
# Alignment (current 2006-row sentence CSV vs historical 2009-row embeddings)
# ----------------------------------------------------------------------------
def _git_sentence_csv(ref=GIT_REF):
    proc = subprocess.run(["git", "-C", REPO, "show", ref], capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("git show %s failed: %s"
                           % (ref, proc.stderr.decode("utf-8", "replace")))
    return pd.read_csv(io.BytesIO(proc.stdout))


def align_sentence_indices(cur, git):
    """Map every current row to its index in the git (2009-row) file.

    The current CSV is the git sequence with 3 rows removed; the embeddings still follow the
    git order, so these indices slice the arrays back to the current rows.
    """
    git_pos = {}
    for i, (split, rid, en) in enumerate(zip(git["split"], git["id"], git["en"])):
        key = (str(split), int(rid), str(en))
        if key in git_pos:
            raise ValueError("duplicate (split,id,en) in git CSV: %r" % (key,))
        git_pos[key] = i
    kept = np.empty(len(cur), dtype=int)
    for i, (split, rid, en) in enumerate(zip(cur["split"], cur["id"], cur["en"])):
        key = (str(split), int(rid), str(en))
        if key not in git_pos:
            raise ValueError("current row %d not found in git CSV: %r" % (i, key))
        kept[i] = git_pos[key]
    if len(np.unique(kept)) != len(kept):
        raise ValueError("current rows are not a unique subset of the git rows")
    if not np.all(np.diff(kept) > 0):
        raise ValueError("current CSV is not a subsequence of the git row order")
    return kept


# ----------------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------------
def class_list(df):
    classes = sorted(df["topic"].astype(str).unique())
    if len(classes) != N_CLASSES:
        raise ValueError("expected %d classes, got %d: %s"
                         % (N_CLASSES, len(classes), classes))
    return classes


def class_indices(df, classes):
    idx = {c: i for i, c in enumerate(classes)}
    out = np.empty(len(df), dtype=int)
    for i, t in enumerate(df["topic"].astype(str)):
        out[i] = idx[t]
    return out


def prepare_sentence():
    """Load + align the sentence CSV; return row indices and the seed-0 val/test split."""
    cur = pd.read_csv(SENT_CSV)
    git = _git_sentence_csv()
    if len(git) != 2009:
        raise ValueError("git CSV has %d rows, expected 2009" % len(git))
    if len(cur) != 2006:
        raise ValueError("current CSV has %d rows, expected 2006" % len(cur))
    kept = align_sentence_indices(cur, git)
    dropped = sorted(set(range(len(git))) - set(kept.tolist()))
    classes = class_list(cur)
    y = class_indices(cur, classes)

    dev_idx = np.where((cur["split"] == "dev").values)[0]
    dt_idx = np.where((cur["split"] == "devtest").values)[0]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(dt_idx)
    n_val = len(perm) // 2
    train_idx, val_idx, test_idx = dev_idx, perm[:n_val], perm[n_val:]

    if len(dev_idx) != 997 or len(dt_idx) != 1009:
        raise ValueError("unexpected split sizes: dev=%d devtest=%d"
                         % (len(dev_idx), len(dt_idx)))
    return {
        "cur": cur, "git": git, "kept": kept, "dropped": dropped,
        "classes": classes, "y": y,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
    }


def prepare_agg(sent_cur, classes):
    """Load the URL-aggregated CSV; a URL is train iff its sentence rows are dev.

    The one orphan (no sentence rows at all) has no dev rows, so it falls into devtest.
    """
    agg = pd.read_csv(AGG_CSV)
    url_split = {}
    for url, grp in sent_cur.groupby("URL"):
        splits = set(grp["split"])
        if len(splits) != 1:
            raise ValueError("URL %s spans splits %s" % (url, splits))
        url_split[url] = splits.pop()

    split = agg["URL"].map(url_split).fillna("devtest")
    if set(agg["topic"].astype(str)) - set(classes):
        raise ValueError("agg labels not a subset of the sentence classes")
    y = class_indices(agg, classes)

    train_idx = np.where((split == "dev").values)[0]
    dt_idx = np.where((split == "devtest").values)[0]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(dt_idx)
    n_val = len(perm) // 2
    return {
        "agg": agg, "y": y, "split": split.values,
        "train_idx": train_idx, "val_idx": perm[:n_val], "test_idx": perm[n_val:],
    }


# ----------------------------------------------------------------------------
# Embeddings + pruning
# ----------------------------------------------------------------------------
def load_sent_embeddings(mode, kept, n_source):
    """Load the sentence embeddings and slice the historical rows down to the current CSV.

    ``n_source`` is the raw embedding row count (the git file length, 2009); the kept
    indices must all be inside it, and the sliced arrays must have ``len(kept)`` rows.
    """
    out = {}
    for lang in LANGS:
        path = os.path.join(EMB_SENT_DIR, mode, f"emb_{lang}.npy")
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[0] != n_source:
            raise ValueError("expected %d rows in %s, got %s"
                             % (n_source, path, arr.shape))
        if int(kept.max()) >= n_source:
            raise ValueError("kept index %d out of range for %s" % (kept.max(), path))
        sliced = arr[kept]
        if sliced.shape[0] != len(kept):
            raise ValueError("alignment produced %d rows, expected %d"
                             % (sliced.shape[0], len(kept)))
        out[lang] = sliced
    return out


def load_agg_embeddings(mode, n_rows):
    out = {}
    for lang in LANGS:
        path = os.path.join(EMB_AGG_DIR, mode, f"emb_{lang}.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError("agg embeddings missing: %s" % path)
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[0] != n_rows:
            raise ValueError("expected %d rows in %s, got %s" % (n_rows, path, arr.shape))
        out[lang] = arr
    return out


def sent_prune_dims(mode):
    """Sentence prune50 dims from the shared language-mass helper (all 2009 embedding rows)."""
    dims, k = top_dims(mode, PRUNE_FRAC)
    return np.asarray(dims, dtype=int), int(k)


def agg_prune_dims(mode, agg_emb):
    """Agg prune50 dims: eta^2 over the four agg languages, top dims covering PRUNE_FRAC."""
    x_all = np.vstack([agg_emb[l] for l in LANGS])
    y_lang = np.repeat(LANGS, agg_emb["en"].shape[0])
    e2 = np.nan_to_num(eta2(x_all, y_lang))
    order = np.argsort(-e2)
    cum = np.cumsum(e2[order]) / e2.sum()
    k = int(np.searchsorted(cum, PRUNE_FRAC) + 1)
    return order[:k].astype(int), k


def apply_prune(arr, dims):
    if dims is None or len(dims) == 0:
        return arr
    out = arr.copy()
    out[:, dims] = 0.0
    return out


# ----------------------------------------------------------------------------
# Model / metrics
# ----------------------------------------------------------------------------
def fit_model(params, X, y):
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    return model


def _macro_f1(y, pred):
    return float(f1_score(y, pred, average="macro", zero_division=0,
                          labels=np.arange(N_CLASSES)))


def metrics(y, probs):
    """Multiclass metrics from class probabilities: argmax accuracy/F1 + OVR AUC."""
    pred = probs.argmax(1)
    labels = np.arange(N_CLASSES)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": _macro_f1(y, pred),
        "per_class_f1": [float(v) for v in
                         f1_score(y, pred, average=None, zero_division=0, labels=labels)],
    }
    try:
        out["auc_ovr_macro"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
        out["per_class_auc_ovr"] = roc_auc_score(
            y, probs, multi_class="ovr", average=None).tolist()
    except ValueError:  # noqa: BLE001
        out["auc_ovr_macro"] = float("nan")
        out["per_class_auc_ovr"] = [float("nan")] * N_CLASSES
    return out


# ----------------------------------------------------------------------------
# One cell
# ----------------------------------------------------------------------------
def cell_name(dataset, lang, mode, condition):
    return f"flores12_{dataset}_{lang}_{mode}_{condition}_optuna10"


def run_cell(dataset, mode, condition, lang, ctx, state):
    classes = ctx["classes"]
    if dataset == "sent":
        src = ctx["sent_emb"][mode]
        idx = (ctx["sent"]["train_idx"], ctx["sent"]["val_idx"], ctx["sent"]["test_idx"])
        dims, k = (ctx["sent_dims"][mode] if condition == "prune50" else (None, 0))
    else:
        src = ctx["agg_emb"][mode]
        idx = (ctx["agg"]["train_idx"], ctx["agg"]["val_idx"], ctx["agg"]["test_idx"])
        dims, k = (ctx["agg_dims"][mode] if condition == "prune50" else (None, 0))
    train_idx, val_idx, test_idx = idx
    other = "it" if lang == "en" else "en"
    y = ctx["sent"]["y"] if dataset == "sent" else ctx["agg"]["y"]
    y_arr = y

    X = apply_prune(src[lang], dims)
    X_other = apply_prune(src[other], dims)
    tr = np.concatenate([train_idx, val_idx])
    name = cell_name(dataset, lang, mode, condition)

    log(f"===== CELL {name} (k={k}) train={len(train_idx)} val={len(val_idx)} "
        f"test={len(test_idx)} =====")
    log(f"[{name}] split labels: train={np.bincount(y_arr[train_idx], minlength=N_CLASSES).tolist()} "
        f"val={np.bincount(y_arr[val_idx], minlength=N_CLASSES).tolist()} "
        f"test={np.bincount(y_arr[test_idx], minlength=N_CLASSES).tolist()}")

    state["cell"] = name
    state["trial"] = None
    state["best_so_far"] = None
    state["cells"][name] = {"state": "running", "dataset": dataset, "lang": lang,
                            "mode": mode, "condition": condition, "k": k, "trial": None,
                            "best_so_far": None}
    write_status(state)

    def objective(trial):
        params = dict(BASE_PARAMS)
        params.update(_optuna_search_space(trial))
        model = fit_model(params, X[train_idx], y_arr[train_idx])
        m = metrics(y_arr[val_idx], model.predict_proba(X[val_idx]))
        trial.set_user_attr("auc_ovr_macro", m["auc_ovr_macro"])
        trial.set_user_attr("accuracy", m["accuracy"])
        trial.set_user_attr("macro_f1", m["macro_f1"])
        log(f"[{name}] trial {trial.number + 1}/{N_TRIALS} auc={m['auc_ovr_macro']:.4f} "
            f"acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")
        return m["auc_ovr_macro"]

    def callback(study, trial):
        state["trial"] = int(trial.number)
        state["best_so_far"] = float(study.best_value)
        state["cells"][name] = {"state": "running", "dataset": dataset, "lang": lang,
                                "mode": mode, "condition": condition, "k": k,
                                "trial": int(trial.number),
                                "best_so_far": float(study.best_value)}
        write_status(state)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])
    best_params = dict(study.best_params)
    log(f"[{name}] optuna best #{study.best_trial.number} val_auc={study.best_value:.4f} "
        f"params={best_params}")

    final_params = dict(BASE_PARAMS)
    final_params.update(best_params)
    final = fit_model(final_params, X[tr], y_arr[tr])

    self_m = metrics(y_arr[test_idx], final.predict_proba(X[test_idx]))
    trans_m = metrics(y_arr[test_idx], final.predict_proba(X_other[test_idx]))
    log(f"[{name}] test self: acc={self_m['accuracy']:.4f} macro={self_m['macro_f1']:.4f} "
        f"auc={self_m['auc_ovr_macro']:.4f} | transfer({other}): acc={trans_m['accuracy']:.4f} "
        f"macro={trans_m['macro_f1']:.4f} auc={trans_m['auc_ovr_macro']:.4f}")

    created = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta = {
        "dataset": dataset, "lang": lang, "mode": mode, "condition": condition,
        "k": int(k),
        "zeroed_dims": (dims.tolist() if dims is not None else []),
        "classes": classes,
        "best_params": best_params,
        "best_trial": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "val_auc": float(study.best_value),
        "val_macro_f1": float(study.best_trial.user_attrs.get("macro_f1", float("nan"))),
        "test_self_auc": self_m["auc_ovr_macro"],
        "test_transfer_auc": trans_m["auc_ovr_macro"],
        "test_self": self_m,
        "test_transfer": trans_m,
        "transfer_lang": other,
        "train_size": int(len(train_idx)), "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "n_trials": N_TRIALS,
        "created": created,
    }
    save_bundle(name, [final], meta, root=MODELS_ROOT)
    log(f"[{name}] saved bundle '{name}'")

    update_registry({
        "name": name, "dataset": dataset, "lang": lang, "mode": mode,
        "condition": condition, "k": int(k), "n_trials": N_TRIALS,
        "val_auc": float(study.best_value),
        "val_macro_f1": float(study.best_trial.user_attrs.get("macro_f1", float("nan"))),
        "test_self_auc": self_m["auc_ovr_macro"],
        "test_self_macro_f1": self_m["macro_f1"],
        "test_transfer_auc": trans_m["auc_ovr_macro"],
        "test_transfer_macro_f1": trans_m["macro_f1"],
        "created": created,
    })

    res = {"name": name, "k": int(k), "best_trial": int(study.best_trial.number),
           "best_value": float(study.best_value), "best_params": best_params,
           "self": self_m, "transfer": trans_m, "transfer_lang": other,
           "sizes": {"train": int(len(train_idx)), "val": int(len(val_idx)),
                     "test": int(len(test_idx))}}
    state["cells"][name] = {"state": "done", "dataset": dataset, "lang": lang,
                            "mode": mode, "condition": condition, "k": int(k),
                            "trial": N_TRIALS, "best_so_far": float(study.best_value),
                            "best_trial": int(study.best_trial.number),
                            "test_self_auc": self_m["auc_ovr_macro"],
                            "test_self_macro_f1": self_m["macro_f1"],
                            "test_transfer_auc": trans_m["auc_ovr_macro"],
                            "test_transfer_macro_f1": trans_m["macro_f1"]}
    state.setdefault("results", {})[name] = res
    write_status(state)
    return res


# ----------------------------------------------------------------------------
# Detached launch
# ----------------------------------------------------------------------------
def launch_detached(passthrough):
    vbs = os.path.join(REPO, "launch_hidden.vbs")
    cmd = subprocess.list2cmdline([sys.executable, os.path.abspath(__file__)] + passthrough)
    subprocess.Popen(["wscript", "//B", "//Nologo", vbs, cmd], cwd=REPO)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_shard(spec):
    """Parse an ``I/N`` shard spec; return ``(i, n)`` with ``0 <= i < n`` and ``n >= 1``."""
    try:
        i_s, n_s = str(spec).split("/")
        i, n = int(i_s), int(n_s)
    except Exception:
        raise SystemExit("--shard must be I/N (e.g. 0/4), got %r" % (spec,))
    if n < 1 or not (0 <= i < n):
        raise SystemExit("--shard out of range: %r (need 0 <= I < N, N >= 1)" % (spec,))
    return i, n


def parse_args(argv):
    ap = argparse.ArgumentParser(description="12-class FLORES topic driver (sentence + agg)")
    ap.add_argument("--launch", action="store_true",
                    help="spawn a detached copy via launch_hidden.vbs and exit")
    ap.add_argument("--trials", type=int, default=N_TRIALS,
                    help="Optuna trials per cell (default %d)" % N_TRIALS)
    ap.add_argument("--shard", default="0/1", metavar="I/N",
                    help="handle cells I of N round-robin; 0/1 (default) runs all 16 cells")
    ap.add_argument("--merge-registry", type=int, default=None, metavar="N",
                    help="merge models_registry_shard0..N-1.json into models_registry.json and exit")
    return ap.parse_args(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    if args.merge_registry is not None:
        n = merge_registries(args.merge_registry)
        print("merged %d entries into %s" % (n, REGISTRY_PATH))
        return 0
    if args.launch:
        launch_detached([a for a in argv if a != "--launch"])
        print("launched detached via launch_hidden.vbs")
        return 0

    shard_i, shard_n = parse_shard(args.shard)
    set_shard_paths(shard_i, shard_n)

    global N_TRIALS
    N_TRIALS = args.trials

    logf = _open_log()
    write_pid()
    t0 = time.time()
    log("train_flores12 starting")
    log(f"python pid={os.getpid()} trials={N_TRIALS} shard={shard_i}/{shard_n} "
        f"xgboost={xgb.__version__} optuna={optuna.__version__}")

    state = {
        "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_trials": N_TRIALS, "shard": [shard_i, shard_n],
        "cell": None, "trial": None, "best_so_far": None,
        "cells": {}, "results": {},
    }
    write_status(state)

    try:
        sent = prepare_sentence()
        classes = sent["classes"]
        log(f"sentence: {len(sent['cur'])} rows aligned to embeddings via {len(sent['kept'])} "
            f"kept indices; dropped git rows {sent['dropped']}")
        log(f"classes ({len(classes)}): {classes}")
        log(f"sentence split: train={len(sent['train_idx'])} val={len(sent['val_idx'])} "
            f"test={len(sent['test_idx'])}")

        agg = prepare_agg(sent["cur"], classes)
        log(f"agg: {len(agg['agg'])} rows; train={len(agg['train_idx'])} "
            f"val={len(agg['val_idx'])} test={len(agg['test_idx'])}")

        state["data"] = {
            "classes": classes,
            "sentence": {"rows": int(len(sent["cur"])), "kept": int(len(sent["kept"])),
                         "dropped_git_rows": sent["dropped"],
                         "train": int(len(sent["train_idx"])),
                         "val": int(len(sent["val_idx"])),
                         "test": int(len(sent["test_idx"]))},
            "agg": {"rows": int(len(agg["agg"])), "train": int(len(agg["train_idx"])),
                    "val": int(len(agg["val_idx"])), "test": int(len(agg["test_idx"]))},
            "prune_dims": {},
        }

        sent_emb = {m: load_sent_embeddings(m, sent["kept"], len(sent["git"]))
                    for m in MODES}
        agg_emb = {m: load_agg_embeddings(m, len(agg["agg"])) for m in MODES}
        sent_dims = {m: sent_prune_dims(m) for m in MODES}
        agg_dims = {m: agg_prune_dims(m, agg_emb[m]) for m in MODES}
        for m in MODES:
            state["data"]["prune_dims"][m] = {"sent_k": sent_dims[m][1], "agg_k": agg_dims[m][1]}
            log(f"prune50 {m}: sentence k={sent_dims[m][1]} dims, agg k={agg_dims[m][1]} dims")
        write_status(state)

        ctx = {"classes": classes, "sent": sent, "agg": agg, "sent_emb": sent_emb,
               "agg_emb": agg_emb, "sent_dims": sent_dims, "agg_dims": agg_dims}

        all_cells = [(d, m, c, l) for d in DATASETS for m in MODES
                     for c in CONDITIONS for l in TRAIN_LANGS]
        cells = [cell for j, cell in enumerate(all_cells) if j % shard_n == shard_i]
        state["cells_order"] = [cell_name(d, l, m, c) for (d, m, c, l) in cells]
        log(f"shard {shard_i}/{shard_n}: {len(cells)}/{len(all_cells)} cells "
            f"{state['cells_order']}")
        write_status(state)

        for dataset, mode, condition, lang in cells:
            try:
                run_cell(dataset, mode, condition, lang, ctx, state)
            except Exception as exc:
                nm = cell_name(dataset, lang, mode, condition)
                state["cells"][nm] = {"state": "failed", "error": repr(exc)}
                write_status(state)
                log(f"[{nm}] FAILED: {exc!r}")

        state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state["elapsed_s"] = round(time.time() - t0, 1)
        write_status(state)
        log(f"ALL DONE in {state['elapsed_s']}s")
    except Exception as exc:
        state["state"] = "failed"
        state["error"] = repr(exc)
        state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_status(state)
        log(f"FATAL: {exc!r}")
        raise
    finally:
        logf.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
