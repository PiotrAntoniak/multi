r"""Last-layer fine-tuning of EuroBERT (210m / 610m) for cross-lingual classification.

Trains ONLY the last transformer block (``model.layers[-1]``) + a linear head; the
embeddings and every lower layer stay frozen.  Pooling is the masked MEAN of the
last hidden state over the plain tokenization (content + the tokenizer's appended
``<|end_of_text|>`` token), exactly as the repo's ``extract_embeddings.py`` mean
mode.  Head = a single ``Linear(hidden, n_labels)``.

Protocol (EuroBERT paper / HF card, arXiv:2503.05500)
----------------------------------------------------
* AdamW, betas (0.9, 0.95), eps 1e-5, weight_decay 0.1, grad-clip 1.0.
* Linear LR schedule, warmup ratio 0.1; batch 32; early stopping, patience 1 epoch.
* Task LR for sequence classification (210m): 3.6e-5.  We grid {1e-5, 3.6e-5, 6e-5,
  1e-4} on the EN validation split and keep the best (default selection metric AUC).
* Single-label -> CrossEntropyLoss; binary -> BCEWithLogitsLoss.

Datasets
--------
* ``sent``  : flores200_en_it_de_fr.csv, 12-class topic.  train=dev(997), val/test =
  seed-0 halves of devtest(1009) -- IDENTICAL split to ``train_flores12.py``.
* ``agg``   : flores_agg.csv, 11-class (drops "business and economy" from every
  split, remaps to 0..10) -- IDENTICAL split to ``train_flores12.py``.
* ``toxic`` : data/toxi_clean/<lang>_{train,validation,test}.parquet, binary.

For every dataset the EN model trains on EN, and is evaluated self (EN test) and
cross-lingually on each target test.  Frozen+XGBoost references are (re)computed in
this script on the same frozen mean embeddings with a fixed, sensible XGBoost config,
so FT vs frozen+XGB is apples-to-apples.  In-language FT (train target -> test target)
gives the fine-tuned ceiling.

Artifacts: finetune_lastlayer.json / .md / .png next to this file; log/pid/status in
%TEMP%\opencode\finetune_lastlayer.*.  Detached launch via the repo's hidden launcher:
``python finetune_lastlayer.py --launch [args]``.  Resumable: completed runs are
skipped, results are flushed after every run.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import contextlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(2)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoModel, AutoTokenizer

import xgboost as xgb

REPO = Path(__file__).resolve().parent
TMP = Path(os.environ.get("TEMP", os.path.expanduser("~"))) / "opencode"
TMP.mkdir(parents=True, exist_ok=True)

SENT_CSV = REPO / "flores200_en_it_de_fr.csv"
AGG_CSV = REPO / "flores_agg.csv"
TOXI_DIR = REPO / "data" / "toxi_clean"
MODEL_ID = "EuroBERT/EuroBERT-210m"
TAG = ""
GIT_REF = "196bbd3:flores200_en_it_de_fr.csv"

FLORES_LANGS = ["en", "it", "de", "fr"]
TOXI_LANGS = ["en", "it", "de", "fr", "es"]
BATCH = 32
MAX_LEN = 512
PAD_ID = 128004  # EuroBERT <|pad|>
LR_GRID = [1e-5, 3.6e-5, 6e-5, 1e-4]
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
EPS = 1e-5
GRAD_CLIP = 1.0
SEED = 0
PATIENCE = 1  # epochs without val improvement before stopping

# (max_epochs, hard_step_cap) per dataset -- the paper's 10k-step budget is capped
# so the whole grid finishes in a bounded time on a single RTX 3060.
DATASET_CAPS = {"sent": (30, 3000), "agg": (30, 3000), "toxic": (2, 1200)}
DATASET_TARGETS = {"sent": ["en", "it", "de", "fr"], "agg": ["en", "it", "de", "fr"],
                   "toxic": ["en", "it", "de", "fr", "es"]}
DATASET_INLANG = {"sent": ["it", "de", "fr"], "agg": ["it", "de", "fr"],
                  "toxic": []}  # toxic in-language FT is too slow at 24k rows; the repo's
                               # native-XGB ceilings (REPO_REF) stand in for it.

# Repo-reported frozen+XGBoost references (for context in the markdown; the figure
# uses the in-script frozen+XGB numbers).
REPO_REF = {
    "sent": {"en_self_auc": 0.9186, "en_self_f1": 0.4817, "trans": {"it": (0.8470, 0.1751)}},
    "agg": {"en_self_auc": 0.9458, "en_self_f1": 0.5864, "trans": {"it": (0.8750, 0.2338)}},
    "toxic": {"en_self_auc": 0.9011, "native": {"it": 0.834, "de": 0.869, "fr": 0.866, "es": 0.797},
              "en_trans": {"it": 0.649, "de": 0.678, "fr": 0.776, "es": 0.639}},
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# bf16 autocast for the training compute (forward + backward), with fp32 master
# weights/optimizer state (standard AMP).  Disabled on CPU / non-bf16 GPUs.
USE_BF16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def amp_ctx():
    """torch.autocast context: bf16 on CUDA when enabled, else a no-op."""
    if USE_BF16 and DEVICE == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _paths():
    base = "finetune_lastlayer" + (f"_{TAG}" if TAG else "")
    return (TMP / f"{base}.log", TMP / f"{base}.pid", TMP / f"{base}_status.json")


LOG_PATH, PID_PATH, STATUS_PATH = _paths()


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

    def isatty(self):
        return False


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _open_log():
    f = open(LOG_PATH, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    return f


def write_pid(pid=None):
    PID_PATH.write_text(str(os.getpid() if pid is None else pid), encoding="utf-8")


def write_status(obj):
    obj["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = str(STATUS_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, STATUS_PATH)


# ----------------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------------
def _git_sentence_csv(ref=GIT_REF):
    proc = subprocess.run(["git", "-C", str(REPO), "show", ref], capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("git show %s failed: %s" % (ref, proc.stderr.decode("utf-8", "replace")))
    import io
    return pd.read_csv(io.BytesIO(proc.stdout))


def align_sentence_indices(cur, git):
    git_pos = {}
    for i, (split, rid, en) in enumerate(zip(git["split"], git["id"], git["en"])):
        key = (str(split), int(rid), str(en))
        git_pos[key] = i
    kept = np.empty(len(cur), dtype=int)
    for i, (split, rid, en) in enumerate(zip(cur["split"], cur["id"], cur["en"])):
        kept[i] = git_pos[(str(split), int(rid), str(en))]
    if not np.all(np.diff(kept) > 0):
        raise ValueError("current CSV not a subsequence of the git row order")
    return kept


def class_indices(df, classes):
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx[t] for t in df["topic"].astype(str)], dtype=int)


def build_flores_task(name):
    if name == "sent":
        df = pd.read_csv(SENT_CSV)
        kept = align_sentence_indices(df, _git_sentence_csv())
        classes = sorted(df["topic"].astype(str).unique())
        y = class_indices(df, classes)
        dev = np.where((df["split"] == "dev").values)[0]
        dt = np.where((df["split"] == "devtest").values)[0]
        perm = np.random.default_rng(0).permutation(dt)
        nv = len(perm) // 2
        splits = {"train": dev, "val": perm[:nv], "test": perm[nv:]}
        langs = FLORES_LANGS
        emb_root = REPO / "embeddings" / "mean"
        frozen = {L: np.load(emb_root / f"emb_{L}.npy")[kept] for L in langs}
    else:
        df = pd.read_csv(AGG_CSV)
        sent = pd.read_csv(SENT_CSV)
        url_split = {}
        for url, grp in sent.groupby("URL"):
            sp = set(grp["split"])
            url_split[url] = sp.pop()
        splitmap = df["URL"].map(url_split).fillna("devtest")
        classes12 = sorted(sent["topic"].astype(str).unique())
        y_full = class_indices(df, classes12)
        train_raw = np.where((splitmap == "dev").values)[0]
        dt = np.where((splitmap == "devtest").values)[0]
        perm = np.random.default_rng(0).permutation(dt)
        nv = len(perm) // 2
        val_raw, test_raw = perm[:nv], perm[nv:]
        excl = classes12.index("business and economy")

        def _drop(idx):
            return idx[y_full[idx] != excl]

        splits = {"train": _drop(train_raw), "val": _drop(val_raw), "test": _drop(test_raw)}
        kept_old = [i for i in range(len(classes12)) if i != excl]
        remap = {o: n for n, o in enumerate(kept_old)}
        mask = y_full != excl
        y = np.full(len(y_full), -1, dtype=int)
        y[mask] = [remap[int(v)] for v in y_full[mask]]
        classes = [classes12[i] for i in kept_old]
        langs = FLORES_LANGS
        emb_root = REPO / "embeddings_agg" / "mean"
        frozen = {L: np.load(emb_root / f"emb_{L}.npy") for L in langs}

    def get(lang, split):
        idx = splits[split]
        return list(df[lang].astype(str).values[idx]), y[idx].astype(int)

    return {"name": name, "n_classes": len(classes), "binary": False, "langs": langs,
            "classes": classes, "splits": splits, "get": get, "frozen": frozen,
            "sizes": {k: int(len(v)) for k, v in splits.items()}}


def build_toxic_task(name, emb_dir):
    cache = {}

    def _read(lang, split):
        key = (lang, split)
        if key not in cache:
            fn = {"train": "train", "val": "validation", "test": "test"}[split]
            d = pd.read_parquet(TOXI_DIR / f"{lang}_{fn}.parquet")
            cache[key] = (list(d["text"].astype(str).values), d["is_toxic"].astype(int).values)
        return cache[key]

    frozen = {}
    for L in TOXI_LANGS:
        for split in ("train", "val", "test"):
            fn = {"train": "train", "val": "validation", "test": "test"}[split]
            p = emb_dir / f"emb_{L}_{fn}.npy"
            if p.exists():
                frozen[(L, split)] = np.load(p)

    def get(lang, split):
        return _read(lang, split)

    return {"name": name, "n_classes": 2, "binary": True, "langs": TOXI_LANGS,
            "classes": ["non_toxic", "toxic"], "get": get, "frozen": frozen,
            "sizes": {L: {s: len(_read(L, s)[0]) for s in ("train", "val", "test")} for L in TOXI_LANGS}}


# ----------------------------------------------------------------------------
# Tokenization / pooling / metrics
# ----------------------------------------------------------------------------
def build_token_cache(tok, texts):
    enc = tok(list(texts), add_special_tokens=True, truncation=True, max_length=MAX_LEN,
              padding=False)["input_ids"]
    return [torch.tensor(x, dtype=torch.long) for x in enc]


def collate(seqs, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


def pool(model, ids, mask):
    h = model(input_ids=ids, attention_mask=mask).last_hidden_state
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1)


def metrics(y, probs, binary, n_classes):
    y = np.asarray(y)
    if binary:
        pred = (probs >= 0.5).astype(int)
        return {"accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "auc": float(roc_auc_score(y, probs))}
    pred = probs.argmax(1)
    labels = np.arange(n_classes)
    out = {"accuracy": float(accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0, labels=labels))}
    try:
        out["auc"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        out["auc"] = float("nan")
    return out


@torch.no_grad()
def predict(tok, model, head, texts, binary):
    model.eval()
    cache = build_token_cache(tok, texts)
    out = []
    for i in range(0, len(cache), BATCH):
        ids, mask = collate(cache[i:i + BATCH], DEVICE)
        with amp_ctx():
            logits = head(pool(model, ids, mask)).float()  # fp32 probs
        p = torch.sigmoid(logits).squeeze(-1) if binary else torch.softmax(logits, -1)
        out.append(p.float().cpu().numpy())
    return np.concatenate(out, 0)


# ----------------------------------------------------------------------------
# Last-layer fine-tuning
# ----------------------------------------------------------------------------
def reset_last_layer(model, layer_init):
    model.layers[-1].load_state_dict({k: v.clone() for k, v in layer_init.items()})


def reset_head(head, seed):
    g = torch.Generator(device=head.weight.device).manual_seed(seed)
    for p in head.parameters():
        if p.dim() >= 2:
            torch.nn.init.xavier_uniform_(p, generator=g)
        else:
            torch.nn.init.zeros_(p)


def head_state(head):
    return {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}


def layer_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.layers[-1].state_dict().items()}


def load_states(model, head, ls, hs):
    model.layers[-1].load_state_dict({k: v.to(DEVICE) for k, v in ls.items()})
    head.load_state_dict({k: v.to(DEVICE) for k, v in hs.items()})


def make_scheduler(opt, total_steps):
    warm = max(1, int(WARMUP_RATIO * total_steps))

    def fn(step):
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def fit(tok, model, head, layer_init, tr_texts, tr_y, val_texts, val_y, n_classes, binary,
        lr, max_epochs, hard_cap, seed, state, tag, do_val=True):
    """Train last layer + head.  Early stop on val; return (best layers, head state, info)."""
    reset_last_layer(model, layer_init)
    reset_head(head, seed)
    params = list(model.layers[-1].parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)
    tr_ids = build_token_cache(tok, tr_texts)
    n = len(tr_ids)
    spe = max(1, math.ceil(n / BATCH))
    total = min(hard_cap, spe * max_epochs)
    sched = make_scheduler(opt, total)
    rng = np.random.default_rng(seed)
    lossfn = torch.nn.BCEWithLogitsLoss() if binary else torch.nn.CrossEntropyLoss()

    best = {"metric": -1.0, "layers": None, "head": None, "epoch": 0, "steps": 0,
            "val": None}
    no_improve = 0
    gstep = 0
    val_hist = []
    for epoch in range(1, max_epochs + 1):
        if gstep >= total:
            break
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, BATCH):
            if gstep >= total:
                break
            idx = perm[i:i + BATCH]
            ids, mask = collate([tr_ids[j] for j in idx], DEVICE)
            yb = torch.tensor(tr_y[idx], dtype=torch.float if binary else torch.long, device=DEVICE)
            opt.zero_grad()
            with amp_ctx():
                logits = head(pool(model, ids, mask)).float()  # fp32 logits for the loss
                loss = lossfn(logits.squeeze(-1) if binary else logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            sched.step()
            gstep += 1
        # ---- epoch-end validation
        if do_val:
            probs = predict(tok, model, head, val_texts, binary)
            m = metrics(val_y, probs, binary, n_classes)
            vm = m["auc"] if np.isfinite(m["auc"]) else m["macro_f1"]
            val_hist.append({"epoch": epoch, "step": gstep, **m})
            log(f"    [{tag}] ep{epoch} step{gstep}/{total} val_auc={m['auc']:.4f} "
                f"val_f1={m['macro_f1']:.4f} val_acc={m['accuracy']:.4f}")
            state["trial"] = {"tag": tag, "epoch": epoch, "step": gstep, "val": m}
            write_status(state)
            if vm > best["metric"]:
                best = {"metric": float(vm), "layers": layer_state(model),
                        "head": head_state(head), "epoch": epoch, "steps": gstep, "val": m}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break
    if not do_val:
        best = {"metric": float("nan"), "layers": layer_state(model), "head": head_state(head),
                "epoch": max_epochs, "steps": gstep, "val": None}
    load_states(model, head, best["layers"], best["head"])
    best["val_hist"] = val_hist
    return best


# ----------------------------------------------------------------------------
# Frozen + XGBoost baseline
# ----------------------------------------------------------------------------
XGB_DEV = "cuda" if torch.cuda.is_available() else "cpu"
XGB_MC = dict(objective="multi:softprob", tree_method="hist", device=XGB_DEV, n_jobs=2,
              max_bin=64, max_depth=6, learning_rate=0.1, n_estimators=400, subsample=0.8,
              colsample_bytree=0.6, reg_lambda=1.0, random_state=SEED)
XGB_BN = dict(objective="binary:logistic", tree_method="hist", device=XGB_DEV, n_jobs=2,
              max_bin=64, max_depth=8, learning_rate=0.1, n_estimators=400, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED)


def frozen_xgb(task, train_lang, targets):
    """Train on frozen mean embeddings of train_lang's train (+val), eval on targets' test."""
    res = {}
    if task["name"] == "toxic":
        Xtr, ytr = task["frozen"][(train_lang, "train")], task["get"](train_lang, "train")[1]
        Xva = task["frozen"].get((train_lang, "val"))
        clf = xgb.XGBClassifier(**XGB_BN)
        if Xva is not None:
            clf.set_params(early_stopping_rounds=None)
        clf.fit(Xtr, ytr)
        for t in targets:
            Xte = task["frozen"].get((t, "test"))
            if Xte is None:
                continue
            probs = clf.predict_proba(Xte)[:, 1]
            res[t] = metrics(task["get"](t, "test")[1], probs, True, 2)
    else:
        idx = np.concatenate([task["splits"]["train"], task["splits"]["val"]])
        Xtr = task["frozen"][train_lang][idx]
        ytr = np.concatenate([task["get"](train_lang, "train")[1],
                              task["get"](train_lang, "val")[1]])
        clf = xgb.XGBClassifier(num_class=task["n_classes"], **XGB_MC)
        clf.fit(Xtr, ytr)
        for t in targets:
            Xte = task["frozen"][t][task["splits"]["test"]]
            probs = clf.predict_proba(Xte)
            res[t] = metrics(task["get"](t, "test")[1], probs, False, task["n_classes"])
    return res


# ----------------------------------------------------------------------------
# One dataset
# ----------------------------------------------------------------------------
def run_dataset(task, tok, model, head, layer_init, state, args):
    name = task["name"]
    max_epochs, hard_cap = DATASET_CAPS[name]
    targets = DATASET_TARGETS[name]
    inlang = DATASET_INLANG[name]
    out = {"n_classes": task["n_classes"], "binary": task["binary"], "sizes": task["sizes"],
           "ft": {}, "frozen_xgb": {}}
    log(f"===== DATASET {name}: n_classes={task['n_classes']} sizes={task['sizes']} =====")

    # ---- frozen + XGBoost baseline
    if "frozen" not in state:
        state["frozen"] = {}
    if name not in state["frozen"]:
        log(f"[{name}] frozen+XGB baseline ...")
        fx = frozen_xgb(task, "en", targets)
        inl = {}
        for L in inlang:
            inl[L] = frozen_xgb(task, L, [L])[L]
        state["frozen"][name] = {"en": fx, "inlang": inl}
        log(f"[{name}] frozen+XGB en_self auc={fx['en']['auc']:.4f} f1={fx['en']['macro_f1']:.4f} "
            f"| trans " + " ".join(f"{t}={fx[t]['auc']:.3f}" for t in targets if t != "en"))
        write_status(state)
        flush_json(state)
    out["frozen_xgb"] = state["frozen"][name]

    # ---- LR grid on EN (train on EN train, early stop on EN val)
    te_tr, te_y = task["get"]("en", "train")
    te_va, te_yv = task["get"]("en", "val")
    if name not in state["lr_grid"]:
        grid = {}
        for lr in LR_GRID:
            tag = f"{name}|EN|grid|lr={lr:g}"
            log(f"[{name}] LR grid {lr:g} ...")
            b = fit(tok, model, head, layer_init, te_tr, te_y, te_va, te_yv,
                    task["n_classes"], task["binary"], lr, max_epochs, hard_cap, SEED,
                    state, tag)
            grid[f"{lr:g}"] = {"lr": lr, "best_epoch": b["epoch"], "best_steps": b["steps"],
                               "val": b["val"], "sel": b["metric"]}
            log(f"[{name}] LR {lr:g}: best_ep={b['epoch']} steps={b['steps']} "
                f"val_auc={b['val']['auc']:.4f} val_f1={b['val']['macro_f1']:.4f}")
            write_status(state)
            flush_json(state)
        state["lr_grid"][name] = grid
        flush_json(state)
    grid = state["lr_grid"][name]
    best_key = max(grid, key=lambda k: grid[k]["sel"] if np.isfinite(grid[k]["sel"]) else -1)
    best = grid[best_key]
    lr = best["lr"]
    out["ft"]["lr_chosen"] = lr
    out["ft"]["lr_grid"] = {k: {"val_auc": v["val"]["auc"], "val_f1": v["val"]["macro_f1"],
                                "best_epoch": v["best_epoch"], "best_steps": v["best_steps"]}
                            for k, v in grid.items()}
    log(f"[{name}] chosen LR = {lr:g} (val_auc={best['val']['auc']:.4f}, "
        f"best_epoch={best['best_epoch']})")

    # ---- final EN model: train on EN train+val for best_steps, eval self + transfer
    if name not in state["final"]:
        tr_texts = te_tr + te_va
        tr_y = np.concatenate([te_y, te_yv])
        tag = f"{name}|EN|final|lr={lr:g}"
        log(f"[{name}] final EN on train+val={len(tr_texts)} for {best['best_steps']} steps ...")
        # schedule total = best_steps via max_epochs trick: run exactly best_steps
        spe = max(1, math.ceil(len(tr_texts) / BATCH))
        ep = max(1, math.ceil(best["best_steps"] / spe))
        b = fit(tok, model, head, layer_init, tr_texts, tr_y, None, None,
                task["n_classes"], task["binary"], lr, ep, best["best_steps"], SEED,
                state, tag, do_val=False)
        en = {}
        for t in targets:
            txt, yt = task["get"](t, "test")
            probs = predict(tok, model, head, txt, task["binary"])
            en[t] = metrics(yt, probs, task["binary"], task["n_classes"])
        en["_train_steps"] = b["steps"]
        state["final"][name] = en
        log(f"[{name}] FT EN self auc={en['en']['auc']:.4f} f1={en['en']['macro_f1']:.4f} | "
            + " ".join(f"{t}={en[t]['auc']:.3f}" for t in targets if t != "en"))
        write_status(state)
        flush_json(state)
    out["ft"]["en"] = state["final"][name]

    # ---- in-language FT (ceiling), chosen LR
    if name not in state["inlang_ft"]:
        state["inlang_ft"][name] = {}
    for L in inlang:
        if L in state["inlang_ft"][name]:
            continue
        tr, ytr = task["get"](L, "train")
        va, yva = task["get"](L, "val")
        tag = f"{name}|{L}|inlang|lr={lr:g}"
        log(f"[{name}] in-language FT {L} ...")
        b = fit(tok, model, head, layer_init, tr, ytr, va, yva, task["n_classes"],
                task["binary"], lr, max_epochs, hard_cap, SEED, state, tag)
        txt, yt = task["get"](L, "test")
        probs = predict(tok, model, head, txt, task["binary"])
        m = metrics(yt, probs, task["binary"], task["n_classes"])
        m["best_epoch"] = b["epoch"]
        m["val_auc"] = b["val"]["auc"] if b["val"] else None
        state["inlang_ft"][name][L] = m
        log(f"[{name}] in-language FT {L}: test auc={m['auc']:.4f} f1={m['macro_f1']:.4f} "
            f"acc={m['accuracy']:.4f} best_ep={b['epoch']}")
        write_status(state)
        flush_json(state)
    out["ft"]["inlang"] = state["inlang_ft"][name]
    return out


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------
JSON_PATH = REPO / "finetune_lastlayer.json"


def flush_json(state):
    payload = {
        "meta": {"model_id": MODEL_ID, "device": DEVICE, "seeds": [SEED], "lr_grid": LR_GRID,
                 "batch": BATCH, "max_len": MAX_LEN, "weight_decay": WEIGHT_DECAY,
                 "betas": list(BETAS), "eps": EPS, "grad_clip": GRAD_CLIP,
                 "warmup_ratio": WARMUP_RATIO, "patience_epochs": PATIENCE,
                 "dataset_caps": {k: {"max_epochs": v[0], "hard_step_cap": v[1]}
                                  for k, v in DATASET_CAPS.items()},
                 "pooling": "masked mean of last hidden state over plain tokenization",
                 "features": ("raw mean-pooled last hidden state; NO perlang_z / L2 / z-score / "
                              "pc / inlp / D shift; frozen+XGB also raw"),
                 "trainable": "model.layers[-1] + head (embeddings & lower layers frozen)",
                 "selection_metric": "validation AUC (OVR macro / binary)",
                 "precision": ("bf16 autocast (compute) + fp32 master weights"
                               if USE_BF16 else "fp32"),
                 "created": time.strftime("%Y-%m-%d %H:%M:%S")},
        "frozen_xgb": state["frozen"], "ft": {"lr_grid": state["lr_grid"],
                                              "final_en": state["final"],
                                              "inlang": state["inlang_ft"]},
        "repo_reference": REPO_REF,
    }
    tmp = str(JSON_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    os.replace(tmp, JSON_PATH)


# ----------------------------------------------------------------------------
# Reporting (md + figure)
# ----------------------------------------------------------------------------
def _chosen(grid):
    key = max(grid, key=lambda k: grid[k]["sel"] if np.isfinite(grid[k]["sel"]) else -1)
    return key, grid[key]


def write_report(state, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [d for d in ("sent", "agg", "toxic") if d in state["final"]]
    md = ["# Last-layer fine-tuning of EuroBERT vs frozen+XGBoost\n"]
    md.append(f"Model `{MODEL_ID}` (only `model.layers[-1]` + a linear head trained; embeddings "
              f"and all lower layers frozen); masked-mean pooling of the last hidden state; "
              f"AdamW(betas=(0.9,0.95), eps=1e-5, weight_decay=0.1), grad-clip 1.0, linear "
              f"schedule with warmup ratio 0.1, batch 32, patience 1 epoch. "
              + ("**Training compute runs under bf16 autocast with fp32 master weights "
                 "(standard AMP); the cross-entropy / BCE loss is computed in fp32.**\n"
                 if USE_BF16 else "Training compute is fp32.\n"))
    md.append("**Features are RAW**: the mean-pooled last hidden state goes straight into the head "
              "with NO per-language normalization, L2, z-score, PC/INLP projection or language-shift "
              "`D`, and every language's test split is scored on its raw features. The frozen+XGBoost "
              "baseline likewise uses raw frozen mean embeddings, so the comparison isolates the "
              "effect of fine-tuning.\n")
    md.append(f"LR grid {LR_GRID} selected on the EN validation split by validation AUC.\n")
    md.append("Each cell is test-set **AUC** (macro-F1 in the table). Frozen+XGBoost is retrained "
              "in-script on the same frozen mean embeddings (fixed config) for apples-to-apples; "
              "in-language FT is the fine-tuned ceiling.\n")

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.8), squeeze=False)

    for ax, d in zip(axes[0], datasets):
        tg = DATASET_TARGETS[d]
        fx = state["frozen"][d]
        en = state["final"][d]
        _, chosen = _chosen(state["lr_grid"][d])
        lr = chosen["lr"]
        keys = ["en"] + [t for t in tg if t != "en"]
        groups = ["EN-self"] + [f"EN->{t}" for t in keys[1:]]
        ft_vals = [en[k]["auc"] for k in keys]
        xg_vals = [fx["en"][k]["auc"] for k in keys]
        x = np.arange(len(groups))
        w = 0.38
        ax.bar(x - w / 2, ft_vals, w, color="#2563eb")
        ax.bar(x + w / 2, xg_vals, w, color="#f59e0b")
        for j, k in enumerate(keys):
            il = state["inlang_ft"].get(d, {}).get(k)
            if il:
                ax.plot(j, il["auc"], marker="D", ms=7, color="#16a34a", zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9)
        ax.set_ylabel("AUC")
        ax.set_ylim(0.4, 1.03)
        ax.set_title(f"{d}  (LR={lr:g}, best epoch {chosen['best_epoch']})", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if d == datasets[0]:
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            handles = [Patch(color="#2563eb", label="last-layer FT"),
                       Patch(color="#f59e0b", label="frozen+XGB"),
                       Line2D([0], [0], marker="D", color="#16a34a", ls="", label="in-lang FT")]
            ax.legend(handles=handles, fontsize=8, loc="lower right")
    fig.suptitle(f"EuroBERT {MODEL_ID.split('/')[-1]}: last-layer FT vs frozen+XGB (test AUC)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = REPO / "finetune_lastlayer.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)

    for d in datasets:
        tg = DATASET_TARGETS[d]
        keys = ["en"] + [t for t in tg if t != "en"]
        _, chosen = _chosen(state["lr_grid"][d])
        md.append(f"\n## {d}\n")
        md.append(f"**Chosen LR = {chosen['lr']:g}** (val AUC {chosen['val']['auc']:.4f}, "
                  f"best epoch {chosen['best_epoch']}, steps {chosen['best_steps']}). "
                  "LR grid: " + ", ".join(
                      f"{k}: val_auc={v['val']['auc']:.3f}/f1={v['val']['macro_f1']:.3f}"
                      for k, v in state["lr_grid"][d].items()) + ".\n")
        md.append("| eval | FT AUC | FT F1 | XGB AUC | XGB F1 | in-lang FT AUC | in-lang FT F1 |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        for k in keys:
            il = state["inlang_ft"].get(d, {}).get(k, {})
            il_auc = f"{il['auc']:.4f}" if il else "-"
            il_f1 = f"{il['macro_f1']:.4f}" if il else "-"
            label = "EN-self" if k == "en" else "EN->" + k
            md.append(f"| {label} | {state['final'][d][k]['auc']:.4f} | "
                      f"{state['final'][d][k]['macro_f1']:.4f} | "
                      f"{state['frozen'][d]['en'][k]['auc']:.4f} | "
                      f"{state['frozen'][d]['en'][k]['macro_f1']:.4f} | {il_auc} | {il_f1} |")
    md.append("\n## Files\n")
    md.append(f"- figure: `{png.name}`\n- data: `finetune_lastlayer.json`\n")
    (REPO / "finetune_lastlayer.md").write_text("\n".join(md), encoding="utf-8")
    return png


# ----------------------------------------------------------------------------
# Detached launch
# ----------------------------------------------------------------------------
def launch_detached(passthrough):
    vbs = REPO / "launch_hidden.vbs"
    # The VBS redirects the process stdout to its own file; the script writes its
    # structured log separately, so the two never contend for the same handle.
    base = "finetune_lastlayer" + (f"_{TAG}" if TAG else "")
    stdout_log = str(TMP / f"{base}.stdout.log")
    cmd = subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve())] + passthrough)
    subprocess.Popen(["wscript", "//B", "//Nologo", str(vbs), cmd, stdout_log, str(REPO)],
                     cwd=str(REPO))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args(argv):
    ap = argparse.ArgumentParser(description="EuroBERT last-layer fine-tuning")
    ap.add_argument("--launch", action="store_true", help="spawn detached via launch_hidden.vbs")
    ap.add_argument("--datasets", default="sent,agg,toxic")
    ap.add_argument("--model-id", default="EuroBERT/EuroBERT-210m")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--resume", action="store_true",
                    help="load completed runs from an existing finetune_lastlayer.json")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override per-dataset max_epochs (applies to every dataset run)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override per-dataset hard step cap (applies to every dataset run)")
    ap.add_argument("--caps", default="",
                    help="per-dataset caps, e.g. 'sent=30/3000,agg=30/3000,toxic=1/400'")
    ap.add_argument("--no-bf16", action="store_true",
                    help="disable bf16 autocast (run training compute in fp32)")
    ap.add_argument("--regen-report", action="store_true",
                    help="rebuild md/png from the existing JSON (no training)")
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate plumbing")
    return ap.parse_args(argv)


def apply_caps(spec):
    """Update DATASET_CAPS from a 'ds=epochs/steps,...' string."""
    global DATASET_CAPS
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        ds, val = part.split("=")
        e, c = val.split("/")
        if ds.strip() not in DATASET_CAPS:
            raise SystemExit("unknown dataset in --caps: %r" % ds)
        DATASET_CAPS[ds.strip()] = (int(e), int(c))


def main(argv=None):
    try:
        from gpu_limit import limit_gpu; limit_gpu()
    except Exception:
        pass
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    global MODEL_ID, TAG, SEED, DATASET_CAPS, LOG_PATH, PID_PATH, STATUS_PATH, USE_BF16
    MODEL_ID = args.model_id
    TAG = args.tag or ""
    LOG_PATH, PID_PATH, STATUS_PATH = _paths()
    if args.no_bf16:
        USE_BF16 = False
    if args.launch:
        launch_detached([a for a in argv if a != "--launch"])
        print("launched detached via launch_hidden.vbs")
        return 0

    if args.regen_report:
        d = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        MODEL_ID = d.get("meta", {}).get("model_id", MODEL_ID)
        USE_BF16 = "bf16" in str(d.get("meta", {}).get("precision", ""))
        st = {"frozen": d.get("frozen_xgb", {}),
              "lr_grid": d.get("ft", {}).get("lr_grid", {}),
              "final": d.get("ft", {}).get("final_en", {}),
              "inlang_ft": d.get("ft", {}).get("inlang", {})}
        png = write_report(st, args)
        print("regenerated report:", png)
        return 0

    SEED = int(args.seeds.split(",")[0])

    if args.smoke:
        DATASET_CAPS = {"sent": (2, 10), "agg": (2, 10), "toxic": (1, 5)}
    if args.max_epochs is not None or args.max_steps is not None:
        md = dict(DATASET_CAPS)
        for k, (e, c) in md.items():
            DATASET_CAPS[k] = (args.max_epochs if args.max_epochs is not None else e,
                               args.max_steps if args.max_steps is not None else c)
    if args.caps:
        apply_caps(args.caps)

    logf = _open_log()
    write_pid()
    t0 = time.time()
    log(f"finetune_lastlayer start: pid={os.getpid()} device={DEVICE} model={MODEL_ID} "
        f"tag={TAG!r} datasets={args.datasets} lr_grid={LR_GRID}")
    log(f"AMP: precision={'bf16 autocast (fp32 master weights)' if USE_BF16 else 'fp32'} "
        f"bf16_supported={torch.cuda.is_bf16_supported() if DEVICE == 'cuda' else False} "
        f"caps={DATASET_CAPS}")
    log("Features: RAW mean-pooled last hidden state (NO perlang_z / L2 / z-score / pc / inlp / "
        "D shift); every language's test rows are scored on raw features. The frozen+XGBoost "
        "baseline also uses RAW frozen mean embeddings. The only cross-lingual mechanism is FT.")

    state = {"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
             "frozen": {}, "lr_grid": {}, "final": {}, "inlang_ft": {}, "trial": None}
    if args.resume and JSON_PATH.exists():
        try:
            old = json.loads(JSON_PATH.read_text(encoding="utf-8"))
            state["frozen"] = old.get("frozen_xgb", {}) or {}
            ft = old.get("ft", {}) or {}
            state["lr_grid"] = ft.get("lr_grid", {}) or {}
            state["final"] = ft.get("final_en", {}) or {}
            state["inlang_ft"] = ft.get("inlang", {}) or {}
            log(f"resumed from {JSON_PATH.name}: frozen={list(state['frozen'])} "
                f"lr_grid={list(state['lr_grid'])} final={list(state['final'])} "
                f"inlang={ {k: list(v) for k, v in state['inlang_ft'].items()} }")
        except Exception as exc:  # noqa: BLE001
            log(f"resume load failed ({exc!r}); starting fresh")
    write_status(state)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.layers[-1].parameters():
        p.requires_grad_(True)
    model.norm.requires_grad_(False)
    hid = model.config.hidden_size
    head = torch.nn.Linear(hid, 12).to(DEVICE)
    layer_init = {k: v.detach().clone() for k, v in model.layers[-1].state_dict().items()}
    log(f"model ready: hidden={hid} layers={len(model.layers)} trainable_last={sum(p.numel() for p in model.layers[-1].parameters())}")

    tobj = tok("a quick brown fox", add_special_tokens=True)["input_ids"]
    log(f"tokenizer sanity: {tobj} (pad_id={PAD_ID})")

    try:
        for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
            if name in ("sent", "agg"):
                task = build_flores_task(name)
            else:
                emb = REPO / ("toxi_emb_610" if "610" in MODEL_ID else "toxi_emb_210")
                task = build_toxic_task(name, emb)
            # resize head per task
            ncl = task["n_classes"] if not task["binary"] else 1
            head = torch.nn.Linear(hid, ncl).to(DEVICE)
            state.setdefault("done", [])
            run_dataset(task, tok, model, head, layer_init, state, args)
            state["done"].append(name)
            write_status(state)
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state["elapsed_s"] = round(time.time() - t0, 1)
        flush_json(state)
        write_status(state)
        png = write_report(state, args)
        log(f"ALL DONE in {state['elapsed_s']}s; figure={png}")
    except Exception as exc:
        import traceback
        log("FATAL: " + repr(exc))
        log(traceback.format_exc())
        state["error"] = repr(exc)
        state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_status(state)
        raise
    finally:
        logf.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
