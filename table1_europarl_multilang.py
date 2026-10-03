"""Europarl v7 multilingual Table 1: exact-translation retrieval for de/es/fr/nl/pt-en.

For each target language xx in {de,es,fr,nl,pt}, mirroring the group's Table 1
protocol (dev-fitted language-shift vector D) on the Europarl v7 xx-en corpus:

  sample : seeded rng(0); skip ~1,000,000 line-aligned lines; scan; skip pairs
           empty on either side; dedupe exact (en, xx) pairs; collect 5,000
           unique; shuffle; dev = first 1,000 / devtest = last 4,000; write
           `europarl_{xx}_en_5k.csv` (utf-8; split,id,en,{xx}).
           If a window yields < 5,000 unique pairs, retry from skip 500k then 0.
  embed  : EuroBERT-210m native fp32.  SINGLE tokenization pass (plain ids,
           truncated to 511); LENGTH-SORTED batch order; batch-local padding
           (pad to the batch max only); batch 128 (fallback 64 on OOM); two
           forwards per batch (plain -> mean/lead/eos; prepended id 128000 ->
           bos); `.float()` right after each forward.  Saves
           `europarl_{xx}_en_5k_embeddings/<mode>/emb_{xx,en}.npy`, float32
           (5000, 768), finite-checked.
  table  : D = mean over dev of (emb_a - emb_b); directions en->xx and xx->en;
           all 4,000 devtest queries vs all 4,000 keys; per-mode markdown table
           (raw -> +D; top-1 / MRR / mean rank).  Regenerates the combined
           `europarl_multilang_table1.md` after each language, including one
           summary line per language (mean-mode +D top-1).

Resumable: stage flags + metrics live in `europarl_multilang_progress.json`; a
stage whose outputs already exist is skipped.  Logs to
`europarl_multilang_run.log`; appends stage lines to
`europarl_multilang_status.txt`; records its PID in `europarl_multilang.pid`.

Usage:
  python table1_europarl_multilang.py smoke      # no model: reproduce it-en 4k
  python table1_europarl_multilang.py all        # full pipeline, 5 languages
  python table1_europarl_multilang.py pair --lang de
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LANGS = ["de", "es", "fr", "nl", "pt"]
DEV_N, TEST_N = 1000, 4000
SAMPLE_N = DEV_N + TEST_N
CAP = SAMPLE_N
MIN_UNIQUE = SAMPLE_N
SKIPS = [1_000_000, 500_000, 0]

POOLS = ["mean", "lead", "bos", "eos"]
MODEL_ID = "EuroBERT/EuroBERT-210m"
BATCH, FALLBACK_BATCH = 128, 64
MAX_LEN = 512
LEAD_FALLBACK = 128000
EPS = 1e-12

CSV_FMT = "europarl_{xx}_en_5k.csv"
EMB_FMT = "europarl_{xx}_en_5k_embeddings"
PROGRESS_FILE = ROOT / "europarl_multilang_progress.json"
SUMMARY = ROOT / "europarl_multilang_table1.md"
RUN_LOG = ROOT / "europarl_multilang_run.log"
STATUS_FILE = ROOT / "europarl_multilang_status.txt"
PID_FILE = ROOT / "europarl_multilang.pid"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

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

    def fileno(self):
        for st in self.streams:
            if hasattr(st, "fileno"):
                return st.fileno()
        return -1

    def __getattr__(self, name):
        streams = self.__dict__.get("streams", ())
        for st in streams:
            if hasattr(st, name):
                return getattr(st, name)
        raise AttributeError(name)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def status(msg):
    with STATUS_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{now()}] {msg}\n")


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def detect_encoding(path):
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                while f.read(1 << 20):
                    pass
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def locate_corpus_paths(d, xx, en="en"):
    xp = ep = None
    if d.is_dir():
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            if p.name.endswith(f".{xx}"):
                xp = p
            elif p.name.endswith(f".{en}"):
                ep = p
    if not xp or not ep:
        raise FileNotFoundError(f"{xx}/{en} corpus sides not found under {d}")
    return xp, ep


def locate_corpus(xx):
    return locate_corpus_paths(ROOT / f"europarl_{xx}_en", xx, "en")


def collect_pairs(xx_path, en_path, skip, cap):
    """Skip `skip` aligned lines, then collect unique non-empty (en, xx) pairs."""
    enc_xx = detect_encoding(xx_path)
    enc_en = detect_encoding(en_path)
    pool, seen = [], set()
    with xx_path.open(encoding=enc_xx, errors="replace") as fx, \
            en_path.open(encoding=enc_en, errors="replace") as fe:
        for _ in range(skip):
            if next(fx, None) is None or next(fe, None) is None:
                break
        scanned = 0
        for lx, le in zip(fx, fe):
            scanned += 1
            a, b = le.strip(), lx.strip()
            if not a or not b:
                continue
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            pool.append(key)
            if len(pool) >= cap:
                break
    return pool, scanned


def sample(xx):
    xx_path, en_path = locate_corpus(xx)
    pool = None
    for skip in SKIPS:
        t = time.time()
        pool, scanned = collect_pairs(xx_path, en_path, skip, CAP)
        print(f"[{xx}-en] window skip={skip:,}: {scanned:,} pairs scanned -> "
              f"{len(pool):,} unique ({time.time() - t:.1f}s)", flush=True)
        if len(pool) >= MIN_UNIQUE or skip == SKIPS[-1]:
            break
    if len(pool) < SAMPLE_N:
        raise RuntimeError(
            f"{xx}-en: only {len(pool)} unique pairs, need {SAMPLE_N}")

    order = np.arange(len(pool))
    np.random.seed(0)
    np.random.shuffle(order)
    sel = order[:SAMPLE_N]
    splits = ["dev"] * DEV_N + ["devtest"] * TEST_N
    rows = [(splits[k], k, pool[i][0], pool[i][1])
            for k, i in enumerate(sel)]

    path = ROOT / CSV_FMT.format(xx=xx)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "id", "en", xx])
        w.writerows(rows)
    print(f"[{xx}-en] wrote {path.name}: {len(rows)} rows "
          f"(dev={DEV_N}, devtest={TEST_N}) from {len(pool)} unique pairs, "
          f"seed=0, window {skip + 1:,}-{skip + scanned:,}", flush=True)
    return path


def read_rows(path, col):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or col not in reader.fieldnames:
            raise ValueError(f"{path}: missing column '{col}' "
                             f"(have {reader.fieldnames})")
        return list(reader)


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        vocab = set(tokenizer.get_vocab().values())
        lead = tokenizer.bos_token_id
        if lead is None or lead not in vocab:
            lead = LEAD_FALLBACK
        if lead not in vocab:
            raise RuntimeError(f"leading special id {lead} not in vocab")
        model = (AutoModel.from_pretrained(MODEL_ID)
                 .to(device).to(torch.float32).eval())
        print(f"[model] {MODEL_ID} on {device}, fp32, hidden="
              f"{model.config.hidden_size}, lead id={lead} "
              f"(auto bos={tokenizer.bos_token_id})", flush=True)
        _MODEL = (tokenizer, model, device, int(lead))
    return _MODEL


def embed_texts(texts, tokenizer, model, device, lead, batch_size):
    """Four poolings of the last hidden state: length-sorted, batch-local pad.

    Single tokenization pass; order restored by scattering per-batch rows back
    to their original indices.  Plain forward -> mean/lead/eos; prepended
    forward (lead id 128000) -> bos.
    """
    import torch

    torch.set_num_threads(2)
    n = len(texts)
    hidden = int(model.config.hidden_size)
    enc = tokenizer(texts, add_special_tokens=True, truncation=True,
                    max_length=MAX_LEN - 1)["input_ids"]
    lengths = np.fromiter((len(x) for x in enc), dtype=np.int64, count=n)
    order = np.argsort(lengths, kind="stable")
    out = {k: np.empty((n, hidden), dtype=np.float32) for k in POOLS}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    i = 0
    bs = batch_size
    next_rep = 1000
    while i < n:
        idx = order[i:i + bs]
        maxlen = int(max(len(enc[j]) for j in idx))
        ids = torch.full((len(idx), maxlen), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), maxlen), dtype=torch.long)
        for r, j in enumerate(idx):
            t = torch.tensor(enc[int(j)], dtype=torch.long)
            ids[r, :t.numel()] = t
            mask[r, :t.numel()] = 1
        try:
            with torch.inference_mode():
                h = model(input_ids=ids.to(device),
                          attention_mask=mask.to(device)).last_hidden_state.float()
            m = mask.to(device).unsqueeze(-1).to(h.dtype)
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
            leadv = h[:, 0, :]
            last = (mask.sum(1) - 1).to(device)
            eos = h[torch.arange(h.shape[0], device=device), last]
            lead_col = torch.full((ids.shape[0], 1), lead, dtype=ids.dtype)
            p_ids = torch.cat([lead_col, ids], dim=1)
            p_mask = torch.cat([torch.ones_like(lead_col), mask], dim=1)
            hb = model(input_ids=p_ids.to(device),
                       attention_mask=p_mask.to(device)).last_hidden_state.float()
            bos = hb[:, 0, :]
            batch = {"mean": mean, "lead": leadv, "eos": eos, "bos": bos}
            for k, v in batch.items():
                out[k][idx] = v.detach().cpu().numpy().astype(np.float32)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > FALLBACK_BATCH:
                print(f"    CUDA OOM at batch={bs}; falling back to "
                      f"{FALLBACK_BATCH}", flush=True)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                bs = FALLBACK_BATCH
                continue
            raise
        i += len(idx)
        if i >= next_rep or i >= n:
            print(f"    embedded {i:,}/{n:,} (batch={bs})", flush=True)
            next_rep = i + 1000
    return out


def embed_pair(xx, csv_path):
    rows = read_rows(csv_path, xx)
    en_texts = [r["en"] for r in rows]
    xx_texts = [r[xx] for r in rows]
    tokenizer, model, device, lead = get_model()
    embdir = ROOT / EMB_FMT.format(xx=xx)
    for mode in POOLS:
        (embdir / mode).mkdir(parents=True, exist_ok=True)
    for lang, texts in ((xx, xx_texts), ("en", en_texts)):
        t = time.time()
        pools = embed_texts(texts, tokenizer, model, device, lead, BATCH)
        for mode in POOLS:
            arr = pools[mode].astype(np.float32)
            path = embdir / mode / f"emb_{lang}.npy"
            np.save(path, arr)
            finite = bool(np.isfinite(arr).all())
            print(f"[{xx}-en] {mode}/emb_{lang}: shape={arr.shape} "
                  f"dtype={arr.dtype} finite={finite} -> "
                  f"{path.relative_to(ROOT)}", flush=True)
            if not finite:
                raise RuntimeError(f"non-finite values in {path}")
        print(f"[{xx}-en] embedded {lang}: {len(texts)} texts in "
              f"{time.time() - t:.1f}s", flush=True)
    return embdir


# --------------------------------------------------------------------------- #
# table-1 evaluation
# --------------------------------------------------------------------------- #
def l2norm(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)


def ranks(sim):
    return 1 + (sim > np.diag(sim)[:, None]).sum(axis=1)


def direction_stats(embs, a, b, dev, test):
    """Top-1 %, MRR, mean rank (raw and +D) for direction a -> b."""
    D = (embs[a][dev] - embs[b][dev]).mean(axis=0)
    E = l2norm(embs[a][test])
    T = l2norm(embs[b][test])
    F = l2norm(embs[b][test] + D)
    n = len(test)
    out = {}
    for tag, sim in (("raw", E @ T.T), ("+D", E @ F.T)):
        r = ranks(sim)
        lo, hi = int(r.min()), int(r.max())
        assert 1 <= lo and hi <= n, f"rank out of [1,{n}]: {lo}..{hi}"
        out[tag] = {"top1": round(100.0 * float((r <= 1).mean()), 4),
                    "mrr": round(float(np.mean(1.0 / r)), 6),
                    "meanrank": round(float(r.mean()), 4),
                    "lo": lo, "hi": hi}
    return out


def evaluate_from(csv_path, embdir, col):
    rows = read_rows(csv_path, col)
    split = np.array([r["split"] for r in rows])
    dev = np.where(split == "dev")[0]
    test = np.where(split == "devtest")[0]
    metrics = {}
    for mode in POOLS:
        embs = {"en": np.load(Path(embdir) / mode / "emb_en.npy"),
                col: np.load(Path(embdir) / mode / f"emb_{col}.npy")}
        metrics[mode] = {
            f"en->{col}": direction_stats(embs, "en", col, dev, test),
            f"{col}->en": direction_stats(embs, col, "en", dev, test),
        }
    return metrics, int(len(dev)), int(len(test))


def evaluate(xx, embdir, csv_path):
    return evaluate_from(csv_path, embdir, xx)


def print_metrics(name, metrics):
    print(f"\nTable 1 ({name}) per-mode, both directions "
          f"(rank columns min..max over devtest queries):", flush=True)
    print("| mode | direction | top-1 raw | top-1 +D | MRR raw | MRR +D "
          "| mean rank raw | mean rank +D |", flush=True)
    print("|---|---|---|---|---|---|---|---|", flush=True)
    for mode in POOLS:
        for d, s in metrics[mode].items():
            print(f"| {mode} | {d} | {s['raw']['top1']:.1f}% "
                  f"| {s['+D']['top1']:.1f}% | {s['raw']['mrr']:.3f} "
                  f"| {s['+D']['mrr']:.3f} | {s['raw']['meanrank']:.1f} "
                  f"| {s['+D']['meanrank']:.1f} |", flush=True)


# --------------------------------------------------------------------------- #
# combined summary
# --------------------------------------------------------------------------- #
def write_summary(progress):
    out = []
    out.append("# Europarl v7 multilingual - exact-translation retrieval (Table 1)\n")
    out.append("Pairs: " + ", ".join(f"{x}-en" for x in LANGS) + ".\n")
    out.append(
        f"Protocol: 5,000 line-aligned Europarl v7 pairs per language (seeded "
        f"rng(0), skip ~1M lines, exact-pair dedupe), dev={DEV_N} / "
        f"devtest={TEST_N}; {TEST_N} devtest queries vs {TEST_N} keys "
        f"(chance rank {(TEST_N + 1) / 2:.1f}); EuroBERT-210m fp32; "
        f"`D = mean_dev(emb_a - emb_b)` fitted on dev and added to the key side "
        f"(`+D` columns). Regenerated as each language finishes.\n")
    out.append("## Summary - mean pooling, +D top-1\n")
    out.append("| pair | en->xx top-1 raw | en->xx top-1 +D | xx->en top-1 raw "
               "| xx->en top-1 +D | en->xx MRR +D | xx->en MRR +D | status |")
    out.append("|---|---|---|---|---|---|---|---|")
    for xx in LANGS:
        st = progress.get(xx, {})
        m = st.get("metrics")
        if not m:
            out.append(f"| {xx}-en | - | - | - | - | - | - | pending |")
            continue
        a = m["mean"][f"en->{xx}"]
        b = m["mean"][f"{xx}->en"]
        out.append(
            f"| {xx}-en | {a['raw']['top1']:.1f}% | {a['+D']['top1']:.1f}% "
            f"| {b['raw']['top1']:.1f}% | {b['+D']['top1']:.1f}% "
            f"| {a['+D']['mrr']:.3f} | {b['+D']['mrr']:.3f} | done |")
    out.append("\n## Per-language summary (mean-mode +D top-1)\n")
    for xx in LANGS:
        st = progress.get(xx, {})
        m = st.get("metrics")
        if not m:
            out.append(f"- **{xx}-en**: _pending_")
            continue
        a = m["mean"][f"en->{xx}"]
        b = m["mean"][f"{xx}->en"]
        out.append(f"- **{xx}-en**: en->{xx} {a['+D']['top1']:.1f}% "
                   f"(MRR {a['+D']['mrr']:.3f}) | {xx}->en "
                   f"{b['+D']['top1']:.1f}% (MRR {b['+D']['mrr']:.3f})")
    out.append("\n## Per-language detail (all modes, raw -> +D)\n")
    for xx in LANGS:
        st = progress.get(xx, {})
        m = st.get("metrics")
        out.append(f"### {xx}-en\n")
        if not m:
            out.append("_pending_\n")
            continue
        out.append(f"dev={st.get('dev', DEV_N)}, devtest N={st.get('test', TEST_N)}, "
                   f"chance rank {(st.get('test', TEST_N) + 1) / 2:.1f}\n")
        out.append("| mode | direction | top-1 raw | top-1 +D | MRR raw | MRR +D "
                   "| mean rank raw | mean rank +D |")
        out.append("|---|---|---|---|---|---|---|---|")
        for mode in POOLS:
            for d in (f"en->{xx}", f"{xx}->en"):
                s = m[mode][d]
                out.append(
                    f"| {mode} | {d} | {s['raw']['top1']:.1f}% "
                    f"| {s['+D']['top1']:.1f}% | {s['raw']['mrr']:.3f} "
                    f"| {s['+D']['mrr']:.3f} | {s['raw']['meanrank']:.1f} "
                    f"| {s['+D']['meanrank']:.1f} |")
        out.append("")
    SUMMARY.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[summary] wrote {SUMMARY.name}", flush=True)


# --------------------------------------------------------------------------- #
# resumable driver
# --------------------------------------------------------------------------- #
def load_progress():
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_progress(p):
    PROGRESS_FILE.write_text(json.dumps(p, indent=2), encoding="utf-8")


def run_lang(xx, progress, force=False):
    st = progress.setdefault(xx, {})
    name = f"{xx}-en"
    csv_path = ROOT / CSV_FMT.format(xx=xx)
    if force or not (st.get("sampled") and csv_path.exists()):
        print(f"[{name}] stage sample", flush=True)
        sample(xx)
        st["sampled"] = True
        save_progress(progress)
    else:
        print(f"[{name}] stage sample: cached", flush=True)

    embdir = ROOT / EMB_FMT.format(xx=xx)
    emb_ok = all((embdir / m / f"emb_{l}.npy").exists()
                 for m in POOLS for l in (xx, "en"))
    if force or not (st.get("embedded") and emb_ok):
        print(f"[{name}] stage embed", flush=True)
        embed_pair(xx, csv_path)
        st["embedded"] = True
        save_progress(progress)
    else:
        print(f"[{name}] stage embed: cached", flush=True)

    if force or not st.get("evaluated"):
        print(f"[{name}] stage eval", flush=True)
        metrics, ndev, ntest = evaluate(xx, embdir, csv_path)
        st["metrics"] = metrics
        st["dev"], st["test"] = ndev, ntest
        st["evaluated"] = True
        save_progress(progress)
        print_metrics(name, metrics)
    else:
        print(f"[{name}] stage eval: cached", flush=True)
    write_summary(progress)


# --------------------------------------------------------------------------- #
# smoke test (no model): reproduce the legacy it-en 4k numbers
# --------------------------------------------------------------------------- #
KNOWN = {
    "en->it": {"raw": (19.2, 0.363), "+D": (71.0, 0.762)},
    "it->en": {"raw": (9.8, 0.277), "+D": (50.0, 0.574)},
}


def smoke():
    csv_path = ROOT / "europarl_it_en.csv"
    embdir = ROOT / "europarl_embeddings"
    if not csv_path.exists() or not embdir.is_dir():
        print("smoke: legacy it-en files missing", file=sys.stderr)
        return 1

    rows = read_rows(csv_path, "it")
    ndev = sum(r["split"] == "dev" for r in rows)
    ntest = sum(r["split"] == "devtest" for r in rows)
    print(f"smoke: {csv_path.name}: {len(rows)} rows "
          f"(dev={ndev}, devtest={ntest})")

    pairs = [(r["en"], r["it"]) for r in rows]
    print(f"smoke: loader checks: unique={len(set(pairs)) == len(pairs)} "
          f"nonempty={all(a and b for a, b in pairs)}")

    xx_path, en_path = locate_corpus_paths(ROOT / "europarl_it_en", "it", "en")
    tiny, scanned = collect_pairs(xx_path, en_path, 0, 100)
    print(f"smoke: sampler check collect_pairs(cap=100): {len(tiny)} unique, "
          f"nonempty={all(a and b for a, b in tiny)}, scanned={scanned}")

    metrics, ndev, ntest = evaluate_from(csv_path, embdir, "it")
    ok = True
    for d, exp in KNOWN.items():
        got = metrics["mean"][d]
        for tag in ("raw", "+D"):
            et, em = exp[tag]
            gt, gm = got[tag]["top1"], got[tag]["mrr"]
            good = abs(gt - et) <= 0.1 and abs(gm - em) <= 0.002
            ok = ok and good
            print(f"smoke: mean {d} {tag}: top1 got {gt:.1f} exp {et:.1f} | "
                  f"mrr got {gm:.3f} exp {em:.3f} -> "
                  f"{'ok' if good else 'MISMATCH'}")
    print(f"\nsmoke: {'MATCH' if ok else 'FAIL'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "all", "pair"])
    ap.add_argument("--lang", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.mode == "smoke":
        sys.exit(smoke())

    if args.mode == "pair" and args.lang not in LANGS:
        raise SystemExit(f"--lang must be one of {LANGS}")

    PID_FILE.write_text(str(os.getpid()), encoding="ascii")
    logf = RUN_LOG.open("a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, logf)
    sys.stderr = _Tee(sys.__stderr__, logf)

    print(f"\n===== pipeline start {now()} pid={os.getpid()} "
          f"mode={args.mode} =====", flush=True)
    status(f"[START] pipeline mode={args.mode} lang={args.lang} pid={os.getpid()}")

    progress = load_progress()
    langs = [args.lang] if args.mode == "pair" else LANGS
    failed = []
    for xx in langs:
        try:
            run_lang(xx, progress, force=args.force)
        except Exception as e:  # noqa: BLE001 - report and continue other langs
            import traceback
            traceback.print_exc()
            failed.append(xx)
            status(f"[ERROR] {xx}-en: {type(e).__name__}: {e}")

    write_summary(progress)
    if failed:
        status(f"[DONE] pipeline WITH FAILURES={','.join(failed)}")
        print(f"\npipeline finished with failures: {failed}", flush=True)
        logf.flush()
        sys.exit(1)
    status("[DONE] pipeline all ok")
    print(f"===== pipeline done {now()} =====", flush=True)
    logf.flush()


if __name__ == "__main__":
    main()
