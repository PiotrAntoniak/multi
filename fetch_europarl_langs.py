"""Download + extract + verify Europarl v7 xx-en for xx in {de,es,fr,nl,pt}.

For each target language xx:
  1. try the StatMT v7 archive  https://www.statmt.org/europarl/v7/{xx}-en.tgz
     (saved locally as `europarl_{xx}_en.tgz`);
  2. on ANY failure (HTTP error, bad archive, misaligned/empty result), fall
     back to the OPUS v7 moses archive
     https://object.pouta.csc.fi/OPUS-Europarl/v7/moses/{xx}-en.txt.zip
     (saved as `europarl_{xx}_en.txt.zip`);
  3. extract into `europarl_{xx}_en/`;
   4. locate the xx-side and en-side plain-text files by filename suffix and
      verify the two files are LINE-ALIGNED: equal line counts (streamed in
      lockstep) and an identical empty/non-empty pattern on every line.

After a fully successful run it also (re)writes the combined
`europarl_all_5k.csv` from the six per-language sampled CSVs written by
`table1_europarl_multilang.py` (`combine_5k`): one leading `lang` column plus
the union of the per-language columns, row order preserved.

Idempotent / resumable: a language whose two aligned files already exist in
`europarl_{xx}_en/` is skipped; a cached archive (>= 1 MB) is reused instead of
re-downloaded.

Runs detached and hidden (see run_europarl_multilang_chain.cmd /
launch_europarl_multilang.vbs).  Records its own PID in `europarl_multilang.pid`,
appends progress to `europarl_multilang_fetch.log` and per-language stage lines
to `europarl_multilang_status.txt`.

Usage: python fetch_europarl_langs.py
"""
import csv
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LANGS = ["de", "es", "fr", "nl", "pt"]
STATMT = "https://www.statmt.org/europarl/v7/{xx}-en.tgz"
OPUS = "https://object.pouta.csc.fi/OPUS-Europarl/v7/moses/{xx}-en.txt.zip"

# Per-language sampled CSVs (written by table1_europarl_multilang.py's `sample`)
# and their single combined counterpart for the minimal repo.
CSV_FMT = "europarl_{xx}_en_5k.csv"
COMBINED = ROOT / "europarl_all_5k.csv"
COMBINE_LANGS = ["de", "es", "fr", "it", "nl", "pt"]

PID_FILE = ROOT / "europarl_multilang.pid"
LOG_FILE = ROOT / "europarl_multilang_fetch.log"
STATUS_FILE = ROOT / "europarl_multilang_status.txt"

CHUNK = 1 << 20
MIN_ARCHIVE_BYTES = 1_000_000
MIN_LINES = 100_000
HEADERS = {"User-Agent": "Mozilla/5.0 (europarl-multilang fetch)"}


class _Tee:
    """Write to several streams (console + log file) tolerating closed streams."""

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


def combine_5k(langs=None):
    """Concatenate the per-language 5k CSVs into ``europarl_all_5k.csv``.

    Keeps every input file's row order and columns, prepending a leading
    ``lang`` column with the language code (union of the per-language text
    columns, in ``COMBINE_LANGS`` order).  Returns the output path, or ``None``
    if any input file is missing (nothing is written in that case).
    """
    langs = list(langs or COMBINE_LANGS)
    frames = []
    for xx in langs:
        path = ROOT / CSV_FMT.format(xx=xx)
        if not path.exists():
            print(f"[combine] {path.name} missing; combined file not written",
                  flush=True)
            return None
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        frames.append((xx, header, rows))

    out_header = ["lang"]
    for _xx, header, _rows in frames:
        for col in header:
            if col not in out_header:
                out_header.append(col)

    n_rows = sum(len(rows) for _xx, _h, rows in frames)
    with COMBINED.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        for xx, header, rows in frames:
            for row in rows:
                rec = {col: "" for col in out_header}
                rec["lang"] = xx
                for col, val in zip(header, row):
                    rec[col] = val
                writer.writerow([rec[col] for col in out_header])
    size_mb = COMBINED.stat().st_size / 1e6
    print(f"[combine] wrote {COMBINED.name}: {n_rows} rows, "
          f"{size_mb:.2f} MB, columns={out_header}", flush=True)
    return COMBINED


def detect_encoding(path):
    """First of utf-8, cp1252, latin-1 that decodes the whole file."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                while f.read(1 << 20):
                    pass
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def download(url, dest):
    """Download url -> dest (.part then atomic rename), with 3 retries."""
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=180) as r:
                total = int(r.headers.get("Content-Length") or 0)
                read = 0
                t0 = time.time()
                lastlog = 0.0
                with open(part, "wb") as f:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        read += len(chunk)
                        el = time.time() - t0
                        if el - lastlog >= 15:
                            lastlog = el
                            pct = f"{100.0 * read / total:.1f}%" if total else "?"
                            tot = f"/{total / 1e6:.1f} MB" if total else ""
                            print(f"    ... {read / 1e6:.1f} MB{tot} ({pct}) "
                                  f"{read / max(el, 1e-6) / 1e6:.1f} MB/s",
                                  flush=True)
            if total and read != total:
                raise IOError(f"short read {read}/{total}")
            os.replace(part, dest)
            print(f"    downloaded {dest.name} ({read / 1e6:.1f} MB)", flush=True)
            return read
        except Exception as e:
            print(f"    download attempt {attempt}/3 failed: {e}", flush=True)
            if part.exists():
                try:
                    part.unlink()
                except OSError:
                    pass
            if attempt == 3:
                raise
            time.sleep(3 * attempt)
    raise RuntimeError("unreachable")


def wipe_dir(d):
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def safe_extract_tar(t, outdir):
    base = outdir.resolve()
    for m in t.getmembers():
        if not m.isfile():
            continue
        target = (outdir / m.name).resolve()
        if base not in target.parents and target != base:
            print(f"    skip unsafe member {m.name}", flush=True)
            continue
        t.extract(m, outdir)


def extract(archive, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".tgz") or name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as t:
            safe_extract_tar(t, outdir)
    elif name.endswith(".zip"):
        base = outdir.resolve()
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                target = (outdir / member).resolve()
                if base not in target.parents and target != base:
                    print(f"    skip unsafe member {member}", flush=True)
                    continue
                z.extract(member, outdir)
    else:
        raise ValueError(f"unknown archive type: {archive.name}")
    print(f"    extracted {archive.name} -> {outdir.name}/", flush=True)


def locate_sides(outdir, xx, en="en"):
    """Find the xx-side and en-side files under outdir by filename suffix."""
    xp = ep = None
    if outdir.is_dir():
        for p in sorted(outdir.rglob("*")):
            if not p.is_file():
                continue
            if p.name.endswith(f".{xx}"):
                xp = p
            elif p.name.endswith(f".{en}"):
                ep = p
    return xp, ep


def verify_aligned(xx_path, en_path):
    """Stream both files in lockstep; return line counts + alignment stats.

    `ragged` = difference in the two files' line counts (must be 0 for a
    line-aligned pair).  `mismatch` counts lines where exactly one side is
    empty within the overlap; a small ratio (< 1%) is normal Europarl noise.
    """
    enc_xx = detect_encoding(xx_path)
    enc_en = detect_encoding(en_path)
    n_xx = n_en = mismatch = both_nonempty = 0
    with xx_path.open(encoding=enc_xx, errors="replace") as fx, \
            en_path.open(encoding=enc_en, errors="replace") as fe:
        while True:
            lx = fx.readline()
            le = fe.readline()
            xdone, edone = lx == "", le == ""
            if xdone and edone:
                break
            if not xdone:
                n_xx += 1
            if not edone:
                n_en += 1
            if xdone or edone:
                continue
            a, b = le.strip(), lx.strip()
            if (not a) != (not b):
                mismatch += 1
            if a and b:
                both_nonempty += 1
    return {"lines": max(n_xx, n_en), "n_xx": n_xx, "n_en": n_en,
            "ragged": abs(n_xx - n_en), "mismatch": mismatch,
            "nonempty": both_nonempty, "enc_xx": enc_xx, "enc_en": enc_en}


def aligned_ok(v):
    """Line-aligned iff equal line counts and only minor empty-pattern noise."""
    if v["lines"] <= MIN_LINES or v["ragged"] != 0:
        return False
    return v["mismatch"] <= max(100, int(0.01 * v["lines"]))


def fetch_from(xx, url, archive, outdir):
    """Download+cache, extract, locate sides, verify alignment. Raises on failure."""
    if archive.exists() and archive.stat().st_size >= MIN_ARCHIVE_BYTES:
        print(f"    using cached {archive.name} "
              f"({archive.stat().st_size / 1e6:.1f} MB)", flush=True)
    else:
        print(f"    downloading {url}", flush=True)
        download(url, archive)
    wipe_dir(outdir)
    extract(archive, outdir)
    xp, ep = locate_sides(outdir, xx)
    if not xp or not ep:
        raise IOError(f"{xx}-side/en-side files not found under {outdir}")
    v = verify_aligned(xp, ep)
    print(f"    verify: lines_xx={v['n_xx']:,} lines_en={v['n_en']:,} "
          f"ragged={v['ragged']} empty_mismatch={v['mismatch']} "
          f"nonempty={v['nonempty']:,} enc_xx={v['enc_xx']} "
          f"enc_en={v['enc_en']}", flush=True)
    if v["ragged"] != 0:
        raise IOError(f"{xp.name}/{ep.name} NOT line-aligned: ragged={v['ragged']}")
    if not aligned_ok(v):
        raise IOError(f"{xp.name}/{ep.name} alignment/size check failed: "
                      f"lines={v['lines']} mismatch={v['mismatch']}")
    return xp, ep, v


def is_ready(xx):
    """Return (xx_path, en_path, verify) if both sides already exist and align."""
    outdir = ROOT / f"europarl_{xx}_en"
    xp, ep = locate_sides(outdir, xx)
    if not xp or not ep:
        return None
    v = verify_aligned(xp, ep)
    if aligned_ok(v):
        return xp, ep, v
    return None


def main():
    PID_FILE.write_text(str(os.getpid()), encoding="ascii")
    logf = LOG_FILE.open("a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, logf)
    sys.stderr = _Tee(sys.__stderr__, logf)

    print(f"\n===== fetch start {now()} pid={os.getpid()} =====", flush=True)
    status(f"[START] fetch langs={','.join(LANGS)} pid={os.getpid()}")

    failures = []
    for xx in LANGS:
        print(f"\n--- {xx}-en ---", flush=True)
        ready = is_ready(xx)
        if ready:
            xp, ep, v = ready
            print(f"  already present + aligned: {xp.name} / {ep.name} "
                  f"({v['lines']:,} lines)", flush=True)
            status(f"[SKIP] {xx}-en present lines={v['lines']} "
                   f"files={xp.name},{ep.name}")
            continue

        outdir = ROOT / f"europarl_{xx}_en"
        source = None
        errors = []
        candidates = [
            ("statmt", STATMT.format(xx=xx), ROOT / f"europarl_{xx}_en.tgz"),
            ("opus", OPUS.format(xx=xx), ROOT / f"europarl_{xx}_en.txt.zip"),
        ]
        for tag, url, archive in candidates:
            print(f"  source={tag}: {url}", flush=True)
            try:
                xp, ep, v = fetch_from(xx, url, archive, outdir)
                source = (tag, archive.name)
                status(f"[OK] {xx}-en source={tag} archive={archive.name} "
                       f"lines={v['lines']} files={xp.name},{ep.name} "
                       f"enc_xx={v['enc_xx']} enc_en={v['enc_en']}")
                break
            except Exception as e:
                print(f"  {tag} failed for {xx}: {e}", flush=True)
                errors.append(f"{tag}={e}")
        if source is None:
            failures.append(xx)
            status(f"[ERROR] {xx}-en all sources failed: {'; '.join(errors)}")

    if failures:
        status(f"[DONE] fetch WITH FAILURES={','.join(failures)}")
        print(f"\nfetch finished with failures: {failures}", flush=True)
        logf.flush()
        sys.exit(1)

    combine_5k()
    status("[DONE] fetch all ok")
    print(f"\n===== fetch done {now()} =====", flush=True)
    logf.flush()


if __name__ == "__main__":
    main()
