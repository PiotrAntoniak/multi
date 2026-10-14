"""fetch_flores_en_pl.py -- create a small EN->PL FLORES-200 sample (create-only).

FLORES-200 is released by NLLB at the public Meta URL
  https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
(the `facebook/flores` and `openlanguagedata/flores_plus` HF repos are gated). This script
downloads that ~25 MB tarball, extracts `eng_Latn.devtest` + `pol_Latn.devtest`, aligns the
two by line index (FLORES is line-aligned across languages), and writes flores_en_pl.csv
with columns `en,pl` (English source + human Polish reference).

Usage:
  python tools/fetch_flores_en_pl.py [out_csv] [n_rows]
"""
import os
import sys
import tarfile
import urllib.request

URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
TMP = os.path.join(os.environ.get("TEMP", "."), "flores200_dataset.tar.gz")
SPLIT = "devtest"
MEMBER_SUFFIX = {
    "en": f"/{SPLIT}/eng_Latn.{SPLIT}",
    "pl": f"/{SPLIT}/pol_Latn.{SPLIT}",
}


def ensure_tarball():
    if not os.path.isfile(TMP) or os.path.getsize(TMP) < 1_000_000:
        print("downloading", URL, flush=True)
        urllib.request.urlretrieve(URL, TMP)
    print("tarball", TMP, os.path.getsize(TMP), "bytes", flush=True)


def read_member(tf, suffix):
    match = next((n for n in tf.getnames() if n.endswith(suffix)), None)
    if match is None:
        raise SystemExit(f"no tarball member ending in {suffix}")
    f = tf.extractfile(match)
    if f is None:
        raise SystemExit(f"{match} not extractable")
    return f.read().decode("utf-8").splitlines()


def main():
    import csv

    out_csv = sys.argv[1] if len(sys.argv) > 1 else "flores_en_pl.csv"
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 25

    ensure_tarball()
    with tarfile.open(TMP, "r:*") as tf:
        en = read_member(tf, MEMBER_SUFFIX["en"])
        pl = read_member(tf, MEMBER_SUFFIX["pl"])

    n = min(len(en), len(pl))
    rows = [(en[i].strip(), pl[i].strip()) for i in range(min(n_rows, n))]

    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["en", "pl"])
        w.writerows(rows)

    print(f"source   : NLLB FLORES-200 tarball ({SPLIT}), eng_Latn + pol_Latn")
    print(f"available: en={len(en)} pl={len(pl)} aligned={n} lines")
    print(f"wrote    : {out_csv}  n={len(rows)}")
    for r in rows[:2]:
        print("EN:", r[0][:100])
        print("PL:", r[1][:100])


if __name__ == "__main__":
    main()
