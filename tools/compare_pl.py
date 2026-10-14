"""compare_pl.py -- side-by-side EN->PL outputs + chrF vs the human FLORES reference.

Reads flores_en_pl.csv (en,pl) and the two model outputs (out_qwen35_2b_pl.csv,
out_qwen25_15b_pl.csv), prints ~N example rows side by side, and reports chrF of each model's
variant-0 output against the human `pl` reference (sacrebleu). Writes compare_pl.json + compare_pl.md.

Usage: python tools/compare_pl.py [n_examples]
"""
import csv
import json
import os
import sys

from sacrebleu.metrics import CHRF

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

REF_CSV = "flores_en_pl.csv"
MODELS = [
    ("Qwen3.5-2B", "out_qwen35_2b_pl.csv"),
    ("Qwen2.5-3B-Instruct", "out_qwen25_3b_pl.csv"),
]
N_EXAMPLES = 4


def clean(s):
    s = (s or "").strip()
    for q in ('"""', '"'):
        s = s[len(q):] if s.startswith(q) else s
        s = s[:-1] if s.endswith(q) else s
    return s.strip()


def load_variants(path):
    """{variant_index: [cleaned Polish, ...]} in file order (parallel to the reference rows)."""
    byv = {}
    with open(os.path.join(ROOT, path), encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            v = int(row.get("variant") or 0)
            byv.setdefault(v, []).append(clean(row.get("polish_text", "")))
    return byv


def main():
    n_examples = int(sys.argv[1]) if len(sys.argv) > 1 else N_EXAMPLES

    with open(os.path.join(ROOT, REF_CSV), encoding="utf-8") as fh:
        refs_rows = list(csv.DictReader(fh))
    en = [r["en"] for r in refs_rows]
    refs = [r["pl"] for r in refs_rows]

    chrf = CHRF()
    report = {"reference": REF_CSV, "n_rows": len(refs), "chrf_vs_human_pl": {}}
    hyps0 = {}
    for name, path in MODELS:
        byv = load_variants(path)
        v0 = byv.get(0, [])
        n = min(len(refs), len(v0))
        variant_scores = {}
        for vi in sorted(byv):
            rows_vi = byv[vi][:n]
            variant_scores[vi] = round(chrf.corpus_score(rows_vi, [refs[:n]]).score, 2)
        score = chrf.corpus_score(v0[:n], [refs[:n]]).score
        report["chrf_vs_human_pl"][name] = {
            "variant0_chrF": round(score, 2),
            "per_variant_chrF": variant_scores,
            "mean_variant_chrF": round(sum(variant_scores.values()) / max(len(variant_scores), 1), 2),
            "n_variant0": len(v0), "n_all": sum(len(v) for v in byv.values()),
        }
        hyps0[name] = v0

    # attach timing if present
    tp = os.path.join(ROOT, "out_timing.json")
    if os.path.isfile(tp):
        with open(tp, encoding="utf-8") as fh:
            report["timing"] = json.load(fh)

    # examples
    examples = []
    for i in range(min(n_examples, len(refs))):
        ex = {"en": en[i], "human_pl": refs[i]}
        for name, _ in MODELS:
            ex[name] = hyps0[name][i] if i < len(hyps0[name]) else ""
        examples.append(ex)
    report["examples"] = examples

    with open(os.path.join(ROOT, "compare_pl.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    lines = [f"# EN->PL FLORES comparison ({len(refs)} rows, devtest)", ""]
    lines.append("## chrF vs human Polish reference")
    lines.append("")
    lines.append("| model | variant-0 chrF | mean of per-variant chrF |")
    lines.append("|---|---|---|")
    for name, _ in MODELS:
        d = report["chrf_vs_human_pl"][name]
        lines.append(f"| {name} | {d['variant0_chrF']} | {d['mean_variant_chrF']} |")
    lines.append("")
    if "timing" in report:
        lines.append("## throughput")
        lines.append("")
        lines.append("| model | tokens | gen seconds | tok/s |")
        lines.append("|---|---|---|---|")
        for name, d in report["timing"].items():
            lines.append(f"| {name} | {d.get('output_tokens','')} | {d.get('gen_seconds','')} "
                         f"| {d.get('tok_per_s','')} |")
        lines.append("")
    lines.append("## examples (paraphrase->translate)")
    lines.append("")
    for i, ex in enumerate(examples, 1):
        lines.append(f"### {i}")
        lines.append(f"- **EN**: {ex['en']}")
        lines.append(f"- **human PL**: {ex['human_pl']}")
        for name, _ in MODELS:
            lines.append(f"- **{name}**: {ex[name]}")
        lines.append("")
    with open(os.path.join(ROOT, "compare_pl.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # console summary (ascii-safe-ish; may mojibake in cp1252 consoles)
    print(json.dumps(report["chrf_vs_human_pl"], indent=2, ensure_ascii=False))
    if "timing" in report:
        print(json.dumps({k: {kk: vv for kk, vv in v.items()
                              if kk in ("output_tokens", "gen_seconds", "tok_per_s")}
                          for k, v in report["timing"].items()}, indent=2))
    print("wrote compare_pl.json + compare_pl.md")


if __name__ == "__main__":
    main()
