"""Input-embedding norms and per-pooling cosine similarity on example pairs."""
import sys

import torch
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

MODEL = "EuroBERT/EuroBERT-210m"
LEAD = 128000
PAIRS = [
    ("unrelated", "The stock market fell sharply today.", "Penguins live in Antarctica."),
    ("unrelated", "The recipe calls for two eggs and a cup of flour.", "Parliament passed the budget after a long debate."),
    ("unrelated", "The concert was cancelled due to rain.", "Photosynthesis converts light into chemical energy."),
    ("related", "The stock market fell sharply today.", "Markets dropped significantly today."),
    ("related", "Scientists announced a new diagnostic tool.", "Researchers presented a new medical test."),
    ("related", "The cat is sleeping on the sofa.", "A cat naps on the couch."),
    ("identical", "The stock market fell sharply today.", "The stock market fell sharply today."),
    ("translation", "The stock market fell sharply today.", "Il mercato azionario e crollato bruscamente oggi."),
]


def pools(tok, model, texts):
    b = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    ids, am = b["input_ids"], b["attention_mask"]
    lead = torch.full((len(ids), 1), LEAD)
    with torch.inference_mode():
        h = model(input_ids=ids, attention_mask=am).last_hidden_state.float()
        hb = model(input_ids=torch.cat([lead, ids], 1),
                   attention_mask=torch.cat([torch.ones_like(lead), am], 1)).last_hidden_state.float()
    out = {"mean": normalize((h * am[..., None]).sum(1) / am.sum(1, keepdim=True), dim=1),
           "lead": normalize(h[:, 0], dim=1),
           "bos": normalize(hb[:, 0], dim=1),
           "eos": normalize(h[torch.arange(len(ids)), am.sum(1) - 1], dim=1)}
    return out


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float32).eval()

    texts = ["Ciao, come stai?"] + list(dict.fromkeys(s for _, a, b in PAIRS for s in (a, b)))
    for s in texts:
        ids = tok(s)["input_ids"]
        print(f"text: {s}")
        print(f"  plain (feeds mean, lead, eos): {tok.convert_ids_to_tokens(ids)}")
        print(f"  bos   (feeds bos only)       : {tok.convert_ids_to_tokens([LEAD] + ids)}")

    w = model.get_input_embeddings().weight.detach().float()
    n = w.norm(dim=1)
    for i in (128000, 128001, 128002):
        print(f"id {i} {tok.convert_ids_to_tokens(i)}: norm {n[i]:.5f}")
    print(f"median row norm {n.median():.5f} | mean row norm {n.mean():.5f} | "
          f"reserved 128003+ median {n[128003:].median():.5f} min {n[128003:].min():.5f} max {n[128003:].max():.5f}")

    print("\ncosine per pooling on example pairs (mean/lead/eos = plain input above, bos = prepended input above)")
    acc = {"unrelated": [], "related": []}
    for kind, a, b in PAIRS:
        r = pools(tok, model, [a, b])
        cos = [float(r[p][0] @ r[p][1]) for p in ("mean", "lead", "bos", "eos")]
        if kind in acc:
            acc[kind].append(cos)
        print(f"{kind:11s} " + " ".join(f"{c:+.4f}" for c in cos) + f" | {a} vs {b}")
    for j, name in enumerate(("mean", "lead", "bos", "eos")):
        u, r = acc["unrelated"], acc["related"]
        print(f"{name}: unrelated {sum(c[j] for c in u) / len(u):+.3f} | "
              f"related {sum(c[j] for c in r) / len(r):+.3f}")


if __name__ == "__main__":
    main()
