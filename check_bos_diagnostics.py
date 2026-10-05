"""BOS diagnostics over 15 English + 15 Italian random FLORES sentences (seed 0), each language
forwarded separately. Prepends special id 128000 (<|begin_of_text|>) and runs EuroBERT-210m
(native path; remote code is broken on transformers 5.x). Three forwards per language: with BOS,
without BOS, and with BOS attention-masked (sanity check: masked == no-BOS, because RoPE is
relative and masking removes BOS from attention). Layer 0 = input embeddings; layers 1-11 =
transformer layer outputs; layer 12 = last-layer output BEFORE the final RMSNorm (captured with a
forward hook on model.norm) - marked with * in the tables.

Table 1 - per-layer BOS / mean-pooling stats (columns per language: en_* / it_*):
  bos_norm    mean L2 norm of the BOS token's hidden state (position 0), averaged over sentences
  bos_cos     mean off-diagonal pairwise cosine between the BOS vectors of different sentences
  bos_cos_c   same, after subtracting the mean BOS vector (removes the shared component)
  mean_norm   mean L2 norm of the masked mean-pooled vector (average over non-BOS tokens)
  mean_cos    mean off-diagonal pairwise cosine of the pooled vectors
  mean_cos_c  same, mean-centered
  tok_norm    mean per-token L2 norm over non-BOS tokens (not the norm of the average)

Table 2 - BOS ablation: change of the OTHER tokens (content + EOS; BOS itself excluded) when BOS
is absent, relative to the with-BOS run:
  noBOS_relD    mean over tokens of ||h_noBOS - h_BOS|| / ||h_BOS|| (per-token relative movement)
  noBOS_cos     mean token-wise cosine between the no-BOS and with-BOS hidden states
  noBOS_poolD   ||mean_noBOS - mean_BOS|| / ||mean_BOS|| for the masked mean-pooled vector

Usage: python check_bos_diagnostics.py
"""
import csv
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

BASE = os.path.dirname(os.path.abspath(__file__))
NAME = "EuroBERT/EuroBERT-210m"
SEED, LANGS, PER_LANG, BOS_ID = 0, ["en", "it"], 15, 128000
METRICS = ["bos_norm", "bos_cos", "bos_cos_c", "mean_norm", "mean_cos", "mean_cos_c", "tok_norm"]
ABL = ["noBOS_relD", "noBOS_cos", "noBOS_poolD"]


def main():
    with open(os.path.join(BASE, "flores200_en_it_de_fr.csv"), encoding="utf-8-sig",
              newline="") as f:
        rows = list(csv.DictReader(f))

    rng = random.Random(SEED)
    samples_by_lang = {}
    for l in LANGS:
        idxs = rng.sample(range(len(rows)), PER_LANG)
        samples_by_lang[l] = [(rows[i]["id"], rows[i][l]) for i in idxs]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModel.from_pretrained(NAME).eval().to(device)  # native; remote code is broken on transformers 5.x
    pre_norm = {}
    model.norm.register_forward_hook(
        lambda m, inp, out: pre_norm.__setitem__("x", inp[0].detach()))  # last-layer output pre-norm

    def encode(texts, mode):
        """mode: 'bos' | 'no' | 'bos_masked'. Returns full hidden states per layer + full mask."""
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        if mode != "no":
            ids = enc["input_ids"]
            col = torch.full((ids.shape[0], 1), BOS_ID, dtype=ids.dtype, device=device)
            enc["input_ids"] = torch.cat([col, ids], dim=1)
            enc["attention_mask"] = torch.cat(
                [torch.ones_like(col), enc["attention_mask"]], dim=1)
            if mode == "bos_masked":
                enc["attention_mask"][:, 0] = 0
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        layers = list(out.hidden_states[:-1]) + [pre_norm["x"]]   # layer 12 = pre-final-norm
        return layers, enc["attention_mask"]

    def pair_cos_off(x):
        n = x.shape[0]
        s = F.cosine_similarity(x.unsqueeze(0), x.unsqueeze(1), dim=-1)
        return ((s.sum() - n) / (n * (n - 1))).item()   # drop the diagonal

    def centered(x):
        return x - x.mean(0, keepdim=True)              # remove the shared direction

    def stats_rows(layers, mask):
        out_rows = []
        for h in layers:
            pos0 = h[:, 0]
            cmask = mask[:, 1:].unsqueeze(-1)
            mean = (h[:, 1:] * cmask).sum(1) / cmask.sum(1)
            tok_norm = ((h[:, 1:].norm(dim=-1) * mask[:, 1:]).sum() / mask[:, 1:].sum())
            out_rows.append([pos0.norm(dim=-1).mean().item(), pair_cos_off(pos0),
                             pair_cos_off(centered(pos0)), mean.norm(dim=-1).mean().item(),
                             pair_cos_off(mean), pair_cos_off(centered(mean)),
                             tok_norm.item()])
        return out_rows

    def compare(ref_layers, ref_mask, cand_layers, cand_mask):
        """Token relD / token cos / pooled relD of the other tokens, candidate vs reference."""
        m = ref_mask & cand_mask
        out_rows = []
        for h_ref, h_cand in zip(ref_layers, cand_layers):
            d = (h_cand - h_ref).norm(dim=-1)
            r = h_ref.norm(dim=-1).clamp_min(1e-6)
            rel = ((d / r) * m).sum() / m.sum()
            cos = (F.cosine_similarity(h_cand, h_ref, dim=-1) * m).sum() / m.sum()
            pool_ref = (h_ref * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
            pool_cand = (h_cand * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
            pool_rel = ((pool_cand - pool_ref).norm(dim=-1)
                        / pool_ref.norm(dim=-1).clamp_min(1e-6)).mean()
            out_rows.append((rel.item(), cos.item(), pool_rel.item()))
        return out_rows

    stats, abl = {}, {}
    for l in LANGS:
        print(f"[{l}] {PER_LANG} random sentences (seed={SEED}):")
        for rid, t in samples_by_lang[l]:
            print(f"  [{l} id={rid}] {t[:80]}")
        texts = [t for _, t in samples_by_lang[l]]
        ref_layers, ref_mask = encode(texts, "bos")
        no_layers, no_mask = encode(texts, "no")
        msk_layers, msk_mask = encode(texts, "bos_masked")
        first = tok(texts[0])["input_ids"][:2]
        print(f"[{l}] first 3 tokens:", tok.convert_ids_to_tokens([BOS_ID] + first))
        stats[l] = stats_rows(ref_layers, ref_mask)
        ref_c, ref_cm = [h[:, 1:] for h in ref_layers], ref_mask[:, 1:]
        no_c, no_cm = [h[:, 0:] for h in no_layers], no_mask[:, 0:]
        msk_c, msk_cm = [h[:, 1:] for h in msk_layers], msk_mask[:, 1:]
        abl[l] = compare(ref_c, ref_cm, no_c, no_cm)
        d_check = (msk_c[-1] - no_c[-1]).abs().max().item()
        print(f"[{l}] sanity: masked vs no-BOS max abs diff (layer 12) = {d_check:.2e} "
              f"(RoPE is relative -> masking BOS == deleting it)")

    names = [f"{l}_{m}" for l in LANGS for m in METRICS]
    w = max(len(n) for n in names) + 1
    print("\n== per-layer BOS / mean-pooling stats ==")
    print(f"{'layer':>5} " + "  ".join(f"{n:>{w}}" for n in names))
    n_layers = len(stats[LANGS[0]])
    for layer in range(n_layers):
        vals = [v for l in LANGS for v in stats[l][layer]]
        tag = str(layer) + ("*" if layer == n_layers - 1 else "")
        print(f"{tag:>5} " + "  ".join(f"{v:>{w}.3f}" for v in vals))

    names2 = [f"{l}_{a}" for l in LANGS for a in ABL]
    w2 = max(len(n) for n in names2) + 1
    print("\n== BOS ablation: other-token change vs with-BOS run ==")
    print(f"{'layer':>5} " + "  ".join(f"{n:>{w2}}" for n in names2))
    for layer in range(n_layers):
        vals = [v for l in LANGS for v in abl[l][layer]]
        tag = str(layer) + ("*" if layer == n_layers - 1 else "")
        print(f"{tag:>5} " + "  ".join(f"{v:>{w2}.4f}" for v in vals))
    print("* layer 12 = pre-final-RMSNorm; 'noBOS' = no BOS in the input. masked == no-BOS "
          "(RoPE relative, BOS excluded from attention), so the ablation isolates the effect of "
          "the other tokens ATTENDING TO BOS.")


if __name__ == "__main__":
    main()
