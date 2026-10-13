"""Reusable paraphrase / cross-lingual rewrite prompt templates.

See ``PARAPHRASE_TEMPLATE.md`` for the design rationale and the literature it is based on
(FinePhrase arXiv:2604.13977; Self-Translate-Train arXiv:2407.00454; "To Translate or Not to
Translate" arXiv:2311.09404; Geometric Filtering arXiv:2608.13866).

Everything is plain strings with ``{src}`` / ``{tgt}`` / ``{text}`` / ``{lang}`` placeholders so it can be
used with any instruct LLM's chat template.

Usage::

    from paraphrase_prompts import xling_system, xling_user, build_messages
    msgs = build_messages("crosslingual", text="On Monday, scientists ...",
                          src="English", tgt="Italian")
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                        enable_thinking=False)

Filtering helpers (apply to generations; small LLMs err) are at the bottom.
"""

# ---------------------------------------------------------------------------
# 1. Cross-lingual rewrite (English -> target) -- label-preserving translate-train
# ---------------------------------------------------------------------------
XLING_SYSTEM = (
    "You are a careful multilingual data generator. You rewrite the user's text into {tgt} while "
    "preserving its meaning exactly. You never add, drop, summarise, or explain information. You output "
    "only the {tgt} text, with no quotes, labels, or commentary."
)
XLING_USER = (
    "Rewrite the following {src} text into natural, fluent {tgt}.\n"
    "- Preserve ALL factual content and entities (names, numbers, dates).\n"
    "- Do not add or remove information.\n"
    "- Do not replace proper nouns with different entities.\n"
    "- Output ONLY the {tgt} text. No preamble, no quotes.\n\n"
    "{src} text:\n\"\"\" {text} \"\"\""
)

# ---------------------------------------------------------------------------
# 2. Monolingual paraphrase
# ---------------------------------------------------------------------------
PARA_SYSTEM = (
    "You are a careful paraphrasing assistant. You restate text in different words while keeping the "
    "meaning identical. You output only the paraphrase."
)
PARA_USER = (
    "Paraphrase the following {lang} text. Keep the meaning identical; change the wording and sentence "
    "structure. Do not add or remove information. Output ONLY the paraphrase.\n\n"
    "Text:\n\"\"\" {text} \"\"\""
)
PARA_STYLE_HINTS = [  # prepend for diverse repeated sampling
    "Use a formal register.",
    "Use a simple, everyday register.",
    "Restructure the sentences.",
    "Use different synonyms throughout.",
]

# ---------------------------------------------------------------------------
# 3. Structured pedagogical rewrite (FinePhrase-style; for pretraining-style text)
# ---------------------------------------------------------------------------
STRUCTURED = {
    "table":    "Rewrite the passage as a well-formed Markdown table that captures the same information.",
    "faq":      "Turn the passage into a list of question-answer pairs covering the same content.",
    "tutorial": "Rewrite the passage as a short step-by-step tutorial that teaches the same material.",
    "math":     "Write a math word problem whose solution requires the facts in the passage, then solve it.",
}

_KINDS = {
    "crosslingual": (XLING_SYSTEM, XLING_USER),
    "paraphrase":   (PARA_SYSTEM, PARA_USER),
}


def build_messages(kind, text, src="English", tgt="Italian", lang=None, style=None):
    """Return a chat ``messages`` list for ``kind`` in {"crosslingual","paraphrase"}.

    ``text`` is the source; ``src``/``tgt`` name the languages (cross-lingual) and ``lang`` the language
    (paraphrase). ``style`` optionally prepends a diversity hint (paraphrase only).
    """
    if kind not in _KINDS:
        raise ValueError("kind must be 'crosslingual' or 'paraphrase', got %r" % kind)
    sys_t, usr_t = _KINDS[kind]
    if kind == "crosslingual":
        system = sys_t.format(tgt=tgt)
        user = usr_t.format(src=src, tgt=tgt, text=text)
    else:
        lang = lang or src
        system = sys_t.format(lang=lang)
        user = usr_t.format(lang=lang, text=text)
        if style:
            user = style + "\n" + user
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Filtering helpers (apply to generated text; see PARAPHRASE_TEMPLATE.md section 5)
# ---------------------------------------------------------------------------
def length_ratio_ok(src, out, lo=0.6, hi=2.0):
    """Cheap termination/coverage filter: drop outputs whose token ratio is out of [lo, hi]."""
    ns = max(1, len(src.split()))
    no = max(0, len(out.split()))
    r = no / ns
    return lo <= r <= hi


def geometric_filter(generated_embs, real_class_embs, thresh):
    """Keep mask by Euclidean distance to the nearest REAL example of the same class.

    ``generated_embs`` (n, d) and ``real_class_embs`` (m, d) are already in the task embedding space
    (e.g. EuroBERT mean, same normalization). Returns ``(keep_mask, distances)``. Simple Euclidean
    distance beat multi-criteria filters in arXiv:2608.13866.
    """
    import numpy as np
    G = np.asarray(generated_embs, dtype=np.float64)
    R = np.asarray(real_class_embs, dtype=np.float64)
    d = np.sqrt(((G[:, None, :] - R[None, :, :]) ** 2).sum(-1)).min(1)
    return d <= thresh, d
