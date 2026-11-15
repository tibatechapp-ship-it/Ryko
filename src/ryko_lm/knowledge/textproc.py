"""Arabic text processing for retrieval and citation checking.

The LM tokenizer keeps orthographic distinctions (hamza seats, feminine taa)
because it *generates* text; retrieval has the opposite goal — collapse
spelling variants so a user query finds the heritage wording.  This module
therefore applies a **search normalization**: diacritics/tatweel removal,
alef/maqsura unification, taa-marbuta folding, plus deterministic light
affix stripping (``ال``، ``و``، ``ب``… and ``ها``، ``ات``، ``ون``…).

All functions are pure and deterministic — same input, same output, no RNG,
no network.  Used by the BM25 index, the hashing embedder and the citation
verifier so the three always agree on what "the same word" means.
"""

from __future__ import annotations

import re

from ryko_lm.tokenizer.normalizers import python_normalize

_AR_TOKEN_RE = re.compile(r"[\u0621-\u064A]+")

# Light prefixes (longest first — one pass, at most one prefix removed).
_PREFIXES = ("وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ك", "ل")
# Light suffixes (longest first — at most one suffix removed).
_SUFFIXES = ("كما", "ها", "ان", "ات", "ون", "ين", "ية", "هم", "هن", "كم", "نا", "ه", "ي", "ك")

MIN_TOKEN_LEN = 3  # after stripping; shorter tokens carry too little signal


def normalize_for_match(text: str) -> str:
    """Aggressive normalization for substring citation checks."""
    return re.sub(
        r"\s+",
        " ",
        python_normalize(
            text,
            remove_diacritics=True,
            unify_alef=True,
            unify_yaa=True,
            taa_marbuta_to_haa=True,
        ),
    ).strip()


def _strip_affixes(word: str) -> str:
    for pre in _PREFIXES:
        if word.startswith(pre) and len(word) - len(pre) >= MIN_TOKEN_LEN:
            word = word[len(pre):]
            break
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= MIN_TOKEN_LEN:
            word = word[: -len(suf)]
            break
    return word


def ar_tokens(text: str) -> list[str]:
    """Deterministic Arabic token stream for BM25 / dense hashing."""
    norm = python_normalize(
        text,
        remove_diacritics=True,
        unify_alef=True,
        unify_yaa=True,
        taa_marbuta_to_haa=True,
    )
    out: list[str] = []
    for raw in _AR_TOKEN_RE.findall(norm):
        tok = _strip_affixes(raw)
        if len(tok) >= MIN_TOKEN_LEN:
            out.append(tok)
    return out
