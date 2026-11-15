"""Arabic-aware normalizers for the Ryko tokenizer.

The tokenizer applies exactly the same orthography cascade as the corpus
cleaning stage (see ``ryko_lm.data.clean_corpus.normalize_arabic``), so that
raw user input (diacritics, tatweel, presentation forms) maps onto the same
token space as the training corpus.

Cascade:
    NFKC  ->  delete [tatweel + harakat + dagger alif + Quranic annotation]
          ->  (optional) unify alef / alif maqsura / taa marbuta

The aggressive unifications are off by default — they shrink the vocabulary
but erase orthographic distinctions (hamza seat, feminine marker).
"""

from __future__ import annotations

import re
import unicodedata

from tokenizers import Regex, normalizers

# tatweel + harakat (fathatan..sukun + U+0653..U+065F) + dagger alif + Quran marks
TATWEEL = "\u0640"
DIACRITICS_PATTERN = "[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]"

_ALEF_RE = re.compile("[\u0623\u0625\u0622]")
_YAA_RE = re.compile("\u0649")
_TAA_MARBUTA_RE = re.compile("\u0629")


def python_normalize(
    text: str,
    remove_diacritics: bool = True,
    unify_alef: bool = False,
    unify_yaa: bool = False,
    taa_marbuta_to_haa: bool = False,
) -> str:
    """Reference Python implementation (used by tests and the data pipeline)."""
    text = unicodedata.normalize("NFKC", text)
    if remove_diacritics:
        text = re.sub(DIACRITICS_PATTERN, "", text)
    if unify_alef:
        text = _ALEF_RE.sub("\u0627", text)
    if unify_yaa:
        text = _YAA_RE.sub("\u064A", text)
    if taa_marbuta_to_haa:
        text = _TAA_MARBUTA_RE.sub("\u0647", text)
    return text


def build_normalizer(
    remove_diacritics: bool = True,
    unify_alef: bool = False,
    unify_yaa: bool = False,
    taa_marbuta_to_haa: bool = False,
) -> "normalizers.Normalizer":
    """Build the Rust-side normalizer used inside the ``tokenizers`` pipeline."""
    steps: list = [normalizers.NFKC()]
    if remove_diacritics:
        steps.append(normalizers.Replace(Regex(DIACRITICS_PATTERN), ""))
    if unify_alef:
        steps.append(normalizers.Replace(Regex("[\u0623\u0625\u0622]"), "\u0627"))
    if unify_yaa:
        steps.append(normalizers.Replace(Regex("\u0649"), "\u064A"))
    if taa_marbuta_to_haa:
        steps.append(normalizers.Replace(Regex("\u0629"), "\u0647"))
    return normalizers.Sequence(steps)
