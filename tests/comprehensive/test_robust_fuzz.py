"""Robustness — deterministic fuzz + pathological inputs + guard (45 tests).

The fuzz corpus is seeded (helpers.fuzz_cases) so every run replays the
identical strings: mixed Arabic/Latin/CJK/emoji, control chars, bidi
overrides, zero-width joins, RTL marks and tatweel/harakat soup.  The
contract under test: nothing in the read/normalize/encode/tokenize/embed
stack ever raises, and byte-level BPE still round-trips exactly.
"""

from __future__ import annotations

import pytest
from comprehensive_helpers import fuzz_cases

from ryko_lm.knowledge.embeddings import DIM, embed
from ryko_lm.knowledge.textproc import ar_tokens, normalize_for_match
from ryko_lm.serving.guard import guard_output
from ryko_lm.tokenizer.loader import RykoTokenizer
from ryko_lm.tokenizer.normalizers import python_normalize

TOKENIZER_PATH = "/home/z/my-project/Ryko/artifacts/tokenizer/ryko-tokenizer.json"


@pytest.fixture(scope="module")
def tok():
    return RykoTokenizer.from_file(TOKENIZER_PATH)


# --------------------------------------------- fuzz round-trip (30)
@pytest.mark.parametrize("text", fuzz_cases(30))
def test_fuzz_never_raises_and_round_trips(text, tok):
    ids = tok.encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) and 0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == python_normalize(text)
    assert tok.token_count(text) == len(ids)


@pytest.mark.parametrize("text", fuzz_cases(30))
def test_fuzz_retrieval_stack_total(text):
    toks = ar_tokens(text)
    assert isinstance(toks, list) and all(isinstance(t, str) for t in toks)
    v = embed(text)
    assert len(v) == DIM
    n1 = normalize_for_match(text)
    assert normalize_for_match(n1) == n1  # idempotent on fuzz inputs too


# --------------------------------------------- pathological (10)
def test_empty_string_stack(tok):
    assert tok.encode("") == []
    assert tok.decode([]) == ""
    assert tok.token_count("") == 0
    assert ar_tokens("") == []
    assert all(x == 0.0 for x in embed(""))


def test_whitespace_flood(tok):
    text = " " * 10000
    assert tok.token_count(text) >= 0
    assert ar_tokens(text) == []


def test_huge_single_word(tok):
    text = "ا" * 100000
    ids = tok.encode(text)
    assert len(ids) >= 1
    assert tok.decode(ids) == text  # byte-level BPE reassembles


def test_diacritics_flood(tok):
    text = "ًٌٍَُِّْ" * 2000
    assert python_normalize(text) == ""
    assert tok.token_count(text) >= 0


def test_tatweel_only(tok):
    text = "\u0640" * 5000
    assert python_normalize(text) == ""


def test_nul_byte_survives(tok):
    text = "نص\x00مقطوع"
    ids = tok.encode(text)
    assert tok.decode(ids) == python_normalize(text)


def test_bidi_override_survives(tok):
    text = "\u202eنص معكوس\u202c"
    ids = tok.encode(text)
    assert tok.decode(ids) == python_normalize(text)


def test_mixed_repeated_runes(tok):
    text = "ا" * 5000 + "ب" * 5000
    ids = tok.encode(text)
    assert len(ids) >= 1


def test_newline_flood(tok):
    text = "\n" * 5000
    assert tok.token_count(text) >= 0


def test_emoji_flood_round_trip(tok):
    text = "🚀" * 500
    assert tok.decode(tok.encode(text)) == python_normalize(text)


# --------------------------------------------- guard verdicts (5)
def test_guard_allows_healthy_arabic():
    verdict = guard_output("قال رسول الله إنما الأعمال بالنيات وكتبها")
    assert verdict["decision"] == "ALLOW" and verdict["reasons"] == []


def test_guard_flags_latin_drift():
    verdict = guard_output("completely english output drift text here")
    assert verdict["decision"] == "REVIEW"
    assert any(r.startswith("low_arabic_ratio") for r in verdict["reasons"])


def test_guard_flags_repetition_loop():
    verdict = guard_output("قال قال قال قال قال قال قال قال قال قال")
    assert verdict["decision"] == "REVIEW"
    assert any(r.startswith("repetition_loop") for r in verdict["reasons"])


def test_guard_flags_residual_markup():
    verdict = guard_output("نص يحمل علامة التحرير %~1# داخل الجملة الطويلة")
    assert verdict["decision"] == "REVIEW"
    assert "residual_markup" in verdict["reasons"]


def test_guard_flags_too_short_and_seals():
    verdict = guard_output("قصير")
    assert "too_short" in verdict["reasons"]
    assert len(verdict["seal"]) == 64
