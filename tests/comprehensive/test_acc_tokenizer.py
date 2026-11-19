"""Accuracy — tokenizer (70 tests): byte round-trips, specials, batching.

The trained byte-level BPE tokenizer must reassemble every normalized
string byte-for-byte, keep its special-token contract stable, and agree
with itself across single/batch encodings.  All expectations are derived
from the committed artifact (artifacts/tokenizer/ryko-tokenizer.json) and
the documented normalizer cascade (NFKC + strip tatweel/harakat).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ryko_lm.tokenizer.loader import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    UNK_ID,
    RykoTokenizer,
)
from ryko_lm.tokenizer.normalizers import python_normalize

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_JSON = ROOT / "artifacts" / "tokenizer" / "ryko-tokenizer.json"
requires_tokenizer = pytest.mark.skipif(
    not TOKENIZER_JSON.exists(), reason="tokenizer artifact not committed"
)


@pytest.fixture(scope="module")
def tok():
    return RykoTokenizer.from_file(TOKENIZER_JSON)


# ---------------------------------------------------------------- round trip
_PHRASES = [
    "اللغة العربية بحر زاخر بالمعاني وأدب عريق في كل العصور",
    "قال رسول الله إنما الأعمال بالنيات",
    "الزكاة فريضة من فرائض الإسلام في المال",
    "القياس على الوضع معنى المقياس في الشريعة",
    "أركان الإسلام خمسة شهادة وصلاة وزكاة وصوم وحج",
    "دلائل السنة النبوية من الأحاديث المتفق عليها",
    "النحو قواعد الإعراب في العربية ركن البلاغة",
    "الاقتصاد علم إدارة الموارد النادرة بين الرغبات",
    "الطهارة شرط صحة الصلاة ولا تصح بدونها",
    "الماء الطهور لا ينجس بتغيير أحد خصائصه",
]
_DIACRITICS = "ًٌٍَُِِّْ"


def _round_trip_cases() -> list[str]:
    """60 deterministic texts: 10 phrases x 6 orthographic variants."""
    cases: list[str] = []
    for i, p in enumerate(_PHRASES):
        words = p.split()
        cases.append(p)
        cases.append(" ".join(
            w[:-1] + _DIACRITICS[i % len(_DIACRITICS)] + w[-1] for w in words))
        cases.append(" ".join(w + "\u0640" * ((j % 2) + 1)
                              for j, w in enumerate(words)))
        cases.append(p + " 123 ٠١٢ (!?) «مقتبس» — end.")
        cases.append("\n".join(words[:3]) + "\n\n" + " ".join(words[3:]))
        cases.append("   " + p * 2 + "\t")
    return cases


@requires_tokenizer
@pytest.mark.parametrize("text", _round_trip_cases())
def test_round_trip_bytes(text, tok):
    """decode(encode(x)) == the normalized form, byte for byte."""
    assert tok.decode(tok.encode(text)) == python_normalize(text)


@requires_tokenizer
@pytest.mark.parametrize("text", _round_trip_cases()[::7])  # 9 of the 60
def test_encode_is_deterministic(text, tok):
    assert tok.encode(text) == tok.encode(text)


@requires_tokenizer
@pytest.mark.parametrize("text", _round_trip_cases()[::5])  # 12 of the 60
def test_token_count_matches_ids(text, tok):
    assert tok.token_count(text) == len(tok.encode(text))


# ------------------------------------------------------------------ specials
@requires_tokenizer
def test_special_token_ids_constant(tok):
    assert (PAD_ID, BOS_ID, EOS_ID, UNK_ID) == (0, 1, 2, 3)
    assert SPECIAL_TOKENS == ["<pad>", "<bos>", "<eos>", "<unk>"]


@requires_tokenizer
def test_specials_registered_in_vocab(tok):
    vocab = tok.tokenizer.get_vocab()
    for name in SPECIAL_TOKENS:
        assert name in vocab


@requires_tokenizer
def test_add_bos_and_eos_wrap(tok):
    ids = tok.encode("سلام", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID


@requires_tokenizer
def test_no_specials_by_default(tok):
    text = "نص عربي عادي"
    ids = tok.encode(text)
    assert BOS_ID not in ids and EOS_ID not in ids


@requires_tokenizer
def test_decode_skips_specials(tok):
    ids = tok.encode("سلام عليكم", add_bos=True, add_eos=True)
    assert tok.decode(ids) == tok.decode(tok.encode("سلام عليكم"))


@requires_tokenizer
def test_decode_keep_specials_renders_marker(tok):
    assert "<pad>" in tok.decode([PAD_ID], skip_special=False)


@requires_tokenizer
def test_literal_special_string_encodes_to_special_id(tok):
    assert tok.encode("<pad>")[0] == PAD_ID


@requires_tokenizer
def test_vocab_size_is_32768(tok):
    assert tok.vocab_size == 32768


@requires_tokenizer
def test_all_ids_in_vocab_range(tok):
    for text in _round_trip_cases():
        assert all(0 <= i < tok.vocab_size for i in tok.encode(text))


@requires_tokenizer
def test_meta_records_vocab(tok):
    meta = tok.meta(ROOT / "artifacts" / "tokenizer")
    assert meta.get("vocab_size") == 32768
    assert meta.get("algo") == "ByteLevelBPE"


@requires_tokenizer
def test_empty_text_encodes_empty(tok):
    assert tok.encode("") == []


# -------------------------------------------------------------------- batch
@requires_tokenizer
def test_batch_equals_single_encoding(tok):
    texts = _round_trip_cases()[:20]
    batch = tok.encode_batch(texts)
    for b, t in zip(batch, texts):
        assert b == tok.encode(t)


@requires_tokenizer
def test_batch_handles_empty_and_plain(tok):
    assert tok.encode_batch(["", "سلام"]) == [[], tok.encode("سلام")]


@requires_tokenizer
def test_batch_large_order_preserved(tok):
    texts = [f"النص رقم {i}" for i in range(200)]
    batch = tok.encode_batch(texts)
    assert len(batch) == 200
    for ids, t in zip(batch, texts):
        assert ids == tok.encode(t)
