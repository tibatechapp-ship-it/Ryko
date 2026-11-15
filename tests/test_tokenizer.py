"""Tests for the trained Ryko Byte-Level BPE tokenizer."""

import pytest
from tokenizers import Tokenizer

from ryko_lm.tokenizer.loader import RykoTokenizer
from ryko_lm.tokenizer.train_tokenizer import EOS_ID, PAD_ID, BOS_ID, UNK_ID

from conftest import TOKENIZER_JSON, requires_tokenizer


@pytest.fixture(scope="module")
def tok():
    return RykoTokenizer.from_file(TOKENIZER_JSON)


@requires_tokenizer
def test_special_token_ids_are_stable(tok):
    assert tok.pad_id == PAD_ID == 0
    assert tok.bos_id == BOS_ID == 1
    assert tok.eos_id == EOS_ID == 2
    assert tok.unk_id == UNK_ID == 3
    assert tok.vocab_size == 32768


@requires_tokenizer
def test_roundtrip_on_hard_arabic(tok):
    samples = [
        "فاستطعتم كتابة النصوص العربية بسهولة",
        "وبالدفتر، فسأكتبها ولن أقرأها!",
        "اللغة العربية لغة شديدة الإلصاق",
        "مدرسة 123 والرقم ٤٥٦",
        "Ryko reads العربية and code=x1 + emoji ☕",
    ]
    for text in samples:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == tok.tokenizer.normalizer.normalize_str(text), text


@requires_tokenizer
def test_no_unk_is_ever_produced(tok):
    hostile = "\uE000\U0001F680\uFDFD这不是阿拉伯语\u0621"
    ids = tok.encode(hostile)
    assert UNK_ID not in ids


@requires_tokenizer
def test_diacritics_map_to_clean_tokens(tok):
    plain = "الكتاب المدرسة"
    diacritized = "الكِتَابُ المَدْرَسَةِ"
    assert tok.encode(plain) == tok.encode(diacritized)


@requires_tokenizer
def test_encode_special_control(tok):
    ids = tok.encode("جملة", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID


@requires_tokenizer
def test_compression_ratio_reasonable(tok):
    text = TOKENIZER_JSON.read_text(encoding="utf-8")[:0] or (
        "اللغة العربية من أغنى لغات العالم وأكثرها تشكلا في الصرف والاشتقاق "
        "وتحتوي على حروف جر وضمائر متصلة بالكلمات "
    ) * 10
    chars, tokens = len(text), tok.token_count(text)
    ratio = chars / tokens
    assert 1.5 < ratio < 8.0, f"unrealistic ratio chars per token {ratio}"


@requires_tokenizer
def test_tokenizer_json_is_loadable_by_tokenizers_directly():
    t = Tokenizer.from_file(str(TOKENIZER_JSON))
    assert t.get_vocab_size(with_added_tokens=True) == 32768
