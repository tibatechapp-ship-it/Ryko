"""Data-pipeline tests: cleaning + dedup logic on synthetic Arabic text."""

import numpy as np

from ryko_lm.data.clean_corpus import _fingerprint, deduplicate, normalize_arabic
from ryko_lm.data.binarize import _encode_file
from ryko_lm.tokenizer.loader import RykoTokenizer

from conftest import TOKENIZER_JSON, requires_tokenizer


def test_deduplicate_drops_exact_paragraphs():
    seen = set()
    docs = [["سطر أول", "سطر أول", "سطر ثانٍ"]]
    kept, dropped = deduplicate(docs, seen)
    assert dropped == 1
    assert kept[0] == ["سطر أول", "سطر ثانٍ"]


def test_fingerprint_ignores_punctuation_and_spaces():
    a = _fingerprint("مرحبا بالعالم!")
    b = _fingerprint("مرحبا  بالعالم")
    assert a == b


def test_normalize_arabic_pipeline_semantics():
    assert normalize_arabic("وَالْكِتَابُ", remove_diacritics=True) == "والكتاب"
    assert normalize_arabic("\u0640\u0640\u0640", remove_diacritics=True).strip() == ""
    assert "أ" in normalize_arabic("أكل", unify_alef=False)
    assert "أ" not in normalize_arabic("أكل", unify_alef=True)


@requires_tokenizer
def test_binarize_file_roundtrip(tmp_path):
    tok = RykoTokenizer.from_file(TOKENIZER_JSON)
    src = tmp_path / "train.txt"
    src.write_text("اللغة العربية جميلة\n\nوثانية تماما\n", encoding="utf-8")
    tokens = _encode_file(tok, src, )
    arr = np.asarray(tokens, dtype=np.uint16)
    assert arr.size >= 8
    # first paragraph starts with BOS, ends with EOS
    assert arr[0] == tok.bos_id
    decoded = tok.decode(arr.tolist())
    assert "اللغة" in decoded
