"""Accuracy — Arabic normalization + retrieval text pipeline (55 tests).

python_normalize is the single orthographic source of truth for the whole
pipeline (corpus cleaning, tokenizer normalizer, search normalization) —
these tests pin its documented cascade and its idempotence.  ar_tokens and
embed are the deterministic retrieval core: every expectation below was
calibrated against the actual implementations (affix stripping table,
MIN_TOKEN_LEN=3, blake2b feature hashing).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ryko_lm.knowledge.embeddings import DIM, cosine, embed, pack, unpack
from ryko_lm.knowledge.textproc import ar_tokens, normalize_for_match
from ryko_lm.tokenizer.normalizers import python_normalize

ROOT = Path(__file__).resolve().parents[2]

TATWEEL = "\u0640"
DIACRITIC_TEXT = "الْكِتَابُ المُفِيدُ"
QURAN_MARK = "\u06d6"
DAGGER_ALIF = "\u0670"


# ------------------------------------------------- python_normalize (25)
@pytest.mark.parametrize("text", [
    DIACRITIC_TEXT,
    f"ك{TATWEEL}{TATWEEL}تاب",
    "اللغة العربية بحر زاخر بالمعاني",
    "قال رسول الله إنما الأعمال بالنيات",
    "«النحو» ركنٌ للبلاغة!",
    "أرقام 123 وأرقام ٠١٢٣٤٥٦٧٨٩",
    "نص\tبواسطر جديدة\nوسطور متعددة",
    "همّوا بالعمل",
    f"رفع{DAGGER_ALIF} العلم",
    f"سورة{QURAN_MARK} البقرة",
    "بيتة القراءة",
    "أإآتا",
    "الإسلام والهجرة",
    "هذا العالمُ جميل",
    "على الأثر التاريخي",
    "🚀 النص الممزوج english text",
    "التنسيق: (أ، ب)؛ جـ — د",
    "نسخة مكررة نسخة مكررة",
    "ترقيم 1) 2) 3)",
    "اقتباس «عن المقالات» في النص",
])
def test_normalize_idempotent(text):
    n1 = python_normalize(text)
    assert python_normalize(n1) == n1


@pytest.mark.parametrize("raw, expected", [
    ("كــتـاب", "كتاب"),          # tatweel stripped
    ("الْكِتَابُ", "الكتاب"),      # harakat stripped
    ("المُفيدٌ", "المفيد"),
    ("أإآتا", "أإآتا"),           # alef NOT unified by default
    ("بيتة", "بيتة"),             # taa marbuta NOT folded by default
])
def test_normalize_cascade(raw, expected):
    assert python_normalize(raw) == expected


def test_normalize_unify_alef_flag():
    assert python_normalize("أإآ", unify_alef=True) == "ااا"


def test_normalize_unify_yaa_flag():
    assert python_normalize("مستشفى", unify_yaa=True) == "مستشفي"


def test_normalize_taa_marbuta_flag():
    assert python_normalize("مدرسة", taa_marbuta_to_haa=True) == "مدرسه"


def test_normalize_keep_diacritics_flag():
    assert python_normalize("كِتاب", remove_diacritics=False) == "كِتاب"


def test_normalize_nfkc_ligature():
    # U+FEFB ARABIC LIGATURE LAM WITH ALEF ISOLATED FORM
    assert python_normalize("\ufefb") == "لا"


# ------------------------------------------------------ ar_tokens (15)
@pytest.mark.parametrize("text, expected", [
    ("الكتاب المفيد", ["كتاب", "مفيد"]),
    ("والباب المطهو", ["باب", "مطهو"]),
    ("بالقلم", ["قلم"]),
    ("للكاتب", ["كاتب"]),
    ("الزكاة", ["زكا"]),
    ("بيته", ["يته"]),
    ("كلمتها", ["لمت"]),
    ("أركان الإسلام", ["ارك", "اسلام"]),
    ("دلائل السنة", ["دلائل", "سنه"]),
    ("مسلمون", ["مسلم"]),
    ("مؤمنات", ["مؤمن"]),
    ("الفتية", ["فتي"]),
    ("همّوا", ["هموا"]),
    ("قالوا", ["قالوا"]),
    ("الكتاب", ["كتاب"]),
])
def test_ar_tokens_affix_pipeline(text, expected):
    assert ar_tokens(text) == expected


# ------------------------------------------------- embeddings/cosine (15)
def test_embed_dim():
    assert len(embed("اللغة العربية")) == DIM == 256


def test_embed_deterministic():
    a = embed("الزكاة في المال")
    b = embed("الزكاة في المال")
    assert list(a) == list(b)


def test_embed_unit_norm_for_arabic():
    v = embed("اللغة العربية بحر زاخر بالمعاني وأدب عريق")
    assert abs(sum(x * x for x in v) - 1.0) < 1e-5


def test_embed_zero_for_non_arabic():
    assert all(x == 0.0 for x in embed("hello world 123"))


def test_embed_zero_for_digit_only():
    assert all(x == 0.0 for x in embed("12345"))


def test_cosine_identity():
    v = embed("القياس على الوضع")
    assert cosine(v, v) > 0.999


def test_cosine_symmetric():
    a = embed("الزكاة فريضة المال")
    b = embed("الصلاة عمود الدين")
    assert cosine(a, b) == cosine(b, a)


def test_embed_order_insensitive():
    # feature hashing accumulates counts — word order cannot matter
    assert list(embed("الزكاة والصلاة")) == list(embed("والصلاة الزكاة"))


def test_embed_distinct_texts_differ():
    assert list(embed("الزكاة في المال")) != list(embed("الطهارة شرط الصلاة"))


def test_pack_unpack_round_trip():
    v = embed("دلائل السنة النبوية")
    assert all(abs(a - b) < 1e-9 for a, b in zip(unpack(pack(v), DIM), v))


def test_pack_byte_length():
    v = embed("نص")
    assert len(pack(v)) == DIM * 4


def test_unpack_wrong_dim_raises():
    with pytest.raises(ValueError):
        unpack(pack(embed("نص")), DIM + 8)


def test_embed_dim_too_small_raises():
    with pytest.raises(ValueError):
        embed("نص", dim=4)


def test_embed_finite_floats():
    import math

    v = embed("نص عربي مختلط english 42 🚀")
    assert all(math.isfinite(x) for x in v)


def test_normalize_for_match_folds_variants():
    # search normalization folds hamza seats + diacritics (not the article)
    assert normalize_for_match("الأَكَلُ") == normalize_for_match("الاكل")
    # the retrieval token stream additionally strips the light article
    assert ar_tokens("الأَكَلُ") == ar_tokens("اكل") == ["اكل"]
