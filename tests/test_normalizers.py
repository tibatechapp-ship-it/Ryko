"""Tests for the Ryko Arabic normalizer cascade (Python reference impl)."""

import unicodedata

from ryko_lm.data.clean_corpus import normalize_arabic
from ryko_lm.tokenizer.normalizers import DIACRITICS_PATTERN, python_normalize

import re


def test_nfkc_folds_presentation_forms():
    lam_alef_ligature = "\uFEFB"  # lam-alef presentation form
    assert normalize_arabic(lam_alef_ligature) == unicodedata.normalize("NFKC", lam_alef_ligature)


def test_removes_tatweel_and_diacritics():
    text = "مُدَرَّسٌ" + "\u0640"
    out = normalize_arabic(text)
    assert "\u0640" not in out          # tatweel gone
    assert re.search(DIACRITICS_PATTERN, out) is None  # harakat gone
    assert out.startswith("مدرس")


def test_keep_diacritics_option():
    text = "كَتَبَ"
    assert normalize_arabic(text, remove_diacritics=False) != "كتب"
    assert normalize_arabic(text) == "كتب"


def test_aggressive_unifications_are_opt_in():
    text = "أإآىة"
    assert normalize_arabic(text) == text  # default: orthography preserved
    unified = normalize_arabic(text, unify_alef=True, unify_yaa=True, taa_marbuta_to_haa=True)
    assert unified == "ااايه"


def test_python_and_reference_agree():
    for text in ["الْعَرَبِيَّةُ", "وَالْكِتَابُ", "مَدْرَسَةٌ \u0640 123"]:
        assert python_normalize(text) == normalize_arabic(text)
