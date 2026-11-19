"""Shared fixtures for the comprehensive 500-test campaign.

The parent tests/conftest.py provides ROOT/TOKENIZER_JSON/FINAL_MODEL and
the skip markers; this file adds the campaign-wide helpers: the seeded
fuzz pool (deterministic across runs and machines) and a synthetic-store
factory for retrieval tests (never touches the committed repo DB).
"""

from __future__ import annotations

import random

import pytest

from ryko_lm.knowledge.store import KnowledgeStore

FUZZ_SEED = 20261119

# character pools for the deterministic fuzz corpus — no surrogate range
# (lone surrogates are not encodable UTF-8 and the Rust tokenizer rejects
# them), no literal special-token strings (they trigger AddedToken
# matching and would break the byte round-trip property under test).
_FUZZ_SEGMENTS = [
    "اللغة", "العربية", "بحر", "زاخر", "بالمعاني", "أدب", "عريق", "في", "كل",
    "العصور", "قال", "رسول", "الله", "الزكاة", "القياس", "السنة", "النبوية",
    "hello", "world", "ryko", "42", "3.14", "GPT", "IBM", "x", "-", "+",
    "＝", "ロンドン", "こんにちは", "🚀", "🌍", "¿", "ñ", "é", "ü", "§", "µ",
    "\u200b", "\u200d", "\u061c", "\u202a", "\u202e", "\x01", "\x7f", "\t",
    "٠١٢", "٣٤٥", "123", "،", "؛", "؟", "…", "«", "»", ":", "(", ")",
    "ً", "ٌ", "ٍ", "َ", "ُ", "ِ", "ّ", "ْ", "\u0640", "\u0670",
]


def fuzz_cases(count: int, min_len: int = 3, max_len: int = 12) -> list[str]:
    """Deterministic mixed-script strings for robustness tests."""
    rng = random.Random(FUZZ_SEED)
    out = []
    for _ in range(count):
        n = rng.randint(min_len, max_len)
        out.append("".join(rng.choice(_FUZZ_SEGMENTS) for _ in range(n)))
    return out


SYNTHETIC_PASSAGES = [
    "الزكاة فريضة من فرائض الإسلام في المال للفقراء",
    "الطهارة شرط صحة الصلاة ولا تصح بدونها",
    "القياس على الوضع معنى المقياس في الشريعة",
    "حديث النبي في فضل الصدقة جاء بسند متصل",
    "الماء الطهور لا ينجس بتغيير أحد خصائصه",
    "فريضة الصيام في رمضان على كل مسلم قادر",
    "دلائل السنة النبوية من الأحاديث المتفق عليها",
    "أركان الإسلام خمسة: الشهادة والصلاة والزكاة والصوم والحج",
    "الشهادة ركن الإسلام الأول وباقي الأركان تبع لها",
    "الصلاة عمود الدين والطهارة مفتاحها",
]


@pytest.fixture()
def synth_store():
    """Small in-memory store (10 passages, one book) for retrieval tests."""
    s = KnowledgeStore(":memory:")
    s.add_source("synth", "Synthetic Bench", "MIT", "declared-open")
    s.add_book("bk1", "synth", "كتاب الاختبارات", category="تجريبي")
    for i, text in enumerate(SYNTHETIC_PASSAGES):
        s.add_passage("bk1", text, page_num=i // 3, seq=i % 3)
    yield s
    s.close()
