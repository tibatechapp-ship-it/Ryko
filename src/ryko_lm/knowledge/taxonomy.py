"""Discourse taxonomy — 12 classical speech acts (بحث RYKO §discourse).

Each passage can be labelled with the speech act it performs.  The classifier
shipped here is an honest, deterministic **rule-based scaffold** (keyword cues,
first match by fixed priority): it exists so the schema, the CLI and the
evaluation gates have a working baseline today, and so a trained classifier
(phase 9 alignment) can replace ``classify`` one day without touching the
schema — the label space is closed and frozen.
"""

from __future__ import annotations

from dataclasses import dataclass

# (code, Arabic name) — closed vocabulary; do not extend without a new release.
DISCOURSE_TYPES: tuple[tuple[str, str], ...] = (
    ("report",       "تقرير"),
    ("explanation",  "شرح"),
    ("argumentation","استدلال"),
    ("objection",    "اعتراض"),
    ("reply",        "رد"),
    ("debate",       "مناظرة"),
    ("transmission", "نقل"),
    ("preference",   "ترجيح"),
    ("summary",      "تلخيص"),
    ("fatwa",        "فتوى"),
    ("critique",     "نقد"),
    ("grounding",    "تأصيل"),
)

# Rule table: ordered by priority; first cue hit wins.
# Cues are matched against the diacritics-stripped text.
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("transmission", ("حدثنا", "أخبرنا", "اخبرنا", "روى", "رواه", "نقل", "عن فلان")),
    ("fatwa",        ("سئل", "سؤال", "الفتوى", "فتوى", "يستفتي")),
    ("objection",    ("اعترض", "الاعتراض", "اعتراض", "شبهة", "استشكل")),
    ("reply",        ("فاجاب", "اجاب", "الجواب", "الرد", "فالرد")),
    ("preference",   ("الارجح", "راجح", "الترجيح", "اقوى", "الاولى")),
    ("critique",     ("النقد", "نقد", "ضعيف", "موضوع", "ساقط", "منكر")),
    ("summary",      ("خلاصة", "الحاصل", "المجموع", "تلخيص", "الخلاصة")),
    ("grounding",    ("التاصيل", "تعريف", "اصطلاح", "مبحث", "الاصل في")),
    ("debate",       ("مناظرة", "المناظرة", "السائل", "المخالف", "الخصم")),
    ("argumentation",("لان", "الدليل", "استدل", "الاستدلال", "قياس", "العلة")),
    ("explanation",  ("شرح", "المقصود", "معناه", "اي ان", "بمعنى")),
    ("report",       ("تقرير", "القول", "المذهب", "اختلف", "قال")),
)

# Diacritics are stripped and hamza seats unified before matching (both in
# the text and in the cues), so 'الأرجح' and 'الارجح' always agree.
def _strip(text: str) -> str:
    from ryko_lm.tokenizer.normalizers import python_normalize

    return python_normalize(
        text,
        remove_diacritics=True,
        unify_alef=True,
        unify_yaa=True,
        taa_marbuta_to_haa=True,
    )


@dataclass(frozen=True)
class DiscourseLabel:
    code: str
    name_ar: str
    cue: str


def classify(text: str) -> DiscourseLabel | None:
    """Deterministic rule-based label, or ``None`` when no cue fires."""
    stripped = _strip(text)
    for code, cues in _RULES:
        for cue in cues:
            if _strip(cue) in stripped:
                name_ar = dict(DISCOURSE_TYPES)[code]
                return DiscourseLabel(code=code, name_ar=name_ar, cue=cue)
    return None


def is_valid(code: str) -> bool:
    return code in {c for c, _ in DISCOURSE_TYPES}
