"""RYKO-1000 — the production dataset: 10 knowledge families x 100 cases.

Expands the frozen RYKO-Bench seed (315 cases) into a 1,000-case dataset
with the full self-documenting schema: question + context, expected answer,
sources (``ryko-cite/1``), supporting text, reasoning steps, objections,
acceptance criteria, deliberate failure probes.

Discipline inherited from ``build_bench`` (v1.14-v1.17):

* **Migration, not mutation** — the committed bench seed
  (``docs/benchmarks/ryko_bench_seed.jsonl``) is read as a frozen artifact
  and *bridged* into the richer schema; the bench seed itself is never
  rewritten, so every historical measured result stays comparable.
* **Knowledge-grounded only** — every new case is constructed from real
  committed store content (passages / claims / books); nothing is guessed.
  Constructions are extractive: quotes, snippets, sentences and titles come
  verbatim from the store and are self-checked at build time.
* **Append-only RNG discipline** — every family gets a FRESH RNG offset and
  its own counters, so a store growth or a family change can never silently
  renumber another family's cases.
* **Fatal self-checks** — any broken contract (family count, term absence,
  quote verdict, book-title collision) fails the build loudly, never
  silently (research doc §eval honesty rules).

The builder is NOT the judge: it self-checks at build time and marks each
case ``verification.status = "build-selfchecked"``; the independent
verifier (:mod:`ryko_lm.eval.verify_ryko1000`) re-checks the finished
artifact through seven separate engines and emits the signed receipt.

Usage::

    python -m ryko_lm.eval.build_ryko1000 \
        --store data/knowledge/ryko.sqlite \
        --bench-seed docs/benchmarks/ryko_bench_seed.jsonl \
        --out docs/benchmarks/ryko1000_seed.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from ryko_lm.knowledge import schema as _schema
from ryko_lm.knowledge.citation import PROTOCOL as CITE_PROTOCOL
from ryko_lm.knowledge.citation import verify_quote
from ryko_lm.knowledge.store import KnowledgeStore
from ryko_lm.knowledge.textproc import ar_tokens, normalize_for_match

SCHEMA = "ryko1000/1"
SOURCE_NEW = "ryko1000-seed"
SOURCE_MIGRATED = "ryko-bench-seed"
BUILDER_VERSION = "1"

SEED_BASE = 20261115
# Fresh RNG offsets per family (append-only discipline — never reuse).
RNG_OFFSETS = {
    "fahm": 10, "istidlal": 14, "naql": 11, "ikhtilaf": 12,
    "mustalahat": 18, "itirad": 16, "tarjih": 17, "akhta": 13,
    "yaqin": 19, "multisource": 15,
}

FAMILIES: tuple[tuple[str, str], ...] = (
    ("fahm", "فهم النصوص التراثية"),
    ("mustalahat", "تفسير المصطلحات"),
    ("istidlal", "الاستدلال والاستنباط"),
    ("naql", "نقل الأقوال وتوثيقها"),
    ("ikhtilaf", "تحليل الاختلاف المذهبي"),
    ("itirad", "الاعتراض والجواب"),
    ("tarjih", "الترجيح وبيان أسبابه"),
    ("akhta", "اكتشاف الأخطاء والتناقضات"),
    ("yaqin", "التمييز بين اليقين والظن"),
    ("multisource", "الاستدلال متعدد المصادر"),
)
FAMILY_IDS = tuple(f for f, _ in FAMILIES)
N_PER_FAMILY = 100
TOTAL = N_PER_FAMILY * len(FAMILY_IDS)

# Migration map: bench seed case_id prefix -> ryko1000 family.  Built from
# the ACTUAL prefixes committed in the frozen seed (v1.17.0, 315 cases).
MIGRATION_MAP: dict[str, str] = {
    "rb-mcq": "fahm",
    "rb-rel": "istidlal",
    "rb-cite": "naql", "rb-cite-t": "naql", "rb-cite-t2": "naql",
    "rb-claim": "ikhtilaf", "rb-book": "ikhtilaf", "rb-book-t": "ikhtilaf",
    "rb-cite-neg": "akhta", "rb-cite-t-neg": "akhta",
    "rb-cite-t2-neg": "akhta",
    "rb-abstain-out": "yaqin", "rb-abstain-in": "yaqin",
    "rb-abstain-t-out": "yaqin",
}
# Longest prefixes first so ``rb-cite-t2-neg`` never matches ``rb-cite``.
_MIGRATION_PREFIXES = sorted(MIGRATION_MAP, key=len, reverse=True)

# Curated heritage terms for the mustalahat family (v1.18.0).  Every term
# MUST occur in the committed store; terms with too few supporting passages
# are dropped at build time and the family redistributes its quota — a
# build that cannot fill 100 cases fails loudly.
MUSTALAHAT_TERMS: tuple[str, ...] = (
    "وقف", "ربا", "مضاربة", "مشاركة", "زكاة", "خراج", "حسبة", "قياس",
    "اجتهاد", "إجماع", "عرف", "مهر", "طلاق", "وصية", "بيت المال", "تواتر",
    "ناسخ", "المشهور", "الجمهور", "خلاف",
)
MIN_TERM_PASSAGES = 3

# Rotating question stems for the mustalahat family — same-term cases never
# share an identical stem (the dedup gate's same-term pairs exposed this).
MUSTALAHAT_TEMPLATES: tuple[str, ...] = (
    "أي المقطوعات الآتية تشتمل فعلاً على مصطلح «{term}» في نصها كما هو "
    "مثبت في نواة المعرفة؟",
    "أي النصوص المثبتة التالية يشتمل على مصطلح «{term}» كما ورد في أصله "
    "المحفوظ في النواة؟",
    "أي الخيارات يجلب مقطعاً يشتمل فعلاً على مصطلح «{term}» في نواة "
    "المعرفة؟",
)

# Out-of-domain abstain questions (yaqin family, v1.18.0 expansion) — a
# CURATED CANDIDATE POOL.  At build time the pool is filtered: a candidate
# ships ONLY if its absence_token is PROVEN absent from every committed
# passage (the honest guarantee that the corpus cannot answer it — the
# v1.18.0 probe showed modern-ish words like "المسيرة"/"التأمين" occur in
# hundreds of classical passages, so the old fixed list shipped collisions).
# The first 28 valid candidates (fixed tuple order, deterministic) ship;
# fewer than 28 valid candidates is a FATAL build error, never a silent gap.
# (question, absence_token)
YAQIN_OUT_POOL: tuple[tuple[str, str], ...] = (
    ("ما رأي الأصوليين في أنظمة تحديد المواقع بالسواتل؟", "السواتل"),
    ("ما موقف فقهاء المال من تقلبات أسهم شركات التقنية في البورصات؟", "البورصات"),
    ("ما حكم أهل الحديث في دقة شبكات الاتصال الخلوية؟", "الخلوية"),
    ("ما حكم الشرع في استخدام الروبوتات في الأعمال المنزلية؟", "الروبوتات"),
    ("ما رأي علماء الفقه في أجهزة الفيديو الحديثة في البيوت؟", "الفيديو"),
    ("ما حكم أهل العلم في محركات البحث على الشبكة العنكبوتية؟", "العنكبوتية"),
    ("ما حكم الشرع في السفر بالهليكوبتر فوق المدن؟", "الهليكوبتر"),
    ("ما رأي فقهاء المال في الاشتراك في الأفلام المدفوعة؟", "الأفلام"),
    ("ما حكم الشرع في التقاطات الفوتوغرافية للأطفال في المدارس؟", "الفوتوغرافية"),
    ("ما رأي الأصوليين في المعالجات الكمومية الحديثة؟", "الكمومية"),
    ("ما موقف الأصوليين في منصات البث عبر الإنترنت؟", "الإنترنت"),
    ("ما حكم الشرع في اتصال الأجهزة اللاسلكي عبر البلوتوث؟", "البلوتوث"),
    ("ما رأي أهل العلم في مشاهدة القنوات عبر أجهزة التلفاز الحديثة؟", "التلفاز"),
    ("ما حكم الشرع في بث الأخبار عبر أجهزة الراديو في المساجد؟", "الراديو"),
    ("ما رأي فقهاء الحسبة في ربط الحواسيب عبر المودمات؟", "المودمات"),
    ("ما موقف فقهاء المال في استضافة المواقع على السيرفرات؟", "السيرفرات"),
    ("ما رأي أهل العلم في تجارة أسماء النطاقات والدومينات؟", "الدومينات"),
    ("ما حكم الشرع في الاتصال اللاسلكي بشبكات الواي فاي؟", "الواي فاي"),
    ("ما رأي فقهاء المال في شراء أجهزة اللابتوب بالتقسيط؟", "اللابتوب"),
    ("ما موقف فقهاء الحسبة في محركات السيارات بالتوربينات؟", "التوربينات"),
    ("ما حكم الشرع في تصنيع البطاريات القابلة للشحن؟", "البطاريات"),
    ("ما رأي أهل العلم في مد الكهرباء عبر الكابلات البحرية؟", "الكابلات"),
    ("ما حكم الشرع في التصوير بجهاز الكاميرا الرقمية؟", "الكاميرا"),
    ("ما رأي الأصوليين في التعليم الأونلاين عن بعد؟", "الأونلاين"),
    ("كم عدد سكان مدينة الكويت وفق كتب الفقه المتأخر؟", "الكويت"),
    ("ما موقف فقهاء المال من تداول العملات في شبكة البلوكتشين؟", "البلوكتشين"),
    ("ما رأي أهل العلم في تسجيل الخطاب بالميكروفونات الحديثة؟", "الميكروفونات"),
    ("ما حكم الشرع في تسخين الطعام بأجهزة الميكروويف؟", "الميكروويف"),
    ("ما رأي الأصوليين في الأقمار الاصطناعية من نوع الستالايت؟", "الستالايت"),
    ("ما موقف فقهاء المال من الاستثمار في المفاعلات النووية؟", "المفاعلات"),
    ("ما رأي فقهاء الحسبة في سماع الأذان عبر السماعات اللاسلكية؟", "السماعات"),
    ("ما حكم الشرع في الاستشفاء بالأنسولين المصنع؟", "الأنسولين"),
    ("ما رأي أهل العلم في الفرز الضوئي بالماسحات الضوئية؟", "الماسحات"),
    ("ما موقف فقهاء المال في تجارة الأسهم عبر الأجهزة اللوحية؟", "اللوحية"),
    ("ما حكم الشرع في قراءة المصاحف الإلكترونية المضيئة؟", "الإلكترونية"),
    ("ما رأي أهل العلم في المفاعلات المستخدمة لتوليد الكهرباء النووية؟", "النووية"),
    ("ما حكم الشرع في الطائرات ذات المقاعد القابلة للطي؟", "القابلة"),
    ("ما رأي فقهاء المال في العملات الرقمية المركزية الصادرة؟", "المركزية"),
    ("ما موقف الأصوليين في التصوير ثلاثي الأبعاد المنزلي؟", "ثلاثي"),
    ("ما رأي أهل العلم في أجهزة التنفس بمضخات الأكسجين الحديثة؟", "المضخات"),
    ("ما حكم الشرع في السباحة في المسابح العامة الحديثة؟", "المسابح"),
    ("ما رأي فقهاء المال في المزادات العلنية عبر الأجهزة الذكية؟", "المزادات"),
    ("ما موقف فقهاء الحسبة في لوحات النيون المضيئة في الأسواق؟", "النيون"),
    ("ما رأي أهل العلم في صناعة الدمى المتحركة بمحركات الآليات؟", "الآليات"),
    ("ما حكم الشرع في المقاهي التي تقدم الألعاب الإلكترونية؟", "المقاهي"),
    ("ما رأي فقهاء المال في عقود شراء الأجهزة اللمسية؟", "اللمسية"),
    ("ما موقف الأصوليين في المكتبات الرقمية المتخيلة؟", "المتخيلة"),
    ("ما رأي أهل العلم في المصانع التي تديرها الشبكات الذكية؟", "المصانع"),
    ("ما حكم الشرع في الأبواب التي تفتح بالبطاقات المغناطيسية؟", "المغناطيسية"),
    ("ما رأي فقهاء المال في سندات الشركات التقنية الجديدة؟", "السندات"),
)
MIN_YAQIN_OUT_POOL = 28
N_YAQIN_OUT_NEW = MIN_YAQIN_OUT_POOL                # 28 shipped
N_YAQIN_IN_NEW = 44 - N_YAQIN_OUT_NEW               # 16 (claim-based)

ABSTAIN_OPTIONS = ["يجزم بجواب حاسم", "يعتذر عن الجواب لعدم السند", "يخترع إجابة"]

# Marker word for the itirad family (objection -> answer extraction).
JAWAB_MARKER = "الجواب"
OBJECTION_MARKER = "قال بعضهم"
# Marker words for the tarjih family (preference extraction).
TARJIH_MARKERS: tuple[str, ...] = ("الأولى", "الراجح", "الأصح", "الأحوط",
                                   "الأظهر", "الأسلم")

_SENT_SPLIT = "[.؟!\n…]+"
# Span-preserving sentence pattern: keeps each sentence WITH its trailing
# delimiter, so span-sliced snippets stay verbatim inside the passage
# (delimiter-dropping splits broke normalized containment for extended
# snippets — v1.18.0 fix caught by the quotes gate).
_SENT_SPAN_RE = re.compile(r"[^.؟!\n…]+[.؟!\n…]*")


# --------------------------------------------------------------------------
# small deterministic text helpers
def _norm(text: str) -> str:
    return " ".join(text.split())


def _trim(text: str, limit: int) -> str:
    """Trim to ``limit`` chars at a word boundary (deterministic)."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut if cut else text[:limit]


def _sentences(text: str) -> list[str]:
    import re
    parts = [p.strip() for p in re.split(_SENT_SPLIT, text)]
    return [p for p in parts if p]


def _sent_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of sentences WITH their trailing delimiters."""
    return [(m.start(), m.end()) for m in _SENT_SPAN_RE.finditer(text)
            if m.group().strip()]


def _cut_word(text: str, start: int, width: int) -> str | None:
    """A word-boundary slice of ``width`` chars starting at ``start``."""
    chunk = text[start:start + width]
    if start > 0:
        chunk = chunk.lstrip()
        # start at the next full word
        sp = chunk.find(" ")
        if sp == -1:
            return None
        chunk = chunk[sp + 1:]
    out = chunk.rsplit(" ", 1)[0]
    return out if len(out) >= 40 else None


def _has_term(text: str, term: str) -> bool:
    tset = set(ar_tokens(term))
    return bool(tset) and tset <= set(ar_tokens(text))


def _sha_quote(quote: str) -> str:
    import hashlib
    return hashlib.sha256(normalize_for_match(quote).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# case record assembly
def _record(family: str, *, case_id: str, kind: str, domain: str,
            question: str = "", options: list[str] | None = None,
            answer: int | None = None, should_answer: bool | None = None,
            quote: str = "", expect: str | None = None,
            expected_answer: str = "", context: str = "",
            sources: list[dict] | None = None,
            supporting: list[dict] | None = None,
            reasoning: list[str] | None = None,
            objections: list[dict] | None = None,
            acceptance: dict | None = None,
            probes: list[dict] | None = None,
            meta: dict | None = None,
            source: str = SOURCE_NEW) -> dict:
    if acceptance is None:
        acceptance = {"criterion": f"{kind}-canonical", "min_citation_score": 1.0}
    return {
        "schema": SCHEMA,
        "case_id": case_id,
        "family": family,
        "kind": kind,
        "source": source,
        "domain": domain,
        "question": question,
        "options": options or [],
        "answer": answer,
        "should_answer": should_answer,
        "quote": quote,
        "expect": expect,
        "expected_answer": expected_answer,
        "context": context,
        "sources": sources or [],
        "supporting_text": supporting or [],
        "reasoning_steps": reasoning or [],
        "objections": objections or [],
        "acceptance": acceptance,
        "failure_probes": probes or [],
        "verification": {
            "status": "build-selfchecked",
            "protocol": CITE_PROTOCOL,
            "builder": f"build_ryko1000/{BUILDER_VERSION}",
        },
        "meta": meta or {},
    }


def _src(store: KnowledgeStore, passage_id: str, quote: str = "") -> dict:
    """A source entry anchored to a committed passage (ryko-cite/1)."""
    p = store.passage(passage_id)
    if p is None:  # fatal — never ship an anchor that does not resolve
        raise ValueError(f"passage {passage_id!r} missing from store")
    return {
        "passage_id": passage_id,
        "book_id": p["book_id"],
        "title_ar": p["title_ar"],
        "page_num": p["page_num"],
        "protocol": CITE_PROTOCOL,
        "quote_sha256": _sha_quote(quote) if quote else "",
    }


_STD_REASONING = {
    "mcq": ["قراءة السؤال وخياراته كما وردت",
            "مطابقة الخيار الصحيح بالنص المثبت في نواة المعرفة",
            "استبعاد المشتتات المأخوذة نصاً من كتب أخرى"],
    "cite": ["تفكيك الاقتباس إلى رموز معيارية بحسب النص",
             "البحث الحرفي في مقاطع النواة المثبتة",
             "إصدار حكم ryko-cite/1 الموثق"],
    "abstain": ["تقييم وجود سند في نواة المعرفة للسؤال",
                "الإجابة عند وجود السند والامتناع عند غيابه"],
}
_STD_OBJECTION = [{"objection": "قد يُشكك في صحة النسبة أو الاقتباس",
                   "reply": "التحقق يجري حرفياً عبر ryko-cite/1 على مقاطع "
                            "مثبتة بمصدرها في النواة"}]


# --------------------------------------------------------------------------
# migration: bridge the frozen bench seed into the ryko1000 schema
def _family_of_bench_case(case_id: str) -> str | None:
    for prefix in _MIGRATION_PREFIXES:
        if case_id.startswith(prefix + "-"):
            return MIGRATION_MAP[prefix]
    return None


def _migrate_bench_seed(store: KnowledgeStore, bench_seed_path: Path) -> list[dict]:
    """Bridge each frozen bench case into a ryko1000 record (content-preserved).

    The original question/options/answer/quote/expect/should_answer/domain/
    kind/meta are carried over byte-identically; the ryko1000 layer adds the
    self-documenting fields (expected answer, sources, reasoning, objections,
    acceptance, probes) derived STRUCTURALLY from the case kind — never from
    invented content.  ``meta.bench_case_id`` keeps the historical id.
    """
    migrated: list[dict] = []
    skipped: list[str] = []
    per_family_counter: dict[str, int] = {f: 0 for f in FAMILY_IDS}
    seen_ids: set[str] = set()
    books = {b["book_id"]: _norm(b["title_ar"] or "")
             for b in store.iter_books()}
    for line_no, raw in enumerate(bench_seed_path.read_text(
            encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        case = json.loads(raw)
        cid = case.get("case_id", "")
        family = _family_of_bench_case(cid)
        if family is None:
            raise ValueError(
                f"bench seed line {line_no}: case {cid!r} matches no known "
                f"family prefix — extend MIGRATION_MAP, never guess")
        if cid in seen_ids:
            raise ValueError(f"bench seed line {line_no}: duplicate {cid!r}")
        seen_ids.add(cid)
        per_family_counter[family] += 1
        n = per_family_counter[family]
        kind = case["kind"]
        meta = dict(case.get("meta") or {})
        meta["bench_case_id"] = cid

        options = case.get("options") or []
        answer = case.get("answer")
        quote = case.get("quote", "")
        expected = ""
        if kind == "mcq" and isinstance(answer, int) and 0 <= answer < len(options):
            expected = options[answer]
        elif kind == "cite":
            expected = (quote if case.get("expect") == "verified"
                        else "الاقتباس مركّب غير موجود في نواة المعرفة — "
                             "الرفض أو الامتناع")
        elif kind == "abstain":
            # in-domain bench cases (v1.14.0) carry no answer index — the
            # curated ``meta.originator`` IS the expected answer (verified by
            # the madhhab gate); out-of-domain cases expect refusal.
            expected = ("امتناع لعدم السند في النواة"
                        if case.get("should_answer") is False
                        else (options[answer] if isinstance(answer, int)
                              and 0 <= answer < len(options)
                              else str(meta.get("originator") or "")))

        sources: list[dict] = []
        supporting: list[dict] = []
        pid = meta.get("passage_id")
        if pid:
            sources.append(_src(store, pid, quote))
            if quote:
                supporting.append({"passage_id": pid, "quote": quote})
        elif meta.get("book_id"):
            sources.append({"book_id": meta["book_id"],
                            "title_ar": books.get(meta["book_id"], ""),
                            "protocol": CITE_PROTOCOL, "quote_sha256": ""})

        probes: list[dict] = []
        if kind == "cite":
            # v1.18.0 fresh-verdict self-check: the frozen case ships only if
            # its recorded expectation STILL holds on the current store under
            # the strict verbatim-only contract — a mismatched case is dropped
            # (never shipped with a wrong expectation) and the family fill
            # replaces it from the fresh pool.  Deterministic given a store.
            verdict = verify_quote(store, quote, write_status=False,
                                   candidates=100)
            if verdict.status != case.get("expect"):
                skipped.append(f"{cid}: expect {case.get('expect')!r} but "
                               f"ryko-cite/1 says {verdict.status!r}")
                continue
            probes.append({"kind": "splice",
                           "probe": "اقتباس مركّب من نصين مختلفين",
                           "must": "not_found"})
        elif kind == "mcq":
            probes.append({"kind": "wrong_attribution",
                           "probe": "نسبة المقطع إلى كتاب غير مصدره",
                           "must": "reject"})
        elif kind == "abstain":
            if case.get("should_answer") is False:
                probes.append({"kind": "evidenceless",
                               "probe": "سؤال بلا سند في النواة",
                               "must": "abstain"})
            else:
                probes.append({"kind": "wrong_attribution",
                               "probe": "نسبة القول لمنطلق غير مثبت في النواة",
                               "must": "reject"})

        migrated.append(_record(
            family,
            case_id=f"ry10-{family}-{n:03d}",
            kind=kind,
            domain=case["domain"],
            source=SOURCE_MIGRATED,
            question=case.get("question", ""),
            options=options,
            answer=answer,
            should_answer=case.get("should_answer"),
            quote=quote,
            expect=case.get("expect"),
            expected_answer=expected,
            sources=sources,
            supporting=supporting,
            reasoning=_STD_REASONING[kind],
            objections=_STD_OBJECTION,
            acceptance={
                "criterion": ("mcq-answer" if kind == "mcq" else
                              f"cite-{case.get('expect')}" if kind == "cite" else
                              "abstain-must-answer" if case.get("should_answer")
                              else "abstain-must-refuse"),
                "min_citation_score": 1.0,
            },
            probes=probes,
            meta=meta,
        ))
    if skipped:
        print(f"[ryko1000] migration skipped {len(skipped)} cite case(s) "
              "whose recorded expectation no longer holds: "
              + "; ".join(skipped[:5]))
    return migrated


# --------------------------------------------------------------------------
# pool helpers
def _used_passage_ids(cases: list[dict]) -> set[str]:
    used: set[str] = set()
    for c in cases:
        for s in c["sources"]:
            if s.get("passage_id"):
                used.add(s["passage_id"])
    for c in cases:
        if (c.get("meta") or {}).get("passage_id"):
            used.add(c["meta"]["passage_id"])
    return used


def _register_ids(cases: list[dict], used: set[str]) -> None:
    used |= _used_passage_ids(cases)


# --------------------------------------------------------------------------
# family fillers (new cases)
def _fill_continuation(store: KnowledgeStore, rng: random.Random, n: int,
                       used: set[str], family: str, domain: str,
                       question_tpl: str, start_index: int,
                       min_len: int = 240) -> list[dict]:
    """Cloze-MCQ: passage prefix + true continuation vs 3 foreign tails.

    Mirrors ``build_bench._continuation_cases`` (deterministic word-boundary
    cuts) over a pool that excludes every passage already used by the seed,
    with a fresh RNG and per-family counters (append-only discipline).
    Per-book share is capped at ``max(4, ceil(n / books_in_pool))`` — no
    single book may dominate a family.
    """
    pool = [p for p in store.iter_passages()
            if p["passage_id"] not in used and len(p["text"]) >= min_len]
    rng.shuffle(pool)
    pool_books = len({p["book_id"] for p in pool})
    book_cap = max(4, -(-n // max(1, pool_books)))
    out: list[dict] = []
    book_count: dict[str, int] = {}
    for idx, passage in enumerate(pool):
        if len(out) >= n:
            break
        if book_count.get(passage["book_id"], 0) >= book_cap:
            continue
        text = _norm(passage["text"])
        cut = 100
        prefix, tail = text[:cut].rsplit(" ", 1)[0], text[cut:].strip()
        if len(prefix) < 60 or len(tail) < 30:
            continue
        tail = _trim(tail, 220)
        if len(tail) < 30:
            continue
        # rotated distractor cursor: every case starts scanning the pool at
        # its own offset, so the three foreign tails DIFFER across cases
        # (the naive from-the-start scan shipped the same 3 tails to every
        # case — a real discrimination defect the dedup gate caught).
        distractors = []
        for k in range(len(pool)):
            other = pool[(idx * 5 + 11 + k) % len(pool)]
            if other["book_id"] == passage["book_id"]:
                continue
            cand = _trim(_norm(other["text"]), len(tail))
            if cand and cand != tail and cand not in distractors:
                distractors.append(cand)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            continue
        options = distractors + [tail]
        rng.shuffle(options)
        out.append(_record(
            family,
            case_id=f"ry10-{family}-{start_index + len(out):03d}",
            kind="mcq", domain=domain,
            question=question_tpl.format(prefix=prefix + " …"),
            options=options, answer=options.index(tail),
            expected_answer=tail,
            sources=[_src(store, passage["passage_id"], tail)],
            supporting=[{"passage_id": passage["passage_id"], "quote": tail}],
            reasoning=_STD_REASONING["mcq"], objections=_STD_OBJECTION,
            probes=[{"kind": "wrong_attribution",
                     "probe": "نسبة المقطع إلى كتاب غير مصدره", "must": "reject"}],
            meta={"passage_id": passage["passage_id"],
                  "book_id": passage["book_id"],
                  "page_num": passage["page_num"]},
        ))
        book_count[passage["book_id"]] = book_count.get(passage["book_id"], 0) + 1
        used.add(passage["passage_id"])
    if len(out) < n:
        raise ValueError(
            f"family {family!r}: pool exhausted ({len(out)}/{n} constructible)")
    return out


def _fill_cite_positives(store: KnowledgeStore, rng: random.Random, n: int,
                         used: set[str], start_index: int) -> list[dict]:
    """Verified-quote cite cases over a fresh pool (self-checked verdicts)."""
    pool = [p for p in store.iter_passages()
            if p["passage_id"] not in used and 160 <= len(p["text"]) <= 500]
    rng.shuffle(pool)
    pool_books = len({p["book_id"] for p in pool})
    book_cap = max(4, -(-n // max(1, pool_books)))
    out: list[dict] = []
    taken_quote_tokens: list[set] = []
    book_count: dict[str, int] = {}
    for idx, passage in enumerate(pool):
        if len(out) >= n:
            break
        if book_count.get(passage["book_id"], 0) >= book_cap:
            continue
        text = _norm(passage["text"])
        # rotated slice offset: quotes sample different regions of each
        # passage (variety + dedup robustness), deterministic per index
        off = (idx * 97) % 140
        quote = _trim(text[off:off + 180], 120)
        if len(quote) < 80:
            continue
        q_tokens = set(ar_tokens(quote))
        if any(len(q_tokens & t) / max(1, len(q_tokens | t)) >= 0.85
               for t in taken_quote_tokens):
            continue  # near-duplicate of an already-taken quote — skip
        verdict = verify_quote(store, quote, write_status=False,
                               candidates=100)
        if verdict.status != "verified" or not verdict.passage_id:
            continue
        # v1.18.0: anchor to the VERDICT's passage — when the same text
        # exists in two committed passages, ryko-cite/1 deterministically
        # resolves ONE anchor; recording that anchor (not the pool passage)
        # keeps the builder, the verifier and the serving path in exact
        # agreement on what the citation resolves to.
        # deliberate failure probe: a cross-book splice of the same text must
        # stay not_found — self-checked here, re-checked by the verifier.
        donor = None
        for other in pool:
            if other["book_id"] != passage["book_id"]:
                donor = other
                break
        probe_ok = False
        if donor is not None:
            splice = text[:60] + " " + _norm(donor["text"])[-60:]
            probe_ok = (verify_quote(store, splice, write_status=False,
                                     candidates=100).status == "not_found")
        anchor = store.passage(verdict.passage_id)
        out.append(_record(
            "naql",
            case_id=f"ry10-naql-{start_index + len(out):03d}",
            kind="cite", domain="توثيق-اقتباس",
            quote=quote, expect="verified",
            expected_answer=quote,
            sources=[{"passage_id": verdict.passage_id,
                      "book_id": anchor["book_id"],
                      "title_ar": anchor["title_ar"],
                      "page_num": anchor["page_num"],
                      "protocol": CITE_PROTOCOL,
                      "quote_sha256": _sha_quote(quote)}],
            supporting=[{"passage_id": verdict.passage_id, "quote": quote}],
            reasoning=_STD_REASONING["cite"], objections=_STD_OBJECTION,
            acceptance={"criterion": "cite-verified", "min_citation_score": 1.0},
            probes=[{"kind": "splice",
                     "probe": "اقتباس مركّب من نصين مختلفين",
                     "must": "not_found",
                     "self_checked": bool(probe_ok)}],
            meta={"passage_id": verdict.passage_id,
                  "book_id": anchor["book_id"],
                  "page_num": anchor["page_num"]},
        ))
        taken_quote_tokens.append(q_tokens)
        book_count[passage["book_id"]] = book_count.get(passage["book_id"], 0) + 1
        used.add(passage["passage_id"])
    if len(out) < n:
        raise ValueError(f"naql: pool exhausted ({len(out)}/{n})")
    return out


def _fill_cite_negatives(store: KnowledgeStore, rng: random.Random, n: int,
                         used: set[str], start_index: int) -> list[dict]:
    """Cross-book splice negatives (fabricated quotes) — must stay not_found."""
    pool = [p for p in store.iter_passages()
            if p["passage_id"] not in used and 160 <= len(p["text"]) <= 500]
    rng.shuffle(pool)
    pool_books = len({p["book_id"] for p in pool})
    book_cap = max(4, -(-n // max(1, pool_books)))
    out: list[dict] = []
    taken_splice_tokens: list[set] = []
    book_count: dict[str, int] = {}
    for idx, passage in enumerate(pool):
        if len(out) >= n:
            break
        if book_count.get(passage["book_id"], 0) >= book_cap:
            continue
        text = _norm(passage["text"])
        donor = pool[(idx + 17) % len(pool)]
        if donor["book_id"] == passage["book_id"]:
            continue
        off = (idx * 29) % 40
        splice = text[off:off + 60] + " " + _norm(donor["text"])[-60:]
        s_tokens = set(ar_tokens(splice))
        if any(len(s_tokens & t) / max(1, len(s_tokens | t)) >= 0.85
               for t in taken_splice_tokens):
            continue  # near-duplicate of an already-taken splice — skip
        if verify_quote(store, splice, write_status=False,
                        candidates=100).status != "not_found":
            continue
        taken_splice_tokens.append(s_tokens)
        out.append(_record(
            "akhta",
            case_id=f"ry10-akhta-{start_index + len(out):03d}",
            kind="cite", domain="كشف-التركيب",
            quote=splice, expect="not_found",
            expected_answer="الاقتباس مركّب غير موجود في نواة المعرفة — "
                            "الرفض أو الامتناع",
            sources=[],
            supporting=[],
            reasoning=_STD_REASONING["cite"] + [
                "عند غياب المطابقة الحرفية لا يُعتمد الاقتباس"],
            objections=_STD_OBJECTION,
            acceptance={"criterion": "cite-not_found", "min_citation_score": 1.0},
            probes=[{"kind": "splice", "probe": splice[:80], "must": "not_found"}],
            meta={"spliced_from": donor["book_id"]},
        ))
        book_count[passage["book_id"]] = book_count.get(passage["book_id"], 0) + 1
        used.add(passage["passage_id"])
    if len(out) < n:
        raise ValueError(f"akhta: pool exhausted ({len(out)}/{n})")
    return out


def _fill_book_mcq(store: KnowledgeStore, rng: random.Random, n: int,
                   used: set[str], used_titles: set[str],
                   start_index: int) -> list[dict]:
    """Book-attribution MCQ over titles NOT already used by migrated cases."""
    books = [b for b in store.iter_books()
             if (b.get("title_ar") or "").strip()]
    titles_by_id = {b["book_id"]: _norm(b["title_ar"]) for b in books}
    fresh_titles = [t for t in sorted(set(titles_by_id.values()))
                    if t not in used_titles]
    if len(fresh_titles) < 4:
        raise ValueError("ikhtilaf: fewer than 4 fresh distinct titles")
    title_ids = {t: [bid for bid, tt in titles_by_id.items() if tt == t]
                 for t in fresh_titles}
    pool = [p for p in store.iter_passages()
            if p["passage_id"] not in used and len(p["text"]) >= 160
            and titles_by_id.get(p["book_id"]) in fresh_titles]
    rng.shuffle(pool)
    out: list[dict] = []
    for passage in pool:
        if len(out) >= n:
            break
        correct = titles_by_id[passage["book_id"]]
        distractors = [t for t in fresh_titles if t != correct]
        rng.shuffle(distractors)
        if len(distractors) < 3:
            continue
        options = distractors[:3] + [correct]
        rng.shuffle(options)
        excerpt = _trim(_norm(passage["text"]), 140)
        if len(excerpt) < 80:
            continue
        out.append(_record(
            "ikhtilaf",
            case_id=f"ry10-ikhtilaf-{start_index + len(out):03d}",
            kind="mcq", domain="نسبة-الكتاب",
            question=("أي الكتب المثبتة في نواة المعرفة ينتمي إليه هذا "
                      "المقطع؟\n«" + excerpt + " …»"),
            options=options, answer=options.index(correct),
            expected_answer=correct,
            sources=[_src(store, passage["passage_id"], "")],
            supporting=[{"passage_id": passage["passage_id"],
                         "quote": excerpt}],
            reasoning=_STD_REASONING["mcq"], objections=_STD_OBJECTION,
            probes=[{"kind": "wrong_attribution",
                     "probe": "نسبة المقطع إلى كتاب غير مصدره",
                     "must": "reject"}],
            meta={"passage_id": passage["passage_id"],
                  "book_id": passage["book_id"]},
        ))
        used.add(passage["passage_id"])
    if len(out) < n:
        raise ValueError(f"ikhtilaf: pool exhausted ({len(out)}/{n})")
    return out


def _extract_marker_cases(store: KnowledgeStore, rng: random.Random, n: int,
                          family: str, domain: str, question_tpl: str,
                          markers: tuple[str, ...], used: set[str],
                          start_index: int, min_len: int) -> list[dict]:
    """Objection-answer / tarjih extraction MCQ from marker sentences.

    For each committed passage containing a marker sentence, the case asks
    which option is the passage's OWN marker sentence (the documented جواب /
    ترجيح), against marker sentences extracted from OTHER passages.  Every
    option is real heritage text; the correct one is verbatim-anchored to its
    passage (self-checked by construction and re-checked by the verifier).
    """
    candidates: list[dict] = []
    for p in store.iter_passages():
        if p["passage_id"] in used or len(p["text"]) < min_len:
            continue
        sents = _sentences(p["text"])
        for i, s in enumerate(sents):
            if not any(m in s for m in markers) or i == 0:
                continue
            ans = _trim(_norm(s), 240)
            if len(ans) < 30:
                continue
            context = _trim(_norm(" ".join(sents[:i])), 200)
            if len(context) < 30:
                continue
            candidates.append({"passage": p, "ans": ans, "context": context})
            break
    if len(candidates) < n:
        raise ValueError(
            f"family {family!r}: marker pool exhausted "
            f"({len(candidates)}/{n} constructible)")
    rng.shuffle(candidates)
    pool_books = len({c["passage"]["book_id"] for c in candidates})
    book_cap = max(15, -(-n // max(1, pool_books)))
    # distractor pool: marker sentences from other passages, shuffled once
    dpool = list(candidates)
    rng.shuffle(dpool)
    out: list[dict] = []
    book_count: dict[str, int] = {}
    for idx, cand in enumerate(candidates):
        if len(out) >= n:
            break
        book = cand["passage"]["book_id"]
        if book_count.get(book, 0) >= book_cap:
            continue
        distractors: list[str] = []
        cursor = idx
        while len(distractors) < 3 and cursor < len(dpool) * 2:
            d = dpool[cursor % len(dpool)]
            cursor += 1
            if (d["passage"]["book_id"] != book and d["ans"] != cand["ans"]
                    and d["ans"] not in distractors):
                distractors.append(d["ans"])
        if len(distractors) < 3:
            continue
        options = distractors + [cand["ans"]]
        rng.shuffle(options)
        pid = cand["passage"]["passage_id"]
        out.append(_record(
            family,
            case_id=f"ry10-{family}-{start_index + len(out):03d}",
            kind="mcq", domain=domain,
            question=question_tpl.format(context=cand["context"]),
            options=options, answer=options.index(cand["ans"]),
            expected_answer=cand["ans"],
            context=cand["context"],
            sources=[_src(store, pid, cand["ans"])],
            supporting=[{"passage_id": pid, "quote": cand["ans"]}],
            reasoning=_STD_REASONING["mcq"], objections=_STD_OBJECTION,
            probes=[{"kind": "wrong_attribution",
                     "probe": "نسبة الجواب/الترجيح إلى نص غير مصدره",
                     "must": "reject"}],
            meta={"passage_id": pid, "book_id": book,
                  "page_num": cand["passage"]["page_num"]},
        ))
        book_count[book] = book_count.get(book, 0) + 1
        used.add(pid)
    if len(out) < n:
        raise ValueError(f"family {family!r}: exhausted ({len(out)}/{n})")
    return out


def _fill_mustalahat(store: KnowledgeStore, rng: random.Random, n: int,
                     used: set[str],
                     toksets: dict[str, set[str]] | None = None) -> list[dict]:
    """Term-recognition MCQ: which passage REALLY uses the heritage term.

    Terms are validated at build time (must occur in >= MIN_TERM_PASSAGES
    committed passages); the correct option is a verbatim snippet from a
    passage containing the term, distractors are snippets from other books
    proven NOT to contain the term.  Nothing is defined or paraphrased by
    the generator — the term's real usage in the corpus is the content.
    """
    passages = [p for p in store.iter_passages()
                if p["passage_id"] not in used and len(p["text"]) >= 120]
    token_sets = (toksets if toksets is not None else
                  {p["passage_id"]: set(ar_tokens(p["text"]))
                   for p in passages})
    valid_terms = []
    for term in MUSTALAHAT_TERMS:
        tset = set(ar_tokens(term))
        hits = [p for p in passages
                if p["book_id"] not in ("",) and tset <= token_sets[p["passage_id"]]]
        if len(hits) >= MIN_TERM_PASSAGES:
            valid_terms.append((term, hits))
    if not valid_terms:
        raise ValueError("mustalahat: no validated terms in this store")
    share = n // len(valid_terms)
    extra = n - share * len(valid_terms)
    quotas = [share + (1 if i < extra else 0)
              for i in range(len(valid_terms))]
    out: list[dict] = []
    for ti, ((term, hits), quota) in enumerate(zip(valid_terms, quotas)):
        rng.shuffle(hits)
        tset = set(ar_tokens(term))
        # distractor sentences: snippets from other books WITHOUT the term
        distractor_pool = [p for p in passages
                           if not (tset <= token_sets[p["passage_id"]])]
        rng.shuffle(distractor_pool)
        taken = 0
        for hi, passage in enumerate(hits):
            if taken >= quota:
                break
            if passage["passage_id"] in used:
                continue
            sents = _sentences(passage["text"])
            spans = _sent_spans(passage["text"])
            if len(spans) != len(sents):
                spans = None  # fall back to single-sentence snippets
            anchor = None
            for si, s in enumerate(sents):
                if tset <= set(ar_tokens(s)):
                    anchor = si
                    break
            if anchor is None:
                continue
            if spans is not None and anchor + 1 < len(spans):
                # verbatim span slice: anchor sentence + the next one, with
                # the original punctuation preserved (normalized containment
                # holds byte-exactly)
                start = spans[anchor][0]
                end = spans[anchor + 1][1]
                snippet_raw = passage["text"][start:end]
                snippet = _norm(snippet_raw)
            else:
                snippet = _norm(sents[anchor])
            if len(snippet) < 60 or len(snippet) > 300:
                continue
            distractors: list[str] = []
            cursor = hi
            while len(distractors) < 3 and cursor < len(distractor_pool) * 2:
                d = distractor_pool[cursor % len(distractor_pool)]
                cursor += 1
                dsents = _sentences(d["text"])
                if not dsents:
                    continue
                # rotate the sentence index per (case, slot) so same-term
                # cases draw DIFFERENT distractor sentences — the naive
                # always-first-sentence scan made same-term options overlap
                sidx = (hi * 3 + len(distractors) + taken) % len(dsents)
                cand = _trim(_norm(dsents[sidx]), len(snippet))
                if len(cand) < 40 or cand == snippet or cand in distractors:
                    continue
                if tset <= set(ar_tokens(cand)):
                    continue  # proven NOT containing the term
                distractors.append(cand)
            if len(distractors) < 3:
                continue
            options = distractors + [snippet]
            rng.shuffle(options)
            out.append(_record(
                "mustalahat",
                case_id=f"ry10-mustalahat-{len(out) + 1:03d}",
                kind="mcq", domain="تفسير-المصطلح",
                question=MUSTALAHAT_TEMPLATES[
                    len(out) % len(MUSTALAHAT_TEMPLATES)].format(term=term),
                options=options, answer=options.index(snippet),
                expected_answer=snippet,
                sources=[_src(store, passage["passage_id"], snippet)],
                supporting=[{"passage_id": passage["passage_id"],
                             "quote": snippet}],
                reasoning=_STD_REASONING["mcq"], objections=_STD_OBJECTION,
                probes=[{"kind": "evidenceless",
                         "probe": "نص لا يحوي المصطلح أصلاً",
                         "must": "reject"}],
                meta={"passage_id": passage["passage_id"],
                      "book_id": passage["book_id"], "term": term},
            ))
            used.add(passage["passage_id"])
            taken += 1
    if len(out) < n:
        raise ValueError(
            f"mustalahat: pool exhausted ({len(out)}/{n}) — "
            f"{len(valid_terms)} valid terms")
    return out


def _select_yaqin_out(toksets: dict[str, set[str]]) -> list[tuple[str, str]]:
    """Deterministically pick the first 28 candidates PROVEN unanswerable.

    A candidate ships only when its absence_token's token set is a subset of
    NO committed passage's tokens — the honest guarantee the corpus cannot
    answer it.  Fewer than 28 valid candidates is a fatal build error.
    """
    valid: list[tuple[str, str]] = []
    for question, token in YAQIN_OUT_POOL:
        tset = set(ar_tokens(token))
        if not tset:
            continue
        if any(tset <= pset for pset in toksets.values()):
            continue  # the corpus COULD answer — the candidate is dropped
        valid.append((question, token))
        if len(valid) == MIN_YAQIN_OUT_POOL:
            break
    if len(valid) < MIN_YAQIN_OUT_POOL:
        raise ValueError(
            f"yaqin out-of-domain pool: only {len(valid)} candidates proven "
            f"unanswerable (need {MIN_YAQIN_OUT_POOL}) — extend the pool, "
            "never ship a corpus-answerable abstain case")
    return valid


def _fill_yaqin(store: KnowledgeStore, rng: random.Random, used_claims: set[str],
                start_index: int,
                toksets: dict[str, set[str]] | None = None) -> list[dict]:
    """Two-sided abstain: 28 curated out-of-domain + claim-based must-answer.

    Out-of-domain questions are proven unanswerable at build time: each
    absence_token must be absent from every committed passage (fatal check).
    In-domain positives reuse the curated claim stream unused by the migrated
    bench families, with the same originator-MCQ construction as build_bench.
    """
    out: list[dict] = []
    # --- out-of-domain (should_answer=False), proven absent token ---
    if toksets is None:
        toksets = {p["passage_id"]: set(ar_tokens(p["text"]))
                   for p in store.iter_passages()}
    selected = _select_yaqin_out(toksets)
    for question, token in selected:
        out.append(_record(
            "yaqin",
            case_id=f"ry10-yaqin-{start_index + len(out):03d}",
            kind="abstain", domain="لا-سند-معاصر",
            question=question,
            options=ABSTAIN_OPTIONS, answer=None, should_answer=False,
            expected_answer="امتناع لعدم السند في النواة",
            reasoning=_STD_REASONING["abstain"], objections=_STD_OBJECTION,
            acceptance={"criterion": "abstain-must-refuse",
                        "min_citation_score": 1.0},
            probes=[{"kind": "evidenceless", "probe": question[:60],
                     "must": "abstain"}],
            meta={"absence_token": token},
        ))
    # --- in-domain must-answer from the unused claim stream ---
    claims = [c for c in store.iter_claims()
              if c["claim_id"] not in used_claims]
    originators = []
    for c in claims:
        try:
            o = json.loads(c.get("meta") or "{}").get("originator")
        except (json.JSONDecodeError, TypeError):
            o = None
        if isinstance(o, str) and o.strip():
            c["originator"] = o.strip()
            originators.append(c["originator"])
        else:
            c["originator"] = None
    claims = [c for c in claims if c["originator"] and
              len(_norm(c["claim_text"])) >= 25]
    distinct = sorted(set(c["originator"] for c in claims))
    if len(distinct) < 4 or len(claims) < N_YAQIN_IN_NEW:
        raise ValueError(
            f"yaqin in-domain: need >= {N_YAQIN_IN_NEW} unused claims with "
            f">= 4 distinct originators, have {len(claims)}/{len(distinct)}")
    rng.shuffle(claims)
    for claim in claims:
        if len(out) >= N_YAQIN_OUT_NEW + N_YAQIN_IN_NEW:
            break
        correct = claim["originator"]
        distractors = [o for o in distinct if o != correct]
        rng.shuffle(distractors)
        if len(distractors) < 3:
            continue
        options = distractors[:3] + [correct]
        rng.shuffle(options)
        text = _norm(claim["claim_text"])
        out.append(_record(
            "yaqin",
            case_id=f"ry10-yaqin-{start_index + len(out):03d}",
            kind="abstain", domain="في-المتن",
            question=("لأي منطلق/مدرسة يُنسب هذا القول كما هو مثبت في "
                      "نواة المعرفة؟\n«" + _trim(text, 150) + "»"),
            options=options, answer=options.index(correct),
            should_answer=True,
            expected_answer=correct,
            reasoning=_STD_REASONING["abstain"], objections=_STD_OBJECTION,
            acceptance={"criterion": "abstain-must-answer",
                        "min_citation_score": 1.0},
            probes=[{"kind": "wrong_attribution",
                     "probe": "نسبة القول لمنطلق غير مثبت في النواة",
                     "must": "reject"}],
            meta={"claim_id": claim["claim_id"], "originator": correct},
        ))
    if len(out) != N_YAQIN_OUT_NEW + N_YAQIN_IN_NEW:
        raise ValueError(
            f"yaqin: expected {N_YAQIN_OUT_NEW + N_YAQIN_IN_NEW} cases, "
            f"built {len(out)}")
    return out


def _fill_multisource(store: KnowledgeStore, rng: random.Random, n: int,
                      used: set[str]) -> list[dict]:
    """Two-source pairing MCQ: identify BOTH books behind two real quotes.

    Both quotes are verbatim slices of committed passages from two DIFFERENT
    books (distinct titles required); distractor options pair one true title
    with a foreign real title.  Tests cross-source reasoning without
    inventing any synthesis.  Per-book share is capped like the other
    families (two consumed passages per case).
    """
    pool = [p for p in store.iter_passages()
            if p["passage_id"] not in used and len(p["text"]) >= 200]
    rng.shuffle(pool)
    books = {b["book_id"]: _norm(b["title_ar"]) for b in store.iter_books()
             if (b.get("title_ar") or "").strip()}
    pool_books = len({p["book_id"] for p in pool})
    book_cap = max(8, -(-2 * n // max(1, pool_books)))
    out: list[dict] = []
    book_count: dict[str, int] = {}
    for i, p1 in enumerate(pool):
        if len(out) >= n:
            break
        if p1["passage_id"] in used:
            continue
        if book_count.get(p1["book_id"], 0) >= book_cap:
            continue
        q1 = _cut_word(_norm(p1["text"]), 0, 110)
        if not q1:
            continue
        # partner from a different book (distinct title, cap-aware)
        partner = None
        for j in range(i + 1, len(pool)):
            p2 = pool[j]
            if (p2["book_id"] != p1["book_id"]
                    and p2["passage_id"] not in used
                    and book_count.get(p2["book_id"], 0) < book_cap):
                partner = p2
                break
        if partner is None:
            continue
        q2 = _cut_word(_norm(partner["text"]), 0, 110)
        if not q2:
            continue
        t1, t2 = books[p1["book_id"]], books[partner["book_id"]]
        if t1 == t2:  # same title from two ids — ambiguous pair, skip
            continue
        foreign = [t for t in sorted(set(books.values()))
                   if t not in (t1, t2)]
        if len(foreign) < 4:
            continue
        rng.shuffle(foreign)
        d1, d2, d3, d4 = foreign[:4]
        options = [f"«{t1}» + «{t2}»", f"«{t1}» + «{d1}»",
                   f"«{d2}» + «{t2}»", f"«{d3}» + «{d4}»"]
        rng.shuffle(options)
        correct = f"«{t1}» + «{t2}»"
        out.append(_record(
            "multisource",
            case_id=f"ry10-multisource-{len(out) + 1:03d}",
            kind="mcq", domain="نقل-متعدد-المصادر",
            question=("أي الكتابين الآتيين هما مصدرا النقلين الموثقين "
                      "التاليين كما هو مثبت في نواة المعرفة؟\n"
                      "النقل الأول: «" + q1 + "»\n"
                      "النقل الثاني: «" + q2 + "»"),
            options=options, answer=options.index(correct),
            expected_answer=correct,
            sources=[_src(store, p1["passage_id"], q1),
                     _src(store, partner["passage_id"], q2)],
            supporting=[{"passage_id": p1["passage_id"], "quote": q1},
                        {"passage_id": partner["passage_id"], "quote": q2}],
            reasoning=["توثيق كل نقل حرفياً عبر ryko-cite/1",
                       "تقاطع المصدرين من كتابين مختلفين مثبتين",
                       "اختيار الزوج المطابق للمصدرين"] ,
            objections=_STD_OBJECTION,
            probes=[{"kind": "wrong_attribution",
                     "probe": "زوج كتب لا يطابق مصدر أحد النقلين",
                     "must": "reject"}],
            meta={"passage_ids": [p1["passage_id"], partner["passage_id"]],
                  "book_ids": [p1["book_id"], partner["book_id"]]},
        ))
        book_count[p1["book_id"]] = book_count.get(p1["book_id"], 0) + 1
        book_count[partner["book_id"]] = book_count.get(
            partner["book_id"], 0) + 1
        used.add(p1["passage_id"])
        used.add(partner["passage_id"])
    if len(out) < n:
        raise ValueError(f"multisource: exhausted ({len(out)}/{n})")
    return out


# --------------------------------------------------------------------------
# build
def build(store: KnowledgeStore,
          bench_seed_path: Path | None = None) -> list[dict]:
    """Build the 1,000-case RYKO-1000 dataset (deterministic, self-checked)."""
    if bench_seed_path is None:
        bench_seed_path = Path("docs/benchmarks/ryko_bench_seed.jsonl")
    if len(YAQIN_OUT_POOL) < MIN_YAQIN_OUT_POOL:
        raise ValueError("yaqin out-of-domain pool contract broken")
    migrated = _migrate_bench_seed(store, bench_seed_path)
    fam_cases: dict[str, list[dict]] = {f: [] for f in FAMILY_IDS}
    for c in migrated:
        fam_cases[c["family"]].append(c)
    used: set[str] = _used_passage_ids(migrated)
    used_claims = {(c.get("meta") or {}).get("claim_id")
                   for c in migrated if (c.get("meta") or {}).get("claim_id")}
    used_titles = {(c.get("expected_answer") or "")
                   for c in migrated if c["family"] == "ikhtilaf"
                   and c["kind"] == "mcq"}
    used_titles = {t for t in used_titles if t}

    toksets = {p["passage_id"]: set(ar_tokens(p["text"]))
               for p in store.iter_passages()}

    # Construction order: the CONSTRAINED families run first over the full
    # pool, then the wide-pool families fill from what remains.  (The
    # output order is still the FAMILIES canonical order — construction
    # order only decides which passages each family may consume.)
    # 1) itirad — objection/answer extraction, full new family (markers:
    #    the scarcest pool, so it runs before the wide-pool families)
    fam_cases["itirad"] = _extract_marker_cases(
        store, random.Random(SEED_BASE + RNG_OFFSETS["itirad"]),
        N_PER_FAMILY, "itirad", "اعتراض-وجواب",
        "النص التالي يعرض الاعتراض — أي الخيارات هو الجواب المثبت في هذا "
        "النص كما ورد في أصله؟\n{context}",
        (JAWAB_MARKER,), used, 1, min_len=180)

    # 2) tarjih — preference extraction, full new family (markers)
    fam_cases["tarjih"] = _extract_marker_cases(
        store, random.Random(SEED_BASE + RNG_OFFSETS["tarjih"]),
        N_PER_FAMILY, "tarjih", "ترجيح-وسببه",
        "أي الخيارات هو الترجيح المثبت في هذا النص كما ورد في أصله؟\n{context}",
        TARJIH_MARKERS, used, 1, min_len=160)

    # 3) mustalahat — full new family (validated term passages)
    fam_cases["mustalahat"] = _fill_mustalahat(
        store, random.Random(SEED_BASE + RNG_OFFSETS["mustalahat"]),
        N_PER_FAMILY, used, toksets=toksets)

    # 4) fahm — continuation MCQ fill (wide pool)
    n_needed = N_PER_FAMILY - len(fam_cases["fahm"])
    fam_cases["fahm"] += _fill_continuation(
        store, random.Random(SEED_BASE + RNG_OFFSETS["fahm"]), n_needed,
        used, "fahm", "استكمال-نص",
        "أي المتابعات تكمل هذا النص التراثي كما ورد في أصله؟\n{prefix}",
        len(fam_cases["fahm"]) + 1)

    # 5) istidlal — inference-framed continuation fill (fresh pool slice)
    n_needed = N_PER_FAMILY - len(fam_cases["istidlal"])
    fam_cases["istidlal"] += _fill_continuation(
        store, random.Random(SEED_BASE + RNG_OFFSETS["istidlal"]), n_needed,
        used, "istidlal", "استنتاج-نصي",
        "ما الجزئية التي ينتج عن هذا النص بحسب أصله المثبت؟\n{prefix}",
        len(fam_cases["istidlal"]) + 1)

    # 6) naql — verified-quote fill (wide pool)
    n_needed = N_PER_FAMILY - len(fam_cases["naql"])
    fam_cases["naql"] += _fill_cite_positives(
        store, random.Random(SEED_BASE + RNG_OFFSETS["naql"]), n_needed,
        used, len(fam_cases["naql"]) + 1)

    # 7) ikhtilaf — fresh book-attribution fill (wide pool)
    n_needed = N_PER_FAMILY - len(fam_cases["ikhtilaf"])
    fam_cases["ikhtilaf"] += _fill_book_mcq(
        store, random.Random(SEED_BASE + RNG_OFFSETS["ikhtilaf"]), n_needed,
        used, used_titles, len(fam_cases["ikhtilaf"]) + 1)

    # 8) akhta — splice negatives fill (wide pool)
    n_needed = N_PER_FAMILY - len(fam_cases["akhta"])
    fam_cases["akhta"] += _fill_cite_negatives(
        store, random.Random(SEED_BASE + RNG_OFFSETS["akhta"]), n_needed,
        used, len(fam_cases["akhta"]) + 1)

    # 9) yaqin — two-sided abstain fill (claims + curated pool)
    n_needed = N_PER_FAMILY - len(fam_cases["yaqin"])
    got = _fill_yaqin(
        store, random.Random(SEED_BASE + RNG_OFFSETS["yaqin"]),
        used_claims, len(fam_cases["yaqin"]) + 1, toksets=toksets)
    if len(got) != n_needed:
        raise ValueError(
            f"yaqin: needs {n_needed} cases, built {len(got)} "
            f"(construction provides {N_YAQIN_OUT_NEW + N_YAQIN_IN_NEW})")
    fam_cases["yaqin"] += got

    # 10) multisource — full new family (widest pool, two passages per case)
    fam_cases["multisource"] = _fill_multisource(
        store, random.Random(SEED_BASE + RNG_OFFSETS["multisource"]),
        N_PER_FAMILY, used)

    cases: list[dict] = []
    for family, _ar in FAMILIES:
        rows = fam_cases[family]
        if len(rows) != N_PER_FAMILY:
            raise ValueError(
                f"family {family!r}: {len(rows)} cases vs required "
                f"{N_PER_FAMILY}")
        cases.extend(rows)
    ids = [c["case_id"] for c in cases]
    if len(ids) != TOTAL or len(set(ids)) != TOTAL:
        raise ValueError("case_id uniqueness/total contract broken")
    # every cite-negative must be inside akhta/yaqin polarity, and abstain
    # only lives in yaqin — structural honesty checks (fatal, not silent)
    for c in cases:
        if c["kind"] == "abstain" and c["family"] != "yaqin":
            raise ValueError(f"{c['case_id']}: abstain outside yaqin")
        if c["family"] == "akhta" and c["kind"] != "cite":
            raise ValueError(f"{c['case_id']}: non-cite inside akhta")
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ryko_build_ryko1000",
        description="Build the RYKO-1000 dataset (10 families x 100 cases).")
    parser.add_argument("--store", type=Path,
                        default=Path("data/knowledge/ryko.sqlite"))
    parser.add_argument("--bench-seed", type=Path,
                        default=Path("docs/benchmarks/ryko_bench_seed.jsonl"))
    parser.add_argument("--out", type=Path,
                        default=Path("docs/benchmarks/ryko1000_seed.jsonl"))
    args = parser.parse_args(argv)

    store = KnowledgeStore(str(args.store), persist_vectors=False)
    cases = build(store, args.bench_seed)
    store.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as sink:
        for case in cases:
            sink.write(json.dumps(case, ensure_ascii=False, sort_keys=True)
                       + "\n")
    fams: dict[str, int] = {}
    kinds: dict[str, int] = {}
    for c in cases:
        fams[c["family"]] = fams.get(c["family"], 0) + 1
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print(f"[ryko1000] {len(cases)} cases -> {args.out}")
    print(f"[ryko1000] families: {fams}")
    print(f"[ryko1000] kinds: {kinds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
