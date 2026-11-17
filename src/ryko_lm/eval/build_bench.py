"""Build the RYKO-Bench seed from the committed Knowledge Core.

Every case is **self-checked at build time**: verified-quote cases only ship
if ``verify_quote`` actually returns ``verified``; negatives only ship if the
verdict is ``not_found``; MCQ positives must out-score the distractors with a
plain-text criterion (longest-common-token overlap) before being included;
attribution MCQs reuse the curated ``claims.meta.originator``; relation MCQs
come only from documented ``argument_relations`` rows (v1.10.0 manifest).
Deterministic: fixed RNG seed, stable sort — same store, same benchmark.

The result is a *seed*, not a headline metric: auto-generated cases still
need human review before they back any release claim (research doc §eval).

Usage::

    python -m ryko_lm.eval.build_bench \
        --store data/knowledge/ryko.sqlite \
        --out docs/benchmarks/ryko_bench_seed.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from ryko_lm.knowledge import schema as _schema
from ryko_lm.knowledge.citation import verify_quote
from ryko_lm.knowledge.relations import RELATION_AR, list_relations
from ryko_lm.knowledge.store import KnowledgeStore
from ryko_lm.knowledge.textproc import ar_tokens

SEED = 20261115
N_CITE_VERIFIED = 48
N_CITE_NEGATIVE = 16
# turath cite family (v1.16.0): self-checked cases from the wide stratified
# selection, generated with a FRESH RNG (SEED + 1) and independent counters
# so every pre-existing case stays byte-identical when the store grows.
N_TURATH_CITE_VERIFIED = 12
N_TURATH_CITE_NEG = 6
# v1.17.0 families: the same append-only discipline as v1.16.0 — fresh RNGs
# (SEED + 2 / SEED + 3) and independent id counters, so the first 277 seed
# lines stay byte-identical and every historical case_id keeps its measured
# result across releases.
N_TURATH2_CITE_VERIFIED = 12
N_TURATH2_CITE_NEG = 6
N_TURATH_BOOK_MCQ = 12
N_TURATH_ABSTAIN_OUT = 8

# Frozen turath book sets (v1.17.0 discipline): the append-only families
# are PINNED to the books that existed when the family was created.  Adding
# more turath books in a later release must NOT change an existing family's
# pool — that would silently alter/renumber measured cases and break
# cross-release comparability.  Growth opens a NEW family over the new
# books instead (the rb-cite-t-* family was created on the 28 v1.16.0
# books; rb-cite-t2-* and rb-book-t-* on the 12 added in v1.17.0).  Build
# time self-checks: every frozen id must exist in the store, and the two
# cite-family sets must stay disjoint — violations are fatal, never silent.
TURATH_CITE_T1_BOOK_IDS = frozenset({
    "turath-1031", "turath-1060", "turath-1073", "turath-1080",
    "turath-1122", "turath-1131", "turath-1152", "turath-1173",
    "turath-1214", "turath-1226", "turath-1287", "turath-130",
    "turath-1351", "turath-1427", "turath-1487", "turath-1590",
    "turath-1669", "turath-1714", "turath-2030", "turath-2087",
    "turath-2672", "turath-3218", "turath-4049", "turath-4242",
    "turath-4243", "turath-4484", "turath-4619", "turath-691",
})  # v1.16.0 creation set (28 books)
TURATH_CITE_T2_BOOK_IDS = frozenset({
    "turath-1018", "turath-1024", "turath-1219", "turath-1245",
    "turath-1412", "turath-1721", "turath-2075", "turath-2096",
    "turath-4061", "turath-4524", "turath-561", "turath-8085",
})  # v1.17.0 creation set (12 books)

# All frozen family book sets, in creation order.  Growth discipline: a NEW
# release that commits more turath books ADDS a new frozen set here and
# opens a new family over those books — it must never enlarge an existing
# family's pool.
TURATH_FROZEN_FAMILY_SETS = (
    TURATH_CITE_T1_BOOK_IDS, TURATH_CITE_T2_BOOK_IDS)


def _turath_books_present(store: KnowledgeStore) -> set[str]:
    return {b["book_id"] for b in store.iter_books()
            if b["book_id"].startswith("turath-")}


def verify_frozen_turath_sets(store: KnowledgeStore) -> None:
    """Fatal check for the REAL committed store (CLI build path + test).

    Rule: a frozen family book set must be either FULLY present (healthy
    store) or FULLY absent (store predates that family — synthetic test
    stores degrade gracefully); PARTIAL coverage means a store that once
    carried the family's books lost some — that silently breaks the
    family's measured cases and the receipt, so it fails loudly.  The
    frozen sets must also stay pairwise disjoint.  NOT called from
    ``build`` — synthetic test stores call build() directly.
    """
    present = _turath_books_present(store)
    for label, ids in (("rb-cite-t (v1.16.0)", TURATH_CITE_T1_BOOK_IDS),
                       ("rb-cite-t2 (v1.17.0)", TURATH_CITE_T2_BOOK_IDS)):
        overlap, missing = ids & present, ids - present
        if overlap and missing:
            raise ValueError(
                f"frozen turath books PARTIALLY missing from the store "
                f"({label}): {sorted(missing)[:5]}")
    for i in range(len(TURATH_FROZEN_FAMILY_SETS)):
        for j in range(i + 1, len(TURATH_FROZEN_FAMILY_SETS)):
            overlap = TURATH_FROZEN_FAMILY_SETS[i] & TURATH_FROZEN_FAMILY_SETS[j]
            if overlap:
                raise ValueError("frozen turath family book sets must stay "
                                 f"disjoint: {sorted(overlap)[:5]}")


def _unclaimed_turath_ids(store: KnowledgeStore) -> set[str]:
    """Turath books covered by NO frozen family set — only the FIRST turath
    family absorbs these (v1.16.0 catch-all semantics, keeps synthetic test
    stores working); later families are strictly pinned."""
    claimed: set[str] = set()
    for ids in TURATH_FROZEN_FAMILY_SETS:
        claimed |= ids
    return _turath_books_present(store) - claimed


N_MCQ = 48
N_CLAIM_MCQ = 40
N_RELATION_MCQ = 30
N_BOOK_MCQ = 30
# Abstain set is now two-sided (v1.14.0): out-of-domain questions the gate
# MUST abstain on (should_answer=False) and in-domain questions the model
# MUST answer (should_answer=True) — without the second half, the
# wrong-abstain (over-refusal) rate is structurally unmeasurable, which is
# exactly the v1.12.0/v1.13.0 blind spot (abstain 0% with no counter-metric).
N_ABSTAIN_OUT = 32
N_ABSTAIN_IN = 16
DOMAIN_ABSTAIN_OUT = {"معاصر", "تقنية"}
DOMAIN_ABSTAIN_IN = {"في-المتن"}

# Closed Arabic labels for the argument_relations verbs — derived from the
# canonical schema vocabulary (single source of truth). Order is part of the
# contract: option list and answer index are derived from this fixed tuple,
# never shuffled.
RELATION_LABELS: tuple[tuple[str, str], ...] = tuple(
    (verb, RELATION_AR[verb]) for verb in _schema.RELATIONS)
_RELATION_LABEL_MAP = dict(RELATION_LABELS)

# Questions the heritage corpus cannot answer — correct behaviour is abstain.
# The first 16 are the historical set (v1.11.0, kept verbatim for
# comparability); the next 16 are the v1.14.0 expansion. All are factual
# questions about things that did not exist in classical texts.
ABSTAIN_CASES = (
    ("ما رأي علماء التراث في سرعة المعالجات الرسومية في الحواسيب الحديثة؟", "تقنية"),
    ("كم عدد سكان مدينة الرياض في عام 2030 حسب كتب الماضي؟", "تقنية"),
    ("ما حكم استخدام تطبيقات الهاتف الذكي في تداول العملات الرقمية؟", "معاصر"),
    ("ما هي أحدث نسخة من نظام التشغيل ويندوز حسب مصنفات الحديث؟", "معاصر"),
    ("ما رأي ابن سينا في سرعة الإنترنت فائق النطاق؟", "معاصر"),
    ("كم عدد محطات مترو أنقرة بحسب كتب التاريخ القديمة؟", "معاصر"),
    ("ما رأي فقهاء المذاهب الأربعة في خوارزميات الذكاء الاصطناعي التوليدي؟", "معاصر"),
    ("ما هي كلفة تذكرة الطيران بين جدة والقاهرة في مصنف اللغة؟", "معاصر"),
    ("ما موقف أئمة الحديث من سرعة المعالجات الكمومية ومعدل نقل البيانات؟", "معاصر"),
    ("كم عدد أقمار زحل الطبيعية وفق كتب أصول الفقه المقارنة؟", "معاصر"),
    ("ما رأي ابن تيمية في تحديثات برمجيات أنظمة التشغيل الحديثة؟", "معاصر"),
    ("كيف تُحتسب رسوم الشحن الدولي في مصنفات كتاب الخراج؟", "معاصر"),
    ("ما حكم فقهاء الحسبة على تشفير الاتصالات عبر شبكة الجوال؟", "معاصر"),
    ("ما رأي أصحاب الموافقات في خوارزميات محركات البحث الحديثة؟", "معاصر"),
    ("ما رأي أصوليي الشافعية في أسعار أسهم البورصة الإلكترونية اليوم؟", "معاصر"),
    ("كم عدد لافتات الطرق السريعة في بغداد وفق كتب الحسبة القديمة؟", "تقنية"),
    # ---- v1.14.0 expansion (16) ----
    ("ما رأي علماء الحديث في دقة شبكات الاتصال الخلوية من الجيل الخامس؟", "تقنية"),
    ("كم عدد خوادم مراكز البيانات العالمية وفق كتب الأصول القديمة؟", "تقنية"),
    ("ما موقف فقهاء الحسبة من إشارات المرور الضوئية في التقاطعات الحديثة؟", "معاصر"),
    ("كم عدد أنواع مضادات الفيروسات الحاسوبية في مصنفات الفقه المتأخر؟", "تقنية"),
    ("ما رأي الأصوليين في دقة أنظمة تحديد المواقع بالأقمار الصناعية؟", "تقنية"),
    ("كم عدد مستخدمي شبكات التواصل الاجتماعي بحسب كتب اللغة الموسمية؟", "معاصر"),
    ("ما حكم فقهاء المال في تسعير أسهم الشركات التقنية في البورصات؟", "معاصر"),
    ("ما رأي صاحب الموافقات في معدلات انبعاثات طائرات نقل البضائع؟", "معاصر"),
    ("كم عدد محطات توليد الكهرباء النووية وفق كتب المعالم والآثار؟", "تقنية"),
    ("ما موقف أصوليي الحنابلة من تطبيقات الذكاء الاصطناعي في هواتف اليوم؟", "معاصر"),
    ("كم عدد قمر الاتصالات التجارية في مدار الأرض وفق كتب التراث؟", "تقنية"),
    ("ما رأي فقهاء المغرب في تطبيقات المحافظ الرقمية المشفرة؟", "معاصر"),
    ("كم عدد سرير مستشفيات الطوارئ في القاهرة وفق كتب الرحلات القديمة؟", "معاصر"),
    ("ما رأي صاحب اقتصادنا في تقلبات العملات المشفرة في السوق الحالي؟", "معاصر"),
    ("كم عدد مستخدمي البريد الإلكتروني وفق مصنفات الآداب والتقليد؟", "تقنية"),
    ("ما حكم شيخ الإسلام ابن تيمية في خوارزميات التوصية بالفيديو؟", "معاصر"),
)

# v1.17.0 — out-of-domain abstain from the turath layer's own categories
# (plan item #10b): questions that ask for the stance of a SPECIFIC real
# classical work that is NOT committed to the Knowledge Core.  Asking for
# the position of an absent book is unanswerable from the store by
# construction, so the correct behaviour is abstain — the honest counterpart
# of the modern-topic set above, phrased inside heritage categories.
# Self-checked at build time: every referenced work keyword MUST be absent
# from all committed book titles, or the build fails loudly (a future
# release committing one of these works must replace the case, not keep it).
# (question, domain = real turath category, work keyword)
TURATH_ABSTAIN_OUT_CASES = (
    ("ما نص حاشية ابن عابدين في «رد المحتار» عند اشتراط الفصل بين الطلاقات الثلاث؟",
     "تراث/14__الفقه-الحنفي", "رد المحتار"),
    ("ما اختيار ابن عاشور في «التحرير والتنوير» عند تفسير قوله تعالى: ﴿الم﴾؟",
     "تراث/03__التفسير", "التحرير والتنوير"),
    ("ما فرّق عليه ابن الجوزي في «تذكرة الحفاظ» بين الحفظ بالسمع والحفظ بالكتابة؟",
     "تراث/10__علوم-الحديث", "تذكرة الحفاظ"),
    ("ما موقف الغزالي في «إحياء علوم الدين» من إنشاد المتولدين عند الذكر؟",
     "تراث/23__الرقائق-والآداب-والأذكار", "إحياء علوم الدين"),
    ("ما اختيار ابن حجر في «فتح الباري» عند شرح حديث إنما الأعمال بالنيات وقوع النية في القلب؟",
     "تراث/07__شروح-الحديث", "فتح الباري"),
    ("ما نص الحجاوي في «زاد المستقنع» عند اشتراط النية في الطهارة من الحدث الأصغر؟",
     "تراث/17__الفقه-الحنبلي", "زاد المستقنع"),
    ("ما ترتيب النووي في «رياض الصالحين» لأبواب الاستعاذة والرُقي؟",
     "تراث/06__كتب-السنة", "رياض الصالحين"),
    ("على أي وجه أفتى الشوكاني في «فتح القدير» عند المسح على الخفين في السفر؟",
     "تراث/16__الفقه-الشافعي", "فتح القدير"),
)


def _continuation_cases(store: KnowledgeStore, rng: random.Random) -> list[dict]:
    """Cloze-MCQ: a passage prefix + true continuation vs 3 foreign tails."""
    pool = [p for p in _legacy_pool(store) if len(p["text"]) >= 240]
    rng.shuffle(pool)
    out: list[dict] = []
    used_prefixes: set[str] = set()
    for passage in pool:
        if len(out) >= N_MCQ:
            break
        text = " ".join(passage["text"].split())
        cut = 100
        prefix, tail = text[:cut].rsplit(" ", 1)[0], text[cut:].strip()
        if len(prefix) < 60 or len(tail) < 30 or prefix in used_prefixes:
            continue
        if len(tail) > 220:  # keep options inside the trained context window
            tail = tail[:220].rsplit(" ", 1)[0]
        if len(tail) < 30:
            continue
        distractors = []
        for other in pool:
            if other["book_id"] == passage["book_id"] or len(other["text"]) < 240:
                continue
            cand = " ".join(other["text"].split())[:len(tail)]
            if cand and cand != tail:
                distractors.append(cand)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            continue
        options = distractors + [tail]
        rng.shuffle(options)
        out.append({
            "case_id": f"rb-mcq-{len(out) + 1:03d}",
            "kind": "mcq",
            "source": "ryko-bench-seed",
            "domain": "استكمال-نص",
            "question": "أي المتابعات تكمل هذا النص التراثي كما ورد في أصله؟\n" + prefix + " …",
            "options": options,
            "answer": options.index(tail),
            "meta": {"passage_id": passage["passage_id"], "book_id": passage["book_id"],
                     "page_num": passage["page_num"]},
        })
        used_prefixes.add(prefix)
    return out


def _claim_originator_rows(store: KnowledgeStore):
    """Claims with a curated originator, deterministic pre-shuffle order."""
    return [c for c in store.iter_claims()
            if (c.get("meta") or "") and _originator_of(c)]


def _claim_families(store: KnowledgeStore, rng: random.Random):
    """Attribution MCQ (v1.7.0+ layers) + in-domain abstain positives.

    Two families share one shuffled claim stream so their case sets are
    disjoint and deterministic: the first ``N_CLAIM_MCQ`` shippable claims
    become accuracy MCQs; the next ``N_ABSTAIN_IN`` become *positive*
    abstain cases (``should_answer=True``) that measure over-refusal —
    questions the model has the curated answer in-front of it and therefore
    must NOT abstain on.

    Self-checked: the correct answer is the stored originator itself;
    distractors are distinct real originators from other claims. Claims
    without an originator or from stores with fewer than two distinct
    originators are skipped — never guessed.
    """
    rows = _claim_originator_rows(store)
    originators = sorted({_originator_of(c) for c in rows})
    if len(originators) < 2:
        return [], []
    rows.sort(key=lambda c: c["claim_id"])
    rng.shuffle(rows)
    mcq_out: list[dict] = []
    abstain_out: list[dict] = []
    for claim in rows:
        correct = _originator_of(claim)
        distractors = [o for o in originators if o != correct]
        rng.shuffle(distractors)
        if len(distractors) < 3:
            continue
        options = distractors[:3] + [correct]
        rng.shuffle(options)
        text = " ".join(claim["claim_text"].split())
        if len(text) < 25:
            continue
        meta = {"claim_id": claim["claim_id"], "book_id": claim["book_id"],
                "originator": correct}
        if len(mcq_out) < N_CLAIM_MCQ:
            mcq_out.append({
                "case_id": f"rb-claim-{len(mcq_out) + 1:03d}",
                "kind": "mcq",
                "source": "ryko-bench-seed",
                "domain": "نسبة-المسألة",
                "question": ("لأي منطلق/مدرسة يُنسب هذا القول كما هو مثبت في "
                             "نواة المعرفة؟\n«" + text[:150] + "»"),
                "options": options,
                "answer": options.index(correct),
                "meta": meta,
            })
        elif len(abstain_out) < N_ABSTAIN_IN:
            abstain_out.append({
                "case_id": f"rb-abstain-in-{len(abstain_out) + 1:03d}",
                "kind": "abstain",
                "source": "ryko-bench-seed",
                "domain": "في-المتن",
                "question": ("لأي منطلق/مدرسة يُنسب هذا القول كما هو مثبت في "
                             "نواة المعرفة؟\n«" + text[:150] + "»"),
                "options": options,
                "should_answer": True,
                "meta": meta,
            })
        if len(mcq_out) >= N_CLAIM_MCQ and len(abstain_out) >= N_ABSTAIN_IN:
            break
    return mcq_out, abstain_out


def _originator_of(claim: dict) -> str | None:
    """Curated originator (school/author) from claim meta — None if absent."""
    try:
        raw = json.loads(claim.get("meta") or "{}")
    except (json.JSONDecodeError, TypeError):
        return None
    originator = raw.get("originator")
    if isinstance(originator, str) and originator.strip():
        return originator.strip()
    return None


def _relation_cases(store: KnowledgeStore) -> list[dict]:
    """Argument-relation MCQ from documented relations (v1.10.0 manifest).

    Only relations whose both endpoints resolve to claim texts become cases;
    the option list is the fixed ``RELATION_LABELS`` tuple (never shuffled)
    so the answer index is stable across builds. Self-checked by refusing
    any relation verb outside the closed label map.
    """
    out: list[dict] = []
    for rel in list_relations(store):
        if len(out) >= N_RELATION_MCQ:
            break
        verb = rel.get("relation")
        label = _RELATION_LABEL_MAP.get(verb)
        src_raw = (rel.get("src") or {}).get("text") or ""
        dst_raw = (rel.get("dst") or {}).get("text") or ""
        src = " ".join(src_raw.split())
        dst = " ".join(dst_raw.split())
        if label is None or len(src) < 20 or len(dst) < 20:
            continue
        options = [lbl for _, lbl in RELATION_LABELS]
        out.append({
            "case_id": f"rb-rel-{len(out) + 1:03d}",
            "kind": "mcq",
            "source": "ryko-bench-seed",
            "domain": "علاقة-حجاج",
            "question": ("القول الأول: «" + src[:110] + "»\n"
                         "القول الثاني: «" + dst[:110] + "»\n"
                         "ما علاقة القول الأول بالثاني كما هي موثقة في "
                         "مزرعة العلاقات؟"),
            "options": options,
            "answer": options.index(label),
            "meta": {"relation_id": rel["relation_id"], "relation": verb,
                     "src_claim": rel["src"]["claim_id"],
                     "dst_claim": rel["dst"]["claim_id"]},
        })
    return out


def _legacy_pool(store: KnowledgeStore) -> list[dict]:
    """Passages of the pre-v1.16.0 store (everything but ``turath-`` books).

    iter_passages orders by (book_id, page_num, seq) and ``turath-`` sorts
    after every existing source prefix, so filtering the wide-selection rows
    out reproduces the exact pre-growth pool sequence — the seed families
    below stay byte-identical on the expanded store.
    """
    return [p for p in store.iter_passages()
            if not p["book_id"].startswith("turath-")]


def _cite_cases(store: KnowledgeStore, rng: random.Random) -> list[dict]:
    """Verified quotes (positives) + cross-book splices (negatives), self-checked.

    Case IDs use separate counters per polarity (``rb-cite-`` / ``rb-cite-neg-``).
    The single shared counter shipped until v1.13.x could collide when a
    negative was numbered while later positives kept arriving
    (``pos + neg + 1`` vs ``pos + 1``) — latent since v1.11.0, exposed by the
    rebuilt v1.13.0 store. Duplicate IDs are loader-fatal, hence this fix.
    """
    pool = [p for p in _legacy_pool(store) if 160 <= len(p["text"]) <= 500]
    rng.shuffle(pool)
    positives: list[dict] = []
    negatives: list[dict] = []
    for idx, passage in enumerate(pool):
        text = " ".join(passage["text"].split())
        quote = text[:120].rsplit(" ", 1)[0]
        if len(quote) < 80:
            continue
        verdict = verify_quote(store, quote, write_status=False)
        if verdict.status == "verified" and len(positives) < N_CITE_VERIFIED:
            positives.append({
                "case_id": f"rb-cite-{len(positives) + 1:03d}",
                "kind": "cite", "source": "ryko-bench-seed", "domain": "توثيق-اقتباس",
                "quote": quote, "expect": "verified",
                "meta": {"passage_id": passage["passage_id"], "book_id": passage["book_id"],
                         "page_num": passage["page_num"]},
            })
        elif len(negatives) < N_CITE_NEGATIVE and positives:
            donor = pool[(idx + 7) % len(pool)]
            if donor["book_id"] == passage["book_id"]:
                continue
            donor_text = " ".join(donor["text"].split())
            splice = text[:60] + " " + donor_text[-60:]
            if verify_quote(store, splice, write_status=False).status == "not_found":
                negatives.append({
                    "case_id": f"rb-cite-neg-{len(negatives) + 1:03d}",
                    "kind": "cite", "source": "ryko-bench-seed", "domain": "توثيق-اقتباس",
                    "quote": splice, "expect": "not_found",
                    "meta": {"spliced_from": donor["book_id"]},
                })
        if len(positives) >= N_CITE_VERIFIED and len(negatives) >= N_CITE_NEGATIVE:
            break
    return positives + negatives


def _book_cases(store: KnowledgeStore, rng: random.Random) -> list[dict]:
    """Book-attribution MCQ (v1.14.0): which committed book owns a passage.

    Passage excerpts come straight from the store; distractors are distinct
    real book titles from *other* books with different titles — never guessed.
    Books sharing a title with the source book are skipped as distractors.
    Self-checked by construction: the excerpt and the title both originate in
    the committed store, and the answer index is derived after shuffling.
    """
    books = [b for b in store.iter_books()
             if (b.get("title_ar") or "").strip()
             and not b["book_id"].startswith("turath-")]
    if len(books) < 4:
        return []
    titles_by_id = {b["book_id"]: " ".join(b["title_ar"].split()) for b in books}
    titles_set = set(titles_by_id.values())
    if len(titles_set) < 4:
        return []
    pool = [p for p in _legacy_pool(store) if len(p["text"]) >= 160]
    rng.shuffle(pool)
    out: list[dict] = []
    used: set[str] = set()
    for passage in pool:
        if len(out) >= N_BOOK_MCQ:
            break
        correct = titles_by_id.get(passage["book_id"])
        if not correct or correct in used:
            continue
        distractors = [t for t in sorted(titles_set) if t != correct]
        rng.shuffle(distractors)
        if len(distractors) < 3:
            continue
        options = distractors[:3] + [correct]
        rng.shuffle(options)
        text = " ".join(passage["text"].split())
        excerpt = text[:140].rsplit(" ", 1)[0]
        if len(excerpt) < 80:
            continue
        out.append({
            "case_id": f"rb-book-{len(out) + 1:03d}",
            "kind": "mcq",
            "source": "ryko-bench-seed",
            "domain": "نسبة-الكتاب",
            "question": ("أي الكتب المثبتة في نواة المعرفة ينتمي إليه هذا "
                         "المقطع؟\n«" + excerpt + " …»"),
            "options": options,
            "answer": options.index(correct),
            "meta": {"passage_id": passage["passage_id"],
                     "book_id": passage["book_id"]},
        })
        used.add(correct)
    return out


def _turath_cite_cases(store: KnowledgeStore) -> list[dict]:
    """Cite family from the wide stratified selection (v1.16.0).

    Same self-checked construction as ``_cite_cases`` (verbatim-verified
    positives + cross-book splices that must stay ``not_found``) but over
    the ``turath-`` pool only, with a FRESH RNG (``SEED + 1``) and its own
    counters (``rb-cite-t-`` / ``rb-cite-t-neg-``) so appending it can never
    renumber or alter a pre-existing case.  The pool is PINNED to the books
    frozen at family creation (``TURATH_CITE_T1_BOOK_IDS``) plus any book no
    frozen family has claimed yet (catch-all for stores predating the
    discipline) — later store growth that freezes its books into a new set
    can never mutate these measured cases.
    """
    rng = random.Random(SEED + 1)
    present = _turath_books_present(store)
    # frozen creation set + any book no family has claimed yet (v1.16.0
    # catch-all semantics for stores predating the discipline)
    pool_ids = (TURATH_CITE_T1_BOOK_IDS & present) | _unclaimed_turath_ids(store)
    pool = [p for p in store.iter_passages()
            if p["book_id"] in pool_ids
            and 160 <= len(p["text"]) <= 500]
    rng.shuffle(pool)
    positives: list[dict] = []
    negatives: list[dict] = []
    for idx, passage in enumerate(pool):
        text = " ".join(passage["text"].split())
        quote = text[:120].rsplit(" ", 1)[0]
        if len(quote) < 80:
            continue
        verdict = verify_quote(store, quote, write_status=False)
        if verdict.status == "verified" and len(positives) < N_TURATH_CITE_VERIFIED:
            positives.append({
                "case_id": f"rb-cite-t-{len(positives) + 1:03d}",
                "kind": "cite", "source": "ryko-bench-seed", "domain": "توثيق-اقتباس",
                "quote": quote, "expect": "verified",
                "meta": {"passage_id": passage["passage_id"], "book_id": passage["book_id"],
                         "page_num": passage["page_num"]},
            })
        elif len(negatives) < N_TURATH_CITE_NEG and positives:
            donor = pool[(idx + 11) % len(pool)]
            if donor["book_id"] == passage["book_id"]:
                continue
            donor_text = " ".join(donor["text"].split())
            splice = text[:60] + " " + donor_text[-60:]
            if verify_quote(store, splice, write_status=False).status == "not_found":
                negatives.append({
                    "case_id": f"rb-cite-t-neg-{len(negatives) + 1:03d}",
                    "kind": "cite", "source": "ryko-bench-seed", "domain": "توثيق-اقتباس",
                    "quote": splice, "expect": "not_found",
                    "meta": {"spliced_from": donor["book_id"]},
                })
        if len(positives) >= N_TURATH_CITE_VERIFIED and len(negatives) >= N_TURATH_CITE_NEG:
            break
    return positives + negatives


def _turath2_cite_cases(store: KnowledgeStore,
                        used_passage_ids: set[str]) -> list[dict]:
    """Second turath cite family (v1.17.0) — the same self-checked
    construction as ``_turath_cite_cases`` with a FRESH RNG (``SEED + 2``)
    and its own counters (``rb-cite-t2-`` / ``rb-cite-t2-neg-``), pinned to
    the books frozen at THIS family's creation (``TURATH_CITE_T2_BOOK_IDS``,
    disjoint from family 1's set — also asserted at build time), so neither
    family's measured cases can ever be mutated by later store growth.
    """
    rng = random.Random(SEED + 2)
    frozen = TURATH_CITE_T2_BOOK_IDS & _turath_books_present(store)
    pool = [p for p in store.iter_passages()
            if p["book_id"] in frozen
            and p["passage_id"] not in used_passage_ids
            and 160 <= len(p["text"]) <= 500]
    rng.shuffle(pool)
    positives: list[dict] = []
    negatives: list[dict] = []
    for idx, passage in enumerate(pool):
        text = " ".join(passage["text"].split())
        quote = text[:120].rsplit(" ", 1)[0]
        if len(quote) < 80:
            continue
        verdict = verify_quote(store, quote, write_status=False)
        if verdict.status == "verified" and len(positives) < N_TURATH2_CITE_VERIFIED:
            positives.append({
                "case_id": f"rb-cite-t2-{len(positives) + 1:03d}",
                "kind": "cite", "source": "ryko-bench-seed", "domain": "توثيق-اقتباس",
                "quote": quote, "expect": "verified",
                "meta": {"passage_id": passage["passage_id"], "book_id": passage["book_id"],
                         "page_num": passage["page_num"]},
            })
        elif len(negatives) < N_TURATH2_CITE_NEG and positives:
            donor = pool[(idx + 13) % len(pool)]
            if donor["book_id"] == passage["book_id"]:
                continue
            donor_text = " ".join(donor["text"].split())
            splice = text[:60] + " " + donor_text[-60:]
            if verify_quote(store, splice, write_status=False).status == "not_found":
                negatives.append({
                    "case_id": f"rb-cite-t2-neg-{len(negatives) + 1:03d}",
                    "kind": "cite", "source": "ryko-bench-seed", "domain": "توثيق-اقتباس",
                    "quote": splice, "expect": "not_found",
                    "meta": {"spliced_from": donor["book_id"]},
                })
        if len(positives) >= N_TURATH2_CITE_VERIFIED and len(negatives) >= N_TURATH2_CITE_NEG:
            break
    return positives + negatives


def _turath_book_cases(store: KnowledgeStore) -> list[dict]:
    """Book-attribution MCQ over the turath layer (v1.17.0) — mirror of
    ``_book_cases`` for the wide stratified selection, with a FRESH RNG
    (``SEED + 3``) and independent ``rb-book-t-`` ids.  The excerpt and the
    correct title both come from the committed store; distractors are real
    titles of OTHER turath books (same-title books skipped).  The pool is
    pinned to the union of the two frozen cite-family book sets (all 40
    books at creation) so later store growth never mutates these cases.
    Fewer than four distinct titles in the frozen set → no cases (never
    guessed)."""
    rng = random.Random(SEED + 3)
    frozen = ((TURATH_CITE_T1_BOOK_IDS | TURATH_CITE_T2_BOOK_IDS)
              & _turath_books_present(store))
    books = [b for b in store.iter_books()
             if b["book_id"] in frozen
             and (b.get("title_ar") or "").strip()]
    if len(books) < 4:
        return []
    titles_by_id = {b["book_id"]: " ".join(b["title_ar"].split()) for b in books}
    titles_set = set(titles_by_id.values())
    if len(titles_set) < 4:
        return []
    pool = [p for p in store.iter_passages()
            if p["book_id"] in frozen and len(p["text"]) >= 160]
    rng.shuffle(pool)
    out: list[dict] = []
    used: set[str] = set()
    for passage in pool:
        if len(out) >= N_TURATH_BOOK_MCQ:
            break
        correct = titles_by_id.get(passage["book_id"])
        if not correct or correct in used:
            continue
        distractors = [t for t in sorted(titles_set) if t != correct]
        rng.shuffle(distractors)
        if len(distractors) < 3:
            continue
        options = distractors[:3] + [correct]
        rng.shuffle(options)
        text = " ".join(passage["text"].split())
        excerpt = text[:140].rsplit(" ", 1)[0]
        if len(excerpt) < 80:
            continue
        out.append({
            "case_id": f"rb-book-t-{len(out) + 1:03d}",
            "kind": "mcq",
            "source": "ryko-bench-seed",
            "domain": "نسبة-الكتاب-التراثي",
            "question": ("أي الكتب المثبتة في طبقة التراث الشامل ينتمي إليه هذا "
                         "المقطع؟\n«" + excerpt + " …»"),
            "options": options,
            "answer": options.index(correct),
            "meta": {"passage_id": passage["passage_id"],
                     "book_id": passage["book_id"]},
        })
        used.add(correct)
    return out


def _turath_abstain_out_cases(store: KnowledgeStore) -> list[dict]:
    """Out-of-domain abstain from heritage categories (v1.17.0, plan #10b).

    Fixed questions asking for the stance of a specific REAL classical work
    that is NOT committed — unanswerable from the store by construction, so
    the correct behaviour is abstain.  Self-checked at build time: every
    referenced work keyword must stay ABSENT from all committed book
    titles; a collision means the work got committed and the case must be
    replaced, so the build fails loudly instead of shipping a broken case.
    """
    if len(TURATH_ABSTAIN_OUT_CASES) != N_TURATH_ABSTAIN_OUT:
        raise ValueError(
            f"turath abstain contract broken: {len(TURATH_ABSTAIN_OUT_CASES)} "
            f"cases vs N_TURATH_ABSTAIN_OUT={N_TURATH_ABSTAIN_OUT}")
    titles = [(b.get("title_ar") or "") for b in store.iter_books()]
    collisions = [(kw, t) for _q, _d, kw in TURATH_ABSTAIN_OUT_CASES
                  for t in titles if kw in t]
    if collisions:
        raise ValueError(
            "turath abstain cases reference committed works — replace them: "
            + "; ".join(f"{kw} -> {t}" for kw, t in collisions[:5]))
    return [{
        "case_id": f"rb-abstain-t-out-{i:03d}",
        "kind": "abstain",
        "source": "ryko-bench-seed",
        "domain": domain,
        "question": question,
        "should_answer": False,
        "options": ["يجزم بجواب حاسم", "يعتذر عن الجواب لعدم السند", "يخترع إجابة"],
        "answer": None,
        "meta": {"referenced_work": kw},
    } for i, (question, domain, kw) in enumerate(TURATH_ABSTAIN_OUT_CASES, 1)]


def build(store: KnowledgeStore) -> list[dict]:
    if len(ABSTAIN_CASES) != N_ABSTAIN_OUT:
        raise ValueError(
            f"abstain contract broken: {len(ABSTAIN_CASES)} cases vs "
            f"N_ABSTAIN_OUT={N_ABSTAIN_OUT}")
    rng = random.Random(SEED)
    claim_cases, abstain_in_cases = _claim_families(store, rng)
    cases = (_cite_cases(store, rng) + _continuation_cases(store, rng)
             + claim_cases + _relation_cases(store) + _book_cases(store, rng))
    for i, (question, domain) in enumerate(ABSTAIN_CASES, 1):
        cases.append({
            "case_id": f"rb-abstain-out-{i:03d}", "kind": "abstain",
            "source": "ryko-bench-seed", "domain": domain,
            "question": question, "should_answer": False,
            "options": ["يجزم بجواب حاسم", "يعتذر عن الجواب لعدم السند", "يخترع إجابة"],
            "answer": None,
        })
    cases.extend(abstain_in_cases)
    # v1.16.0: the turath cite family appends LAST so the committed seed's
    # first 259 lines stay byte-identical and every old case_id keeps its
    # historical result across releases.
    t1 = _turath_cite_cases(store)
    cases.extend(t1)
    # v1.17.0: three more append-only families (fresh RNGs, independent ids)
    # — the first 277 seed lines stay byte-identical, the new families tail.
    used_t1 = {c["meta"]["passage_id"] for c in t1
               if (c.get("meta") or {}).get("passage_id")}
    cases.extend(_turath2_cite_cases(store, used_t1))
    cases.extend(_turath_book_cases(store))
    cases.extend(_turath_abstain_out_cases(store))
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ryko_build_bench", description="Build the self-checked RYKO-Bench seed.")
    parser.add_argument("--store", type=Path, default=Path("data/knowledge/ryko.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks/ryko_bench_seed.jsonl"))
    args = parser.parse_args(argv)

    store = KnowledgeStore(str(args.store), persist_vectors=False)
    # strict frozen-set check on the real build path (synthetic test stores
    # call build() directly and degrade gracefully instead)
    verify_frozen_turath_sets(store)
    cases = build(store)
    store.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as sink:
        for case in cases:
            sink.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    kinds: dict[str, int] = {}
    domains: dict[str, int] = {}
    for case in cases:
        kinds[case["kind"]] = kinds.get(case["kind"], 0) + 1
        if case["kind"] == "abstain":
            flag = "should_answer" if case.get("should_answer") else "abstain"
            kinds[f"abstain:{flag}"] = kinds.get(f"abstain:{flag}", 0) + 1
        domains[case["domain"]] = domains.get(case["domain"], 0) + 1
    print(f"[bench] {len(cases)} cases -> {args.out} ({kinds})")
    print(f"[bench] domains: {domains}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
