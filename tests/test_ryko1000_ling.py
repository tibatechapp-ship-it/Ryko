"""RYKO-1000 linguistic expansion contracts (v1.22.0).

Six tests over the appender (``build_ryko1000_ling``) and the contract-scoped
seven-gate verifier:

1. structure / quotas / sequential numbering (3+11+43+43 = 100 per family).
2. dual build byte-identical.
3. frozen-prefix drift rejection (corrupted pin fails loud BEFORE any write).
4. the seven gates on a SYNTHETIC unified seed with contract injection +
   scope proof: the absence-token gate runs on the store it is measured
   against (the committed core), not on the unified store.
5. RNG-offset independence (fresh offsets, never the core families' 10..19).
6. the REAL committed-seed contract: 1,300 cases / 13 x 100 with the frozen
   first 1,000 lines pinned byte-exactly by ``FROZEN_PREFIX_SHA256``.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from ryko_lm.eval.build_ryko1000_ling import (
    AUTHOR_PARTITION,
    FROZEN_PREFIX_LINES,
    FROZEN_PREFIX_SHA256,
    LINGUA_FAMILY_IDS,
    N_AUTHOR,
    N_BOOK,
    N_CONT,
    N_PER_LING_FAMILY,
    N_RULE,
    RNG_OFFSETS,
    build_ling_cases,
    frozen_prefix_hash,
    load_frozen_seed,
)
from ryko_lm.eval.verify_ryko1000 import check_abstention, verify_all
from ryko_lm.knowledge.store import KnowledgeStore

# ---------------------------------------------------------------------------
# synthetic lugha store — 36 authored catalog books x 9 passages + 12 rules
LEX = ("النحو", "الصرف", "البلاغة", "الإعراب", "الإواب", "النعت", "الفعل",
       "المصدر", "الجامد", "المشتق", "الخفض", "الرفع", "التاء", "الميم",
       "الوزن", "القياس", "الاستعارة", "التشبيه", "المجاز", "الكناية",
       "الطباق", "الجناس", "السجع", "التقسيم", "المعجم", "العين", "القهوة",
       "اللغة", "الفصاحة", "الشعر", "النثر", "القافية", "الروي", "البحر",
       "العروض", "المعنى", "البيان", "البدیع", "الرباعي", "الخماسي",
       "الملكية", "الفاعل", "المفعول", "الحروف", "الأسماء", "الضمة",
       "الفتحة", "الكسرة", "السكون", "التشديد", "المدود", "الهمزات",
       "المثال", "الأجوف", "الناقص", "اللفيف", "المهمل", "المعرب",
       "المبني", "الجملة", "الشبهة", "الخبر", "المبتدأ", "كان",
       "إن", "ظن", "الحال", "التوكيد", "البدل", "العطف", "الاستثناء",
       "النداء", "التعجب", "المدح", "الذم", "القسم", "الشرط", "التعليل",
       "النتيجة", "الفصيح", "الركيك", "الأغراب", "العلم", "التركية",
       "الدخيلة", "المعرب", "الجني", "المولد", "الهجين", "المراكب",
       "الصرفي", "النحوي", "اللغوي", "الأدبي", "البلاغي", "النقدي",
       "الاستشهاد", "الإضمار", "التقديم", "التأخير", "الحذف", "الزيادة",
       "التغيير", "الاسكات", "الفصل", "الوصل", "السكتة", "الوقفات",
       "الرجوع", "التفصيل", "التاج", "التكميل", "الإيضاح", "التبيين",
       "التعريف", "التنكير", "العهد", "الاصطلاح", "الأولى", "الراجح",
       "الأصح", "الأحوط", "الأظهر", "الأسلم", "المشهور", "الجمهور",
       "المحققين", "المعتبرين", "الأصحاب", "المنظور", "الوجه")


def _passage_text(b: int, i: int) -> str:
    """A ~300-char lugha passage with a unique 3-word lexical signature."""
    sig = (f"{LEX[(b * 7 + i) % len(LEX)]} {LEX[(b * 13 + i) % len(LEX)]} "
           f"{LEX[(b * 29 + i) % len(LEX)]}")
    # per-passage PRNG draw (deterministic by (b, i)); the 10 lex tokens are
    # spread across BOTH the 100-char prefix region and the tail so the
    # continuation tails (options) stay lexically diverse too — any two
    # passages share ~2-3 of 13 content tokens (Jaccard far below the dedup
    # threshold; real store passages are diverse throughout)
    rot = random.Random(f"lugha-{b}-{i}").sample(LEX, 10)
    return ("في هذا الباب بيان مسألة " + sig + " والكلام في " + rot[0]
            + " و " + rot[1] + " إنما يرجع إلى " + rot[2] + " مع " + rot[3]
            + " والفرق بينهما في " + rot[4] + " واضح، والمشهور عند أصحابنا "
            "أن " + rot[5] + " مبنية على " + rot[6] + " المشهورة في كتب "
            + rot[7] + " والقصد فيها " + rot[8] + " لا " + rot[9]
            + " عند أهل هذا الفن من النحاة واللغويين، وهذا ما قرره "
              "المحققون في هذا الباب")


NB, PB = 36, 9


@pytest.fixture()
def ling_store():
    s = KnowledgeStore(":memory:")
    s.add_source("lugha", "طبقة اللغة", "MIT declared by dataset publisher",
                 "declared-open")
    for b in range(NB):
        author_id = f"a-l{b:02d}"
        s.add_author(author_id, f"الإمام اللغوي رقم {b}",
                     meta={"school": "nahw"})
        s.add_book(f"lugha-{b:03d}", "lugha",
                   f"كتاب اللغة المبين رقم {b}",
                   author_id=author_id, category="لغة/nahw",
                   rights_status="declared-open",
                   meta={"catalog": True})
        for i in range(PB):
            s.add_passage(f"lugha-{b:03d}", _passage_text(b, i),
                          page_num=i, seq=i)
    for c in range(12):
        c1, c2 = LEX[c % len(LEX)], LEX[(c * 5 + 3) % len(LEX)]
        c3 = LEX[(c * 11 + 1) % len(LEX)]
        s.add_claim(
            f"{c1} أصل في باب {c2} والكلام عليه أن المبدأ فيه يعرب ب "
            f"{c3} عند من أخذ بالوجه الأول من قول أصحاب هذا الفن",
            claim_type="قاعدة",
            meta={"layer": "lugha", "school": ("nahw", "sarf", "balagha")[c % 3],
                  "originator": f"المنشأ اللغوي المعتمد رقم {c}",
                  "source": "lugha-manifest"})
    return s


def _build(ling_store):
    return build_ling_cases(ling_store, [])


# ------------------------------------------------------- 1. structure/quotas
def test_ling_structure_quotas_numbering(ling_store):
    cases = _build(ling_store)
    assert len(cases) == N_PER_LING_FAMILY * len(LINGUA_FAMILY_IDS)
    fams = Counter(c["family"] for c in cases)
    for fam in LINGUA_FAMILY_IDS:
        assert fams[fam] == N_PER_LING_FAMILY, fam
    ids = [c["case_id"] for c in cases]
    assert len(set(ids)) == len(ids)
    for fam in LINGUA_FAMILY_IDS:
        expected = [f"ry10-{fam}-{n:03d}"
                    for n in range(1, N_PER_LING_FAMILY + 1)]
        got = [i for i in ids if i.startswith(f"ry10-{fam}-")]
        assert got == expected, fam  # sequential 001..100, no gaps
    fam_cases = {f: [c for c in cases if c["family"] == f]
                 for f in LINGUA_FAMILY_IDS}
    for fam, rows in fam_cases.items():
        patterns = Counter((c.get("meta") or {}).get("ling_pattern")
                           for c in rows)
        assert patterns["rule"] == N_RULE, (fam, patterns)
        assert patterns["author"] == N_AUTHOR, (fam, patterns)
        assert patterns["book"] == N_BOOK, (fam, patterns)
        assert patterns["continuation"] == N_CONT, (fam, patterns)
    for c in cases:
        assert c["kind"] == "mcq", c["case_id"]  # abstain lives only in yaqin
        assert c["verification"]["status"] == "build-selfchecked"
        assert c["failure_probes"]
    # the shared rule pool: no claim attributed twice across the families
    claims = [(c.get("meta") or {}).get("claim_id") for c in cases
              if (c.get("meta") or {}).get("claim_id")]
    assert len(claims) == len(set(claims)) == N_RULE * 3
    # the author partition: irab/sarf/balagha take disjoint catalog slots
    slots = {fam: [] for fam in LINGUA_FAMILY_IDS}
    for c in cases:
        m = c.get("meta") or {}
        if m.get("ling_pattern") == "author":
            slots[c["family"]].append(m["author_id"])
    for fam in LINGUA_FAMILY_IDS:
        lo, hi = AUTHOR_PARTITION[fam]
        want = {f"a-l{b:02d}" for b in range(lo, hi)}
        assert set(slots[fam]) == want, fam
    # per-family book quota cap over the two passage-consuming patterns
    for fam, rows in fam_cases.items():
        shares = Counter()
        for c in rows:
            m = c.get("meta") or {}
            if m.get("ling_pattern") in ("book", "continuation"):
                shares[m["book_id"]] += 1
        total = sum(shares.values())
        assert total == N_BOOK + N_CONT, fam
        assert shares.most_common(1)[0][1] <= 0.4 * total + 1, fam


# ------------------------------------------------------- 2. dual build
def test_ling_dual_build_byte_identical(ling_store):
    a = _build(ling_store)
    b = _build(ling_store)
    sa = "\n".join(json.dumps(c, sort_keys=True, ensure_ascii=False)
                   for c in a)
    sb = "\n".join(json.dumps(c, sort_keys=True, ensure_ascii=False)
                   for c in b)
    assert hashlib.sha256(sa.encode()).hexdigest() == \
        hashlib.sha256(sb.encode()).hexdigest()


# ------------------------------------------------------- 3. prefix contract
def test_ling_prefix_drift_rejected(tmp_path, monkeypatch):
    lines = [json.dumps({"schema": "ryko1000/1", "case_id": f"x-{n:04d}"},
                        sort_keys=True) for n in range(10)]
    seed = tmp_path / "seed.jsonl"
    seed.write_text("\n".join(lines) + "\n", encoding="utf-8")
    real = frozen_prefix_hash(lines)
    monkeypatch.setattr("ryko_lm.eval.build_ryko1000_ling."
                        "FROZEN_PREFIX_LINES", 10)
    monkeypatch.setattr("ryko_lm.eval.build_ryko1000_ling."
                        "FROZEN_PREFIX_SHA256", real)
    assert load_frozen_seed(seed) == lines  # exact pin passes
    monkeypatch.setattr("ryko_lm.eval.build_ryko1000_ling."
                        "FROZEN_PREFIX_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="FROZEN-PREFIX CONTRACT BROKEN"):
        load_frozen_seed(seed)  # corrupted pin fails loud, no write
    truncated = tmp_path / "short.jsonl"
    truncated.write_text("\n".join(lines[:9]) + "\n", encoding="utf-8")
    monkeypatch.setattr("ryko_lm.eval.build_ryko1000_ling."
                        "FROZEN_PREFIX_SHA256", real)
    with pytest.raises(ValueError, match="frozen"):
        load_frozen_seed(truncated)


# --------------------------------------- 4. seven gates, synthetic + scoped
def test_ling_verify_seven_gates_synthetic(ling_store, tmp_path):
    cases = _build(ling_store)
    # one synthetic must-refuse case whose absence token occurs ONLY in the
    # unified store (not in the retrieval core) — the scope proof
    unified = ling_store
    unified.add_passage("lugha-000",
                        _passage_text(0, 0) + " وكلمة ثلاثي الوضوء "
                        "مذكورة هنا في النص اللغوي الموسع",
                        page_num=99, seq=99)
    abstain_case = {
        "schema": "ryko1000/1", "case_id": "ry10-yaqin-999",
        "family": "yaqin", "kind": "abstain",
        "source": "ryko1000-seed", "domain": "لا-سند-معاصر",
        "question": "ما حكم الشرع في وضوء الثلاثي الحديث؟",
        "options": ["يجزم بجواب حاسم", "يعتذر عن الجواب لعدم السند",
                    "يخترع إجابة"],
        "answer": None, "should_answer": False, "quote": "", "expect": None,
        "expected_answer": "امتناع لعدم السند في النواة", "context": "",
        "sources": [], "supporting_text": [], "reasoning_steps": [],
        "objections": [], "acceptance": {"criterion": "abstain-must-refuse",
                                         "min_citation_score": 1.0},
        "failure_probes": [{"kind": "evidenceless", "probe": "وضوء",
                            "must": "abstain"}],
        "verification": {"status": "build-selfchecked", "protocol": "ryko-cite/1",
                         "builder": "test"},
        "meta": {"absence_token": "الوضوء"},
    }
    # "الوضوء" occurs in the EXTENDED unified passage but NOT in the 9
    # canonical passages of the retrieval-core copy below
    core = KnowledgeStore(":memory:")
    core.add_source("lugha", "طبقة اللغة (نواة الاسترجاع)", "MIT", "declared-open")
    for b in range(NB):
        core.add_book(f"lugha-{b:03d}", "lugha",
                      f"كتاب اللغة المبين رقم {b}", category="لغة/nahw")
        for i in range(PB):
            core.add_passage(f"lugha-{b:03d}", _passage_text(b, i),
                             page_num=i, seq=i)
    all_cases = cases + [abstain_case]
    path = tmp_path / "seed.jsonl"
    path.write_text("\n".join(json.dumps(c, sort_keys=True,
                                         ensure_ascii=False)
                              for c in all_cases), encoding="utf-8")
    # scope proof: on the unified store the absence token OCCURS -> the gate
    # must fail; on the retrieval-core store it is proven absent -> pass
    fails_unified = check_abstention(unified, [abstain_case])
    assert fails_unified, "scope proof broken: token should occur there"
    report = verify_all(unified, path,
                        abstain_store=core, cite_store=core,
                        expect_total=len(all_cases),
                        expect_families={f: N_PER_LING_FAMILY
                                         for f in LINGUA_FAMILY_IDS}
                        | {"yaqin": 1})
    assert report["overall"] == "pass", {
        k: v["failures"][:3] for k, v in report["gates"].items()
        if not v["pass"]}
    for name in ("citations", "quotes", "answers", "contradictions",
                 "madhhab", "abstention", "dedup"):
        assert report["gates"][name]["pass"], name


# --------------------------------------------- 5. RNG-offset independence
def test_ling_rng_offsets_independent(ling_store, monkeypatch):
    base = _build(ling_store)
    monkeypatch.setattr("ryko_lm.eval.build_ryko1000_ling.RNG_OFFSETS",
                        {f: o + 1000 for f, o in RNG_OFFSETS.items()})
    shifted = _build(ling_store)
    h1 = hashlib.sha256("\n".join(
        json.dumps(c, sort_keys=True, ensure_ascii=False)
        for c in base).encode()).hexdigest()
    h2 = hashlib.sha256("\n".join(
        json.dumps(c, sort_keys=True, ensure_ascii=False)
        for c in shifted).encode()).hexdigest()
    assert h1 != h2  # fresh offsets change the cases — no silent reuse
    core_offsets = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
    assert set(RNG_OFFSETS.values()).isdisjoint(core_offsets)


# ------------------------------------- 6. the REAL committed-seed contract
def test_ling_committed_seed_contract():
    seed = Path("docs/benchmarks/ryko1000_seed.jsonl")
    lines = [ln for ln in seed.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    assert len(lines) == FROZEN_PREFIX_LINES + 300
    frozen = frozen_prefix_hash(lines)
    assert frozen == FROZEN_PREFIX_SHA256, \
        "the committed first 1,000 lines drifted off the pinned contract"
    cases = [json.loads(ln) for ln in lines]
    fams = Counter(c["family"] for c in cases)
    assert len(fams) == 13
    for fam, n in fams.items():
        assert n == 100, fam
    ling = [c for c in cases if c["family"] in LINGUA_FAMILY_IDS]
    patterns = Counter((c.get("meta") or {}).get("ling_pattern")
                       for c in ling)
    assert patterns == {"rule": 9, "author": 33, "book": 129,
                        "continuation": 129}
    frozen_cases = [json.loads(ln) for ln in lines[:FROZEN_PREFIX_LINES]]
    used_ids = {c["case_id"] for c in frozen_cases}
    ling_ids = {c["case_id"] for c in ling}
    assert not (used_ids & ling_ids)  # append-only: no id reuse
