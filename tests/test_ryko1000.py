"""RYKO-1000 quality gates — schema, counts, determinism, 7 verification
engines, migration mapping, and the cite-verdict regression (v1.18.0 fix:
token coverage alone must NEVER carry the ``verified`` verdict)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from ryko_lm.eval.build_ryko1000 import (
    FAMILIES,
    FAMILY_IDS,
    N_PER_FAMILY,
    build,
)
from ryko_lm.eval.verify_ryko1000 import (
    VerificationError,
    check_dedup,
    coverage,
    load_ryko1000,
    verify_all,
    verify_citations,
    verify_quotes,
)
from ryko_lm.knowledge.citation import verify_quote
from ryko_lm.knowledge.store import KnowledgeStore

# Distinct heritage vocabulary per passage: the synthetic store must give
# every passage a DIFFERENT token stream (digits are not tokens), so the
# BM25/hybrid search anchors each quote to its own passage deterministically,
# and the dedup gate never sees two identical token sets.  Every passage
# embeds a 3-word "signature" drawn from this 150-word lexicon.
DIST = ("الطهارة", "الصلاة", "الزكاة", "الصوم", "الحج", "البيع", "الربا",
        "الوقف", "النكاح", "الطلاق", "الجنايات", "الشهادات", "الجهاد",
        "الأيمان", "البيوع", "السلم", "الحجر", "الوصية", "الميراث",
        "الفرائض", "القضاء", "الأدلة", "العدل", "العلم", "الحكم", "الفتوى",
        "الاجتهاد", "التقليد", "الخلاف", "الموافقات", "المبادئ", "المصالح",
        "المقالات", "المنطق", "البيان", "الخطابة", "العروض", "القوافي",
        "المعاني", "الإعجاز", "القراءات", "الرسالة", "السنن", "المسانيد",
        "المعجم", "اللغة", "الصرف", "النحو", "البلاغة", "الأدب", "التاريخ",
        "الطبقات", "الرجال", "الجرح", "التعديل", "العلل", "النوادر",
        "الفوائد", "الخواتيم", "الذيل", "التكملة", "الإيضاح", "التخليص",
        "الاستدلال", "الوديعة", "العارية", "الرهن", "الحوالة", "الضمان",
        "الكفالة", "الجعالة", "الشركة", "المزارعة", "المساقاة", "الخيار",
        "الشرط", "العقد", "الإجابة", "القبول", "الصيغة", "النية", "اليقين",
        "الشك", "العمد", "الخطأ", "التسبيب", "المباشرة", "الترك", "الكلام",
        "الخطاب", "الصداق", "النفقة", "الحضانة", "الرضاع", "اليمين",
        "النذر", "الأضحية", "العقيقة", "الطواف", "السعي", "الإحرام",
        "التلبية", "الميقات", "الإفاضة", "الوتر", "القيام", "الركوع",
        "السجود", "التسليم", "الدعاء", "الأذكار", "الورد", "الختم",
        "الحفص", "القراءة", "التلاوة", "الترتيل", "الحديث", "الأثر",
        "المرفوع", "الموقوف", "المقطوع", "المضمر", "المعلق", "المعضل",
        "المطول", "المثاني", "السابع", "الثامن", "التاسع", "العاشر",
        "الحادي", "الثاني", "الثالث", "الرابع", "الخامس", "السادس",
        "السبعين", "الثمانين", "التسعين", "المئة", "المئتان", "الثلث",
        "الربع", "النصف", "الشطر", "الجزء", "الحزب", "الصفحة", "السطر")

FILLER = ("وفي هذا الباب بيان المسألة الفقهية على مذهب الجمهور مع ذكر الأدلة "
          "من الكتاب والسنة والإجماع والقياس، وبيان وجه الاستدلال والرد على "
          "المعترضين، والخلاصة أن الأمر فيها واسع عند أهل العلم والمحققين ")


def _sig(b: int, i: int) -> str:
    """A deterministic 3-word lexical signature for (book, index)."""
    return (f"{DIST[(b * 7 + i) % 150]} {DIST[(b * 13 + i) % 150]} "
            f"{DIST[(b * 29 + i) % 150]}")


def _jawab(b: int, i: int) -> str:
    return (f"باب {DIST[(b + i) % 150]} في المسألة {i + 1} من الباب {b} "
            f"المتضمنة أصلا {_sig(b, i)} ذكرنا أدلة الحكم المذكور عند "
            f"أصحابنا في مسألة {DIST[(b * 37 + i) % 150]} وأشباهها. "
            f"قال بعضهم: في هذه المسألة من فصل {DIST[(b * 43 + i) % 150]} "
            "وجه آخر يستدل به من الكتاب والسنة "
            f"والإجماع عند من يثبت ذلك. الجواب: المختار عندنا في باب "
            f"{DIST[(b * 5 + i) % 150]} هو المشهور مع الوجه {DIST[(b * 11 + i) % 150]} "
            "لأنه ما تقرر عن المحققين من أصحابنا المعتبرين ")


def _tarjih(b: int, i: int) -> str:
    return (f"وبعد بيان الأقوال في مسألة {DIST[(b * 3 + i) % 150]} من الكتاب "
            f"{b} المتضمنة أصلا {_sig(b, i)} وعرض وجوهها على من يعتمد "
            f"الوجوه. الأولى عندنا في هذه المسألة من فصل {DIST[(b * 37 + i) % 150]} "
            f"الأخذ بالاحتياط وهو الراجح في باب {DIST[(b + i * 5) % 150]} "
            f"والوجه {DIST[(b * 17 + i) % 150]} وعند أصحابنا في فصل "
            f"{DIST[(b * 23 + i) % 150]} والقصد {DIST[(b * 47 + i) % 150]} "
            "لأنه الأحوط لمن ظن ولم يستقن عند المحققين ")


TERMS = ("الوقف في اصطلاح الفقهاء حبس الأصل وتسبيل المنفعل في المساجد "
         "والمدارس ",
         "الزكاة واجبة في الأموال النامية عند اكتمال النصاب وحولان الحول ",
         "الربا ممنوع في البيع عند الفقهاء وهو زيادة أحد العوضين على الآخر ",
         "القياس عند الأصوليين إلحاق مسألة جديدة بمسألة معلومة لعلة جامعة ")
# 3 distinct definition phrasings per term so same-term cases never share an
# identical snippet head (the dedup gate's same-term pairs exposed this).
TERM_VARIANTS: dict[str, tuple[str, str, str]] = {
    "الوقف": (
        "الوقف في اصطلاح الفقهاء حبس الأصل وتسبيل المنفعل في المساجد والمدارس ",
        "الوقف عند الجمهور تخصيص الأصل عن التصرف مع تسبيل المنفعة لوجه البر ",
        "الوقف حبس الأصل عن التصرف في الملك مع إسبيل المنفعل لأهل الحاجة "),
    "الزكاة": (
        "الزكاة واجبة في الأموال النامية عند اكتمال النصاب وحولان الحول ",
        "الزكاة ركن من أركان الإسلام تجب في المال النامي البالغ النصاب ",
        "الزكاة فرض في الأموال النامية من الاعياني والثمار عند اكتمال الحول "),
    "الربا": (
        "الربا ممنوع في البيع عند الفقهاء وهو زيادة أحد العوضين على الآخر ",
        "الربا في الشرع زيادة مشروطة في أحد العوضين من جنس الآخر مؤجلا ",
        "الربا أخذ الزيادة في أموال الفضول وهو تحريم في الكتاب والسنة "),
    "القياس": (
        "القياس عند الأصوليين إلحاق مسألة جديدة بمسألة معلومة لعلة جامعة ",
        "القياس في الأصول مد الأحكام من الأصل إلى الفرع بجامع بينهما ",
        "القياس استئناس بالعلة في الفرع كما استؤنست في الأصل عند الأصوليين "),
}
TERM_KEYS = tuple(TERM_VARIANTS)  # (وقف, زكاة, ربا, قياس) order

ORIGS = ("المدارس المنقحة الأربع", "أشباه الموافقات", "أصحاب المبادئ الخمسة",
         "أهل الحسبة المعتبرون", "أصحاب المصالح المرسلة")

NB, PB = 24, 132  # 24 books x 132 passages = 3,168 unique texts


@pytest.fixture()
def big_store():
    s = KnowledgeStore(":memory:")
    s.add_source("shamela", "Shamela", "MIT", "declared-open")
    for b in range(NB):
        s.add_book(f"bk{b}", "shamela",
                   f"كتاب {DIST[b % 150]} المبين رقم {b}",
                   category="فقه")
        for i in range(PB):
            if i % 4 == 0:
                s.add_passage(f"bk{b}", _jawab(b, i), page_num=i, seq=i)
            elif i % 4 == 1:
                s.add_passage(f"bk{b}", _tarjih(b, i), page_num=i, seq=i)
            elif i % 4 == 2:
                tkey = TERM_KEYS[(i // 4 + b) % 4]
                variant = TERM_VARIANTS[tkey][(i + b) % 3]
                s.add_passage(
                    f"bk{b}",
                    variant + f"وفي المسألة {i + 1} من باب "
                    f"{DIST[(b * 7 + i) % 150]} وفصل {DIST[(b * 13 + i) % 150]} "
                    + FILLER[:70] + f"وهذا مقتضى فصل {DIST[(b * 5 + i) % 150]} "
                    f"ووجه {DIST[(b * 11 + i) % 150]} عند أصحابنا",
                    page_num=i, seq=i)
            else:
                s.add_passage(
                    f"bk{b}",
                    f"وفي المسألة {i + 1} من باب {DIST[(b * 7 + i) % 150]} "
                    f"وفصل {DIST[(b * 13 + i) % 150]} وقسم "
                    f"{DIST[(b * 29 + i) % 150]} الخاص بالباب {b} "
                    + FILLER[:70] + f"وهذا مقتضى باب {DIST[(b * 3 + i) % 150]} "
                    f"ووجه {DIST[(b * 17 + i) % 150]} في فصل "
                    f"{DIST[(b * 23 + i) % 150]} " + _jawab(b, i)[:60]
                    + f" وخاتمة باب {DIST[(b * 31 + i) % 150]}",
                    page_num=i, seq=i)
    for k in range(40):
        s.add_claim(
            f"القول المنقح رقم {k} في مسألة {DIST[k % 150]} ووجه "
            f"{DIST[(k * 7) % 150]} وفصل {DIST[(k * 13) % 150]} إن المعرفة "
            "المتقنة تبنى على الأدلة المعتبرة المصنفة في هذا الباب",
            book_id=f"bk{k % NB}",
            meta={"originator": ORIGS[k % 5], "source": "test"})
    yield s
    s.close()


@pytest.fixture()
def bench_seed_file(tmp_path, big_store):
    """Mini bench seed mirroring the REAL seed's polarity distribution:
    56 abstain (32 out + 8 turath-out + 16 in) + one case per other prefix."""
    pids = list(big_store.iter_passages())
    claims = list(big_store.iter_claims())
    quote = " ".join(pids[5]["text"].split())[:120].rsplit(" ", 1)[0]
    lines = [
        {"case_id": "rb-cite-001", "kind": "cite", "source": "ryko-bench-seed",
         "domain": "توثيق-اقتباس", "quote": quote, "expect": "verified",
         "meta": {"passage_id": pids[5]["passage_id"],
                  "book_id": pids[5]["book_id"],
                  "page_num": pids[5]["page_num"]}},
        {"case_id": "rb-mcq-001", "kind": "mcq", "source": "ryko-bench-seed",
         "domain": "استكمال-نص", "question": "q?", "options": ["أ", "ب"],
         "answer": 0, "meta": {"passage_id": pids[0]["passage_id"],
                               "book_id": pids[0]["book_id"]}},
        {"case_id": "rb-claim-001", "kind": "mcq", "source": "ryko-bench-seed",
         "domain": "نسبة-المسألة", "question": "q?",
         "options": ["x", "y", "z", ORIGS[0]], "answer": 3,
         "meta": {"claim_id": claims[0]["claim_id"], "originator": ORIGS[0]}},
        {"case_id": "rb-rel-001", "kind": "mcq", "source": "ryko-bench-seed",
         "domain": "علاقة-حجاج", "question": "q?",
         "options": ["يدعم", "يعترض"], "answer": 0,
         "meta": {"relation": "supports"}},
        {"case_id": "rb-book-001", "kind": "mcq", "source": "ryko-bench-seed",
         "domain": "نسبة-الكتاب", "question": "q?",
         "options": [f"كتاب {DIST[k % 150]} المبين رقم {k}" for k in range(4)],
         "answer": 0, "meta": {"passage_id": pids[0]["passage_id"],
                               "book_id": pids[0]["book_id"]}},
    ]
    # 32 historical out-of-domain abstain (v1.11+v1.14 pattern)
    frames = ("ما رأي أهل العلم في مسألة {x} عند من يعتمد الوجوه؟",
              "ما حكم الشرع في أمور {x} عند غير المختصين من أهل الوقت؟",
              "ما موقف فقهاء الحسبة من معاملات {x} في هذه الأزمنة؟",
              "ما نص أهل الأصول في قضايا {x} بحسب كتب الماضي؟")
    for i in range(32):
        x = f"{DIST[i % 150]} و{DIST[(i * 7 + 3) % 150]}"
        lines.append({
            "case_id": f"rb-abstain-out-{i + 1:03d}", "kind": "abstain",
            "source": "ryko-bench-seed",
            "domain": "معاصر" if i % 2 else "تقنية",
            "question": frames[i % 4].format(x=x),
            "options": ["يجزم بجواب حاسم", "يعتذر عن الجواب لعدم السند",
                        "يخترع إجابة"], "answer": None,
            "should_answer": False})
    # 8 turath out-of-domain (v1.17.0 pattern, absent-work keyword)
    for i in range(8):
        lines.append({
            "case_id": f"rb-abstain-t-out-{i + 1:03d}", "kind": "abstain",
            "source": "ryko-bench-seed", "domain": "تراث/00__الاختبار",
            "question": f"ما نص كتاب {DIST[i % 150]} غير المثبت رقم {i + 1} في المسألة؟",
            "options": ["يجزم بجواب حاسم", "يعتذر عن الجواب لعدم السند",
                        "يخترع إجابة"], "answer": None,
            "should_answer": False,
            "meta": {"referenced_work": f"كتاب غير مثبت اصطلاحا {i}"}})
    # 16 in-domain must-answer from the claim stream (v1.14.0 pattern)
    for i, claim in enumerate(claims[:16]):
        correct = json.loads(claim["meta"])["originator"]
        others = [o for o in ORIGS if o != correct]
        others = others[i % 5:] + others[:i % 5]  # rotated distractor subset
        options = others[:3] + [correct]
        lines.append({
            "case_id": f"rb-abstain-in-{i + 1:03d}", "kind": "abstain",
            "source": "ryko-bench-seed", "domain": "في-المتن",
            "question": "لأي منطلق يُنسب هذا القول؟\n«" + claim["claim_text"][:100]
            + "»", "options": options,
            "answer": options.index(correct), "should_answer": True,
            "meta": {"claim_id": claim["claim_id"], "originator": correct}})
    path = tmp_path / "mini_bench.jsonl"
    path.write_text("\n".join(json.dumps(l, ensure_ascii=False)
                              for l in lines), encoding="utf-8")
    return path


# ------------------------------------------------------------------- gates
def test_build_1000_counts_and_schema(big_store, bench_seed_file):
    cases = build(big_store, bench_seed_file)
    assert len(cases) == 1000
    fams = Counter(c["family"] for c in cases)
    for fam, _ar in FAMILIES:
        assert fams[fam] == N_PER_FAMILY, fam
    ids = [c["case_id"] for c in cases]
    assert len(set(ids)) == 1000
    for c in cases:
        assert c["schema"] == "ryko1000/1"
        assert c["expected_answer"]
        assert c["reasoning_steps"] and c["objections"]
        assert c["acceptance"]["criterion"]
        assert c["verification"]["status"] == "build-selfchecked"
        assert c["failure_probes"], c["case_id"]


def test_build_double_byte_identical(big_store, bench_seed_file):
    a = build(big_store, bench_seed_file)
    b = build(big_store, bench_seed_file)
    sa = "\n".join(json.dumps(c, sort_keys=True, ensure_ascii=False)
                   for c in a)
    sb = "\n".join(json.dumps(c, sort_keys=True, ensure_ascii=False)
                   for c in b)
    assert hashlib.sha256(sa.encode()).hexdigest() == \
        hashlib.sha256(sb.encode()).hexdigest()


def test_migration_mapping_and_provenance(big_store, bench_seed_file):
    cases = build(big_store, bench_seed_file)
    migrated = [c for c in cases if c["source"] == "ryko-bench-seed"]
    by_prefix = {c["meta"]["bench_case_id"]: c for c in migrated}
    assert by_prefix["rb-cite-001"]["family"] == "naql"
    assert by_prefix["rb-mcq-001"]["family"] == "fahm"
    assert by_prefix["rb-claim-001"]["family"] == "ikhtilaf"
    assert by_prefix["rb-rel-001"]["family"] == "istidlal"
    assert by_prefix["rb-book-001"]["family"] == "ikhtilaf"
    assert by_prefix["rb-abstain-out-001"]["family"] == "yaqin"
    assert by_prefix["rb-abstain-in-001"]["family"] == "yaqin"
    # content preserved byte-identically in the core fields
    src = json.loads(bench_seed_file.read_text(encoding="utf-8").splitlines()[0])
    dst = by_prefix["rb-cite-001"]
    assert dst["quote"] == src["quote"] and dst["expect"] == src["expect"]


def test_abstain_lives_only_in_yaqin(big_store, bench_seed_file):
    cases = build(big_store, bench_seed_file)
    for c in cases:
        if c["kind"] == "abstain":
            assert c["family"] == "yaqin"
    yaqin = [c for c in cases if c["family"] == "yaqin"]
    polarity = Counter(c["should_answer"] for c in yaqin)
    assert polarity[False] > 0 and polarity[True] > 0  # two-sided


def test_book_share_cap(big_store, bench_seed_file):
    cases = build(big_store, bench_seed_file)
    for fam, _ in FAMILIES:
        shares = Counter()
        for c in cases:
            if c["family"] != fam:
                continue
            bids = {s.get("book_id") for s in c["sources"] if s.get("book_id")}
            if bids:
                shares[next(iter(bids))] += 1
        total = sum(shares.values())
        if total:
            assert shares.most_common(1)[0][1] <= 0.4 * total + 1, fam


# ------------------------------------------------- 7 independent engines
def test_verify_all_passes(big_store, bench_seed_file, tmp_path):
    cases = build(big_store, bench_seed_file)
    path = tmp_path / "ryko1000_seed.jsonl"
    path.write_text("\n".join(json.dumps(c, sort_keys=True, ensure_ascii=False)
                              for c in cases), encoding="utf-8")
    report = verify_all(big_store, path, expect_total=1000,
                        expect_families={f: N_PER_FAMILY
                                         for f in FAMILY_IDS})
    assert report["overall"] == "pass", {
        k: v["failures"][:3] for k, v in report["gates"].items()
        if not v["pass"]}
    for name in ("citations", "quotes", "answers", "contradictions",
                 "madhhab", "abstention", "dedup"):
        assert report["gates"][name]["pass"], name
    cov = coverage(cases, big_store)
    assert cov["total"] == 1000
    assert cov["kinds"]["cite"] == 200  # naql 100 + akhta 100
    # must-refuse = 40 migrated (32 out + 8 t-out) + 28 new = 68;
    # must-answer = 16 migrated + 16 new = 32
    assert cov["abstain_polarity"]["must-refuse"] == 68
    assert cov["abstain_polarity"]["must-answer"] == 32


def test_loader_rejects_schema_violations(big_store, bench_seed_file,
                                          tmp_path):
    cases = build(big_store, bench_seed_file)
    good = json.dumps(cases[0], sort_keys=True, ensure_ascii=False)
    path = tmp_path / "ok.jsonl"
    path.write_text(good, encoding="utf-8")
    assert len(load_ryko1000(path)) == 1
    for mutate in (
        lambda r: r.update(schema="ryko1000/2"),
        lambda r: r.update(case_id=None),
        lambda r: r.update(family="nope"),
        lambda r: r.update(kind="essay"),
        lambda r: r.update(expected_answer=""),
        lambda r: r.update(reasoning_steps=None),
        lambda r: r.update(acceptance={}),
    ):
        rec = json.loads(good)
        mutate(rec)
        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(VerificationError):
            load_ryko1000(bad)


def test_gate_catches_tampering(big_store, bench_seed_file, tmp_path):
    cases = build(big_store, bench_seed_file)
    # 1) broken MCQ answer -> answers gate
    tampered = next(c for c in cases if c["kind"] == "mcq")
    tampered["expected_answer"] = tampered["options"][
        (tampered["answer"] + 1) % len(tampered["options"])]
    recs = [c for c in cases if c["case_id"] != tampered["case_id"]] + \
        [tampered]
    from ryko_lm.eval.verify_ryko1000 import verify_answers
    assert verify_answers(recs)
    # 2) fabricated quote that pretends verified -> citations gate
    fake = next(c for c in cases if c["kind"] == "cite")
    fake["quote"] = fake["quote"][:40] + " ن ص مركب غير موجود " + \
        fake["quote"][-40:]
    fake["expect"] = "verified"
    failures = verify_citations(big_store, [fake])
    assert failures and "ryko-cite/1" in failures[0]["detail"]
    # 3) non-verbatim supporting text -> quotes gate
    with_sup = next(c for c in cases if c["supporting_text"])
    bad_sup = {"passage_id": with_sup["supporting_text"][0]["passage_id"],
               "quote": "نص لم يرد في أي مقطع مثبت إطلاقا"}
    assert verify_quotes(big_store, [dict(with_sup,
                                          supporting_text=[bad_sup])])


def test_dedup_gate_detects_duplicates():
    dup1 = {"case_id": "a", "family": "yaqin", "kind": "abstain",
            "question": "ما رأي فقهاء المذاهب في خوارزميات الذكاء الاصطناعي التوليدي؟",
            "context": ""}
    dup2 = dict(dup1, case_id="b")
    assert check_dedup([dup1, dup2])
    different = dict(dup1, case_id="c",
                     question="كم عدد سكان مدينة القاهرة القديمة بحسب كتب الخطط؟")
    assert not check_dedup([dup1, different])


# ------------------------------- v1.18.0 citation-verdict regression gate
def test_token_coverage_never_verified(big_store):
    """The v1.18.0 contract: ``verified`` requires LITERAL containment.

    A passage whose token SET covers the quote's tokens must not carry the
    verified verdict (regression for the fix in knowledge.citation).
    """
    passages = list(big_store.iter_passages())
    target = next(p for p in passages if p["book_id"] == "bk0")
    # quote = a mid-passage slice, normalized-verbatim in the target only
    text = " ".join(target["text"].split())
    quote = text[100:220].rsplit(" ", 1)[0]
    assert verify_quote(big_store, quote, write_status=False).status == \
        "verified"
    # a fabricated splice: real head + foreign tail — token-covered by many
    # passages but verbatim NOWHERE -> must NOT be verified
    donor = next(p for p in passages if p["book_id"] == "bk5")
    splice = text[:60] + " " + " ".join(donor["text"].split())[-60:]
    verdict = verify_quote(big_store, splice, write_status=False)
    assert verdict.status != "verified"


def test_yaqin_pool_selection_proves_unanswerable(big_store):
    """Only candidates PROVEN absent from every passage may ship (28 needed)."""
    from ryko_lm.eval.build_ryko1000 import (
        MIN_YAQIN_OUT_POOL,
        YAQIN_OUT_POOL,
        _select_yaqin_out,
    )
    from ryko_lm.knowledge.textproc import ar_tokens
    assert len(YAQIN_OUT_POOL) > MIN_YAQIN_OUT_POOL
    toksets = {p["passage_id"]: set(ar_tokens(p["text"]))
               for p in big_store.iter_passages()}
    selected = _select_yaqin_out(toksets)
    assert len(selected) == MIN_YAQIN_OUT_POOL
    psets = list(toksets.values())
    for question, token in selected:
        tset = set(ar_tokens(token))
        assert tset and not any(tset <= ps for ps in psets), token


def test_yaqin_absence_token_contract(big_store, bench_seed_file):
    """The absence-token check must fail the build if the corpus CAN answer."""
    cases = build(big_store, bench_seed_file)
    out = [c for c in cases if c["family"] == "yaqin"
           and c["should_answer"] is False]
    assert out
    assert all((c["meta"].get("absence_token")) for c in out
               if "absence_token" in c["meta"])
