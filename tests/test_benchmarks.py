"""Benchmark gates tests — loader, runner determinism, cite integration."""

from __future__ import annotations

import json

import pytest

from ryko_lm.eval.benchmarks import (
    BenchmarkError,
    load_benchmark,
    run_abstain,
    run_cite,
    run_mcq,
    summarize,
    write_report,
)
from ryko_lm.eval.build_bench import build as build_bench_cases
from ryko_lm.knowledge.store import KnowledgeStore

from conftest import TOKENIZER_JSON, requires_tokenizer


@pytest.fixture()
def store():
    s = KnowledgeStore(":memory:")
    s.add_source("shamela", "Shamela", "MIT", "declared-open")
    filler = ("وفي هذا الباب بيان المسألة الفقهية على مذهب الجمهور مع ذكر الأدلة "
              "من الكتاب والسنة والإجماع والقياس، وبيان وجه الاستدلال والرد على "
              "المعترضين، والخلاصة أن الأمر فيها واسع عند أهل العلم والمحققين. ")
    for b in range(4):
        book = f"bk{b}"
        s.add_book(book, "shamela", f"كتاب رقم {b}", category="فقه")
        for i in range(3):
            s.add_passage(book, filler * (2 + (i + b) % 2), page_num=10 + i, seq=0)
    yield s
    s.close()


def build_bench_cases_from(lines):
    """Decode serialized seed lines back to case dicts (test helper)."""
    return [json.loads(l) if isinstance(l, str) else l for l in lines]


def _write_cases(tmp_path, rows: list[dict]) -> object:
    p = tmp_path / "cases.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                 encoding="utf-8")
    return p


def _mcq_row(i: int = 1) -> dict:
    return {"case_id": f"m{i}", "kind": "mcq", "source": "test", "domain": "د",
            "question": "سؤال؟", "options": ["أ", "ب"], "answer": 0}


def test_loader_roundtrip(tmp_path):
    p = _write_cases(tmp_path, [_mcq_row()])
    cases = load_benchmark(p)
    assert len(cases) == 1 and cases[0].kind == "mcq" and cases[0].answer == 0


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(kind="essay"),
    lambda r: r.update(case_id=None),
    lambda r: r.update(answer=9),
    lambda r: r.update(source=None),
])
def test_loader_rejects_bad_schema(tmp_path, mutate):
    row = _mcq_row()
    mutate(row)
    p = _write_cases(tmp_path, [row])
    with pytest.raises(BenchmarkError):
        load_benchmark(p)


def test_loader_rejects_bad_abstain_and_cite(tmp_path):
    bad_abstain = {"case_id": "a1", "kind": "abstain", "source": "s", "domain": "d",
                   "question": "؟", "options": ["أ", "ب"]}
    bad_cite = {"case_id": "c1", "kind": "cite", "source": "s", "domain": "d",
                "expect": "verified"}
    p = _write_cases(tmp_path, [bad_abstain, bad_cite])
    with pytest.raises(BenchmarkError):
        load_benchmark(p)


def test_loader_empty_rejected(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(BenchmarkError):
        load_benchmark(p)


@requires_tokenizer
def test_mcq_real_model(store, tmp_path):
    import torch

    from ryko_lm.model.ryko_model import RykoConfig, RykoLM
    from ryko_lm.tokenizer.loader import RykoTokenizer

    torch.manual_seed(0)
    tok = RykoTokenizer.from_file(TOKENIZER_JSON)
    cfg = RykoConfig(vocab_size=tok.vocab_size, block_size=96, n_layer=1, n_head=2,
                     n_kv_head=1, n_embd=32, ffn_hidden=64)
    model = RykoLM(cfg).eval()
    case_rows = [{
        "case_id": "m1", "kind": "mcq", "source": "t", "domain": "استكمال-نص",
        "question": "أي المتابعات تكمل النص؟",
        "options": ["وفي هذا الباب بيان", "كلمة أخرى مختلفة تماماً"], "answer": 0,
    }]
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps(case_rows[0], ensure_ascii=False), encoding="utf-8")
    cases = load_benchmark(p)
    rows1 = run_mcq(model, tok, cases)
    rows2 = run_mcq(model, tok, cases)
    assert rows1 == rows2  # deterministic
    assert rows1[0]["predicted"] in {0, 1}
    assert isinstance(rows1[0]["confidence"], float)


@requires_tokenizer
def test_abstain_threshold_logic(store, tmp_path):
    import torch

    from ryko_lm.model.ryko_model import RykoConfig, RykoLM
    from ryko_lm.tokenizer.loader import RykoTokenizer

    torch.manual_seed(0)
    tok = RykoTokenizer.from_file(TOKENIZER_JSON)
    cfg = RykoConfig(vocab_size=tok.vocab_size, block_size=96, n_layer=1, n_head=2,
                     n_kv_head=1, n_embd=32, ffn_hidden=64)
    model = RykoLM(cfg).eval()
    rows = [{
        "case_id": "a1", "kind": "abstain", "source": "t", "domain": "معاصر",
        "question": "سؤال معاصر بلا سند في المتون؟",
        "options": ["إجابة أولى", "إجابة ثانية"], "should_answer": False,
    }]
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps(rows[0], ensure_ascii=False), encoding="utf-8")
    cases = load_benchmark(p)
    out = run_abstain(model, tok, cases, threshold=0.99)
    assert out[0]["abstained"] is True  # confidence never reaches 0.99
    assert out[0]["correct_abstain"] is True
    assert out[0]["wrong_abstain"] is False


def test_cite_runner(store, tmp_path):
    passage = next(p for p in store.iter_passages())
    quote = " ".join(passage["text"].split())[:100].rsplit(" ", 1)[0]
    rows = [{
        "case_id": "c1", "kind": "cite", "source": "t", "domain": "توثيق",
        "quote": quote, "expect": "verified",
    }, {
        "case_id": "c2", "kind": "cite", "source": "t", "domain": "توثيق",
        "quote": "قطعة من كتاب آخر غير موجود في المتن إطلاقا بتات", "expect": "not_found",
    }]
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    cases = load_benchmark(p)
    out = run_cite(store, cases)
    assert out[0]["verdict"] == "verified" and out[0]["match"] is True
    assert out[1]["verdict"] == "not_found" and out[1]["match"] is True


def test_summarize_and_report(store, tmp_path):
    mcq_rows = [{"case_id": "m1", "domain": "د", "correct": True, "source": "s"}]
    abstain_rows = [{"case_id": "a1", "domain": "معاصر", "should_answer": False,
                     "abstained": True, "correct_abstain": True, "wrong_abstain": False}]
    cite_rows = [{"case_id": "c1", "verdict": "verified", "expected": "verified",
                  "match": True, "domain": "د", "source": "s", "score": 1.0}]
    report = summarize(mcq_rows, abstain_rows, cite_rows)
    assert report["mcq"]["accuracy"] == 1.0
    assert report["abstain"]["correct_abstain_rate"] == 1.0
    assert report["cite"]["match_rate"] == 1.0
    json_path, md_path = write_report(report, {"checkpoint": "x", "params_M": 0.1},
                                      tmp_path / "bench")
    assert json_path.exists() and md_path.exists()
    assert "accuracy" in json.loads(json_path.read_text(encoding="utf-8"))["results"]["mcq"]


def test_build_bench_from_fixture(store):
    cases = build_bench_cases(store)
    kinds = {c["kind"] for c in cases}
    assert kinds <= {"mcq", "cite", "abstain"}
    assert sum(1 for c in cases if c["kind"] == "cite") >= 2
    for c in cases:
        if c["kind"] == "cite":
            assert c["expect"] in {"verified", "not_found"}
        if c["kind"] == "abstain":
            assert isinstance(c["should_answer"], bool)
            # out-of-domain abstain cases keep the historical negative flag
            if c["case_id"].startswith("rb-abstain-out-"):
                assert c["should_answer"] is False
            if c["case_id"].startswith("rb-abstain-in-"):
                assert c["should_answer"] is True
    # determinism: same store, same seed -> same cases
    again = build_bench_cases(store)
    assert [json.dumps(c, sort_keys=True, ensure_ascii=False) for c in cases] == \
        [json.dumps(c, sort_keys=True, ensure_ascii=False) for c in again]


def _add_claims_and_relations(s: KnowledgeStore) -> None:
    """Curated-claim fixture: four originators + one documented relation."""
    for aid, name, yr in [("auth-g", "الغزالي", 505), ("auth-t", "ابن تيمية", 728),
                          ("auth-q", "ابن القيم", 751), ("auth-sh", "الشاطبي", 790)]:
        s.add_author(aid, name, yr)
    s.add_book("bkG", "shamela", "منقذ من الضلال", author_id="auth-g")
    s.add_book("bkT", "shamela", "درء تعارض العقل والنقل", author_id="auth-t")
    s.add_book("bkQ", "shamela", "مدارج السالكين", author_id="auth-q")
    s.add_book("bkS", "shamela", "الموافقات في أصول الشريعة", author_id="auth-sh")
    g = s.add_claim("العقل قد يخطئ فيما نطقت به النظرات، والوحي صادر من الله",
                    book_id="bkG",
                    meta={"layer": "fikr", "originator": "الغزالي"})
    t = s.add_claim("تعارض العقل الصحيح الصريح مع ما نبه عليه الوحي محال",
                    book_id="bkT",
                    meta={"layer": "fikr", "originator": "ابن تيمية"})
    s.add_claim("المقصود بالسير إلى الله إصلاح القلب، وبالعمل تكميل المراتب",
                book_id="bkQ",
                meta={"layer": "fikr", "originator": "ابن القيم"})
    s.add_claim("المقاصد الشرعية ثابتة بالدليل، والمفاد حفظ المصالح الجالية",
                book_id="bkS",
                meta={"layer": "fikr", "originator": "الشاطبي"})
    s.add_relation(g, t, "supports", evidence="نص في المقدمة")


def test_build_bench_claim_attribution_family(store):
    _add_claims_and_relations(store)
    cases = build_bench_cases(store)
    claim_cases = [c for c in cases if c["case_id"].startswith("rb-claim-")]
    assert claim_cases, "claim-attribution family must produce cases"
    for c in claim_cases:
        assert c["domain"] == "نسبة-المسألة"
        correct = c["options"][c["answer"]]
        assert correct == c["meta"]["originator"]
        assert sum(1 for o in c["options"] if o == correct) == 1
    # determinism across builds
    again = [c for c in build_bench_cases(store)
             if c["case_id"].startswith("rb-claim-")]
    assert claim_cases == again


def test_build_bench_relation_family(store):
    _add_claims_and_relations(store)
    cases = build_bench_cases(store)
    rel_cases = [c for c in cases if c["case_id"].startswith("rb-rel-")]
    assert len(rel_cases) == 1
    c = rel_cases[0]
    assert c["domain"] == "علاقة-حجاج"
    from ryko_lm.eval.build_bench import RELATION_LABELS

    labels = [lbl for _, lbl in RELATION_LABELS]
    assert c["options"] == labels
    assert len(labels) == 8  # full closed vocabulary
    assert c["options"][c["answer"]] == "يدعم"
    assert c["meta"]["relation"] == "supports"


def test_build_bench_book_family(store):
    cases = build_bench_cases(store)
    book_cases = [c for c in cases if c["case_id"].startswith("rb-book-")]
    assert book_cases, "book-attribution family must produce cases from fixture"
    titles = {c["options"][c["answer"]] for c in book_cases}
    assert len(titles) == len(book_cases)  # one case per distinct book title
    for c in book_cases:
        assert c["domain"] == "نسبة-الكتاب"
        assert len(set(c["options"])) == 4  # correct + 3 distinct distractors
        again = [x for x in build_bench_cases(store)
                 if x["case_id"] == c["case_id"]]
        assert again == [c]  # determinism per case


def test_build_bench_abstain_contract():
    from ryko_lm.eval.build_bench import (ABSTAIN_CASES, N_ABSTAIN_IN,
                                          N_ABSTAIN_OUT, DOMAIN_ABSTAIN_OUT,
                                          DOMAIN_ABSTAIN_IN)

    # two-sided abstain set (v1.14.0): 32 out-of-domain + 16 in-domain positives
    assert len(ABSTAIN_CASES) == N_ABSTAIN_OUT == 32
    assert N_ABSTAIN_IN == 16
    assert all(domain in DOMAIN_ABSTAIN_OUT for _, domain in ABSTAIN_CASES)
    assert DOMAIN_ABSTAIN_IN == {"في-المتن"}
    # the first 16 cases are the historical set, kept verbatim
    assert ABSTAIN_CASES[0][0].startswith("ما رأي علماء التراث في سرعة المعالجات")
    assert ABSTAIN_CASES[15][0].startswith("كم عدد لافتات الطرق السريعة")


def test_summarize_per_domain_cite_and_abstain(tmp_path):
    mcq_rows = [{"case_id": "m1", "domain": "د", "correct": True, "source": "s"}]
    abstain_rows = [
        {"case_id": "a1", "domain": "معاصر", "should_answer": False,
         "abstained": True, "correct_abstain": True, "wrong_abstain": False},
        {"case_id": "a2", "domain": "تقنية", "should_answer": True,
         "abstained": True, "correct_abstain": False, "wrong_abstain": True},
    ]
    cite_rows = [
        {"case_id": "c1", "domain": "توثيق-اقتباس", "verdict": "verified",
         "expected": "verified", "match": True, "score": 1.0},
        {"case_id": "c2", "domain": "توثيق-اقتباس", "verdict": "not_found",
         "expected": "verified", "match": False, "score": 0.1},
    ]
    report = summarize(mcq_rows, abstain_rows, cite_rows)
    assert report["cite"]["per_domain"]["توثيق-اقتباس"] == {"n": 2, "rate": 0.5}
    assert report["abstain"]["per_domain"]["معاصر"] == {
        "n": 1, "correct_abstain": 1, "wrong_abstain": 0}
    _, md_path = write_report(report, {"checkpoint": "x", "params_M": 0.1},
                              tmp_path / "b")
    md = md_path.read_text(encoding="utf-8")
    assert "مجال الامتناع" in md and "مجال الاقتباس" in md


def test_build_bench_abstain_in_family():
    """In-domain abstain positives: claims beyond the MCQ batch (should_answer=True)."""
    from ryko_lm.eval.build_bench import N_ABSTAIN_IN, N_CLAIM_MCQ, _claim_families

    s = KnowledgeStore(":memory:")
    s.add_source("shamela", "Shamela", "MIT", "declared-open")
    s.add_author("auth-x", "منسِب الاختبار", 700)
    s.add_book("bkX", "shamela", "كتاب الاختبار الطويل في الأصول", author_id="auth-x")
    origins = ["مدرسة أ", "مدرسة ب", "مدرسة ج", "مدرسة د"]
    text = ("هذا قول منسوب طويل بما يكفي للتجاوز الحد الأدنى للطول "
            "في بانية البذرة بوضوح تام دون شك أو خلاف.")
    total = N_CLAIM_MCQ + N_ABSTAIN_IN + 2
    for i in range(total):
        s.add_claim(f"{text} التكرار رقم {i:03d}", book_id="bkX",
                    meta={"layer": "fikr", "originator": origins[i % 4]})
    import random as _random
    mcq, abstain_in = _claim_families(s, _random.Random(1))
    s.close()
    assert len(mcq) == N_CLAIM_MCQ
    assert len(abstain_in) == N_ABSTAIN_IN
    mcq_ids = {c["meta"]["claim_id"] for c in mcq}
    in_ids = {c["meta"]["claim_id"] for c in abstain_in}
    assert not (mcq_ids & in_ids)  # disjoint families from one shuffled stream
    for c in abstain_in:
        assert c["kind"] == "abstain" and c["should_answer"] is True
        assert c["domain"] == "في-المتن"
        assert c["options"][3] == c["meta"]["originator"] or \
            c["meta"]["originator"] in c["options"]
        assert c.get("answer") is None


def test_sweep_abstain_thresholds_deterministic():
    """Operating point balances correct abstain against over-refusal."""
    from ryko_lm.eval.benchmarks import sweep_abstain_thresholds

    # out-of-domain low-confidence (should abstain) + in-domain high-confidence
    confidences = [0.2, 0.3, 0.4, 0.6, 0.8, 0.9]
    flags = [False, False, False, True, True, True]
    cal = sweep_abstain_thresholds(confidences, flags)
    op = cal["operating_point"]
    assert op is not None and cal["n_out"] == 3 and cal["n_in"] == 3
    # threshold 0.45 is the first grid point catching all 3 out-of-domain
    # while sparing all in-domain (strict-greater sweep keeps the lowest tie)
    assert op["threshold"] == 0.45
    assert op["correct_abstain_rate"] == 1.0
    assert op["wrong_abstain_rate"] == 0.0
    assert op["balanced_score"] == 2.0
    # determinism: identical inputs -> identical curve
    again = sweep_abstain_thresholds(confidences, flags)
    assert again == cal
    # one-sided degradation still works (legacy fixture shape)
    cal2 = sweep_abstain_thresholds([0.1, 0.9], [False, False])
    assert cal2["operating_point"]["threshold"] == 0.95


def test_iter_books_read_accessor(store):
    books = list(store.iter_books())
    assert len(books) == 4
    ids = [b["book_id"] for b in books]
    assert ids == sorted(ids)  # deterministic order
    assert all("title_ar" in b and "rights_status" in b for b in books)


def test_build_bench_turath_growth_stability(store):
    """v1.16.0/v1.17.0 contract: adding turath (wide-selection) books must
    append new families with independent ids and leave every legacy case
    byte-identical — the seed grows by append, never by renumbering.  The
    unclaimed book flows into the FIRST turath family (catch-all); the
    frozen v1.17.0 sets stay untouched."""
    legacy = [json.dumps(c, sort_keys=True, ensure_ascii=False)
              for c in build_bench_cases(store)]

    # add a turath book with verifiable passages
    store.add_source("turath", "التراث الشامل المُنتقى", "MIT", "declared-open")
    filler_t = ("وفي هذا الفصل من كتاب التراث الشامل تفصيل مسألة الشرط في "
                "العقد وأثره في الانعقاد واللزوم عند الفقهاء مع دليله من "
                "النصوص، وذكر اختلافهم في تفريعه والترجيح في الأمر المشكل. ")
    store.add_book("turath-9001", "turath", "كتاب التراث الشامل تجربة",
                   category="تراث/تجربة")
    for i in range(4):
        store.add_passage("turath-9001", filler_t * (2 + i), page_num=1 + i, seq=0)

    from ryko_lm.eval.build_bench import N_TURATH_ABSTAIN_OUT
    grown = build_bench_cases(store)
    t_cases = [c for c in grown if c["case_id"].startswith("rb-cite-t-")]
    assert t_cases, "turath cite family must produce cases from turath books"
    assert all(c["kind"] == "cite" for c in t_cases)
    assert any(c["expect"] == "verified" for c in t_cases)
    # positional byte-identity for the prefix BEFORE the fixed abstain tail:
    # on a store that gains its FIRST turath books, the cite-t family is
    # appended before the store-independent rb-abstain-t-out tail, so the
    # tail shifts position (its CONTENT is identical); on real growth the
    # tail never shifts because new families append after it.
    stable_len = len(legacy) - N_TURATH_ABSTAIN_OUT
    assert [json.dumps(c, sort_keys=True, ensure_ascii=False)
            for c in grown[:stable_len]] == legacy[:stable_len]
    # every legacy case keeps its id AND its exact content (no renumbering,
    # no mutation) — checked per case, position-independent for the tail
    legacy_by_id = {c["case_id"]: json.dumps(c, sort_keys=True, ensure_ascii=False)
                    for c in build_bench_cases_from(legacy)}
    for c in grown:
        if c["case_id"] in legacy_by_id:
            assert json.dumps(c, sort_keys=True, ensure_ascii=False) == legacy_by_id[c["case_id"]], \
                f"case mutated on growth: {c['case_id']}"
    # the new ids are exactly the appended families, in build order
    new_ids = [c["case_id"] for c in grown if c["case_id"] not in legacy_by_id]
    assert new_ids == [c["case_id"] for c in grown[len(legacy) - N_TURATH_ABSTAIN_OUT:]
                       if c["case_id"] not in legacy_by_id]
    # determinism on the grown store as well
    again = build_bench_cases(store)
    assert [json.dumps(c, sort_keys=True, ensure_ascii=False) for c in grown] == \
        [json.dumps(c, sort_keys=True, ensure_ascii=False) for c in again]


def test_build_bench_turath2_cite_and_book_families():
    """v1.17.0: rb-cite-t2-*/rb-book-t-* families over the frozen T2 set —
    self-checked, deterministic, and disjoint from family 1's quotes."""
    from ryko_lm.eval.build_bench import (
        N_TURATH2_CITE_VERIFIED, _turath2_cite_cases, _turath_book_cases)

    s = KnowledgeStore(":memory:")
    s.add_source("turath", "التراث الشامل المُنتقى", "MIT", "declared-open")
    # DISTINCT filler per book: cross-book splices must stay not_found, so
    # the negative half of the family has material to work with
    fillers = [
        "مسألة بيوع الفضايل بالذهب الصريح عند الفقهاء وذكر أدلتهم من الكتاب. ",
        "باب في أحكام اليمين والنذر وفرق الحلف بالنزول والقسم عند الأصوليين. ",
        "فصل في الطلاق الرجعي والعدة والرضاع وما يتعلق بأحكام الأسرة. ",
        "كلام على الإجارة والأجرة وأحكام المزارعة والمساقاة في الفقه المعاملات. ",
    ]
    t2_ids = ["turath-1018", "turath-1024", "turath-1219", "turath-1245"]
    for bid, title, fl in zip(t2_ids,
                              ["كتاب التجربة الأول", "كتاب التجربة الثاني",
                               "كتاب التجربة الثالث", "كتاب التجربة الرابع"],
                              fillers):
        s.add_book(bid, "turath", title, category="تراث/تجربة")
        for i in range(8):  # > the 12-positive quota overall → negatives material
            # a distinct per-passage suffix keeps texts unique (the store
            # dedups identical (book, text) rows)
            s.add_passage(bid, (fl + f" والنوع الأول رقم {{i}} فيه تفصيل "
                                f"الجواب وأثره في الانعقاد عند المحققين ") .format(i=i) * (3 + (i % 2)),
                          page_num=1 + i, seq=0)
    t2 = _turath2_cite_cases(s, used_passage_ids=set())
    assert t2, "cite-t2 family must produce cases from T2 books"
    assert all(c["case_id"].startswith("rb-cite-t2-") for c in t2)
    pos = [c for c in t2 if c["expect"] == "verified"]
    assert 1 <= len(pos) <= N_TURATH2_CITE_VERIFIED
    # book family: one case per distinct title (4 fixture books → 4 cases;
    # the committed 40-book store reaches the 12 quota — see the seed test)
    books = _turath_book_cases(s)
    assert len(books) == 4
    assert all(c["case_id"].startswith("rb-book-t-") for c in books)
    assert all(c["domain"] == "نسبة-الكتاب-التراثي" for c in books)
    assert len({c["meta"]["book_id"] for c in books}) == 4
    used = {c["meta"]["passage_id"] for c in pos}
    t2b = _turath2_cite_cases(s, used_passage_ids=used)
    assert {c["meta"]["passage_id"] for c in t2b if c["expect"] == "verified"}.isdisjoint(used)
    s.close()


def test_build_bench_turath_abstain_out_family(store):
    """v1.17.0: rb-abstain-t-out-* — 8 heritage-category out-of-domain cases
    with a build-time self-check that the referenced works stay uncommitted."""
    from ryko_lm.eval.build_bench import (N_TURATH_ABSTAIN_OUT,
                                          _turath_abstain_out_cases)

    cases = _turath_abstain_out_cases(store)
    assert len(cases) == N_TURATH_ABSTAIN_OUT == 8
    for c in cases:
        assert c["kind"] == "abstain" and c["should_answer"] is False
        assert c["case_id"].startswith("rb-abstain-t-out-")
        assert c["domain"].startswith("تراث/")
        assert c["meta"]["referenced_work"]
    # determinism
    again = _turath_abstain_out_cases(store)
    assert cases == again

    # a committed book whose title collides with a referenced work must
    # fail the build loudly (the case becomes answerable → must be replaced)
    from ryko_lm.eval.build_bench import TURATH_ABSTAIN_OUT_CASES
    colliding_title = TURATH_ABSTAIN_OUT_CASES[0][2]  # "رد المحتار"
    store.add_book("bk-bad", "shamela", colliding_title + " نسخة تجريبية")
    with pytest.raises(ValueError, match="committed works"):
        _turath_abstain_out_cases(store)


def test_build_bench_cli_is_sha256_neutral(tmp_path):
    """v1.17.0 regression guard: the seed-build CLI is a read-only consumer
    of the store — verify_quote's internal searches must never persist the
    derived passage_vec rows (the old default silently doubled the committed
    DB and invalidated its receipt)."""
    import hashlib

    from ryko_lm.eval.build_bench import main as build_main

    db = tmp_path / "bench_neutral.sqlite"
    s = KnowledgeStore(str(db))
    s.add_source("shamela", "Shamela", "MIT", "declared-open")
    filler = ("وفي هذا الباب بيان المسألة الفقهية على مذهب الجمهور مع ذكر "
              "الأدلة من الكتاب والسنة والإجماع والقياس، والخلاصة أن الأمر "
              "واسع عند أهل العلم. ")
    s.add_book("bkN", "shamela", "كتاب النيادة التجريبي", category="فقه")
    for i in range(6):
        s.add_passage("bkN", filler * (2 + i % 2), page_num=1 + i, seq=0)
    s.close()

    sha_before = hashlib.sha256(db.read_bytes()).hexdigest()
    out = tmp_path / "seed.jsonl"
    assert build_main(["--store", str(db), "--out", str(out)]) == 0
    sha_after = hashlib.sha256(db.read_bytes()).hexdigest()
    assert sha_before == sha_after, (
        "the seed build mutated the store (vectors persisted) — main() must "
        "open the store persist_vectors=False")

    import sqlite3
    conn = sqlite3.connect(db)
    n_vec = conn.execute("SELECT COUNT(*) FROM passage_vec").fetchone()[0]
    conn.close()
    assert n_vec == 0, "passage_vec must stay empty for read-only sessions"


RYKO_SEED_PREFIX_SHA256 = \
    "23469dbd817255b489f7ce8c62f0ecade6f0b1a96c11df6aece1b1497682f180"


def test_committed_seed_grows_by_append_only():
    """Release contract pinned at v1.17.0: the committed seed's first 277
    lines (everything up to and including the v1.16.0 rb-cite-t-* family)
    must stay byte-identical FOREVER — any future release appends new
    families with independent ids; it may never renumber or mutate a
    historical case, because measured results hang on these lines."""
    import hashlib
    from pathlib import Path

    seed = Path(__file__).resolve().parents[1] / \
        "docs" / "benchmarks" / "ryko_bench_seed.jsonl"
    if not seed.exists():
        pytest.skip("committed seed not present")
    lines = seed.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 315
    prefix = "\n".join(lines[:277]) + "\n"
    sha = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    assert sha == RYKO_SEED_PREFIX_SHA256, (
        "the committed seed's historical prefix changed — measured case "
        "results must stay pinned; append new families instead")
    import re
    ids = [json.loads(l)["case_id"] for l in lines]
    # independent per-family counters, grouped and ordered as documented
    legacy_cite = [i for i in ids if re.fullmatch(r"rb-cite-(neg-)?\d{3}", i)]
    assert len(legacy_cite) == 64  # 48 verified + 16 spliced negatives
    assert len([i for i in ids if re.fullmatch(r"rb-cite-t-(neg-)?\d{3}", i)]) == 18
    assert len([i for i in ids if re.fullmatch(r"rb-cite-t2-(neg-)?\d{3}", i)]) == 18
    assert len([i for i in ids if i.startswith("rb-book-t-")]) == 12
    assert len([i for i in ids if i.startswith("rb-abstain-t-out-")]) == 8
    # duplicate ids are loader-fatal — the counters above also prove uniqueness
    assert len(set(ids)) == len(ids) == 315
