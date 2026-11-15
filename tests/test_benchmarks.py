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
            assert c["should_answer"] is False
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


def test_build_bench_abstain_contract():
    from ryko_lm.eval.build_bench import ABSTAIN_CASES, N_ABSTAIN

    assert len(ABSTAIN_CASES) == N_ABSTAIN == 16
    assert all(domain in {"معاصر", "تقنية"} for _, domain in ABSTAIN_CASES)


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
