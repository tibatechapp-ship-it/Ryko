"""Turath layer (v1.16.0): wide stratified selection — ingest, policy, isolation."""

from __future__ import annotations

import json

import pytest

from ryko_lm.knowledge import ingest as ingest_mod
from ryko_lm.knowledge.store import KnowledgeStore
from ryko_lm.knowledge.turath import (
    build_manifest,
    record_size,
    select_turath_core,
)


@pytest.fixture()
def store(tmp_path):
    st = KnowledgeStore(str(tmp_path / "turath_test.sqlite"))
    yield st
    st.close()


def _rec(book_id: str, cat: str, pages: list[str], title: str | None = None) -> dict:
    return {
        "id": f"shamela-{book_id}",
        "book": title or f"كتاب {book_id}",
        "cat": cat,
        "pages": len(pages),
        "page_records": [{"page_num": i + 1, "body": b}
                         for i, b in enumerate(pages)],
    }


def test_turath_book_id_namespacing():
    # wide ids never collide with the committed shamela-seed ids
    assert ingest_mod.turath_book_id("shamela-1234") == "turath-1234"
    assert ingest_mod.turath_book_id("turath-1234") == "turath-1234"
    assert ingest_mod.turath_book_id("1234") == "turath-1234"
    assert ingest_mod.turath_book_id("shamela-1234") != "shamela-1234"


def test_ingest_turath_page_provenance_and_source(store):
    recs = [_rec("111", "10__علوم-الحديث", ["نص الصفحة الأولى كافٍ للتجربة، فالفقرة تُقسم حسب الهدف المحدد للمقطع في القاعدة الدقيقة والمعتمدة.",
                                            "نص الصفحة الثانية للتجربة نفسها مع التنويع في الحروف والجمل ليكون القسم صحيحًا ومطابقًا."], "مختصر التجربة"),
            _rec("222", "05__التراجم", ["مدخل الترجمة في الكتاب الثاني، نص كافٍ يمر فوق الحد الأدنى المطلوب للتجربة الموثقة هنا."])]
    rep = ingest_mod.ingest_turath_jsonl(store, recs)
    assert rep["format"] == "turath"
    assert rep["books"] == 2
    assert rep["passages"] > 0
    assert rep["duplicates_skipped"] == 0

    srcs = [r["source_id"] for r in store.conn.execute("SELECT source_id FROM sources")]
    assert "turath" in srcs
    books = {r["book_id"]: r for r in store.conn.execute("SELECT * FROM books")}
    assert set(books) == {"turath-111", "turath-222"}
    assert books["turath-111"]["source_id"] == "turath"
    assert books["turath-111"]["category"].startswith("تراث/")
    pages = store.conn.execute(
        "SELECT DISTINCT page_num FROM passages WHERE book_id='turath-111' "
        "ORDER BY page_num").fetchall()
    assert [p["page_num"] for p in pages] == [1, 2]


def test_ingest_turath_idempotent(store):
    recs = [_rec("333", "01__القرآن-الكرم-والعلوم-القرآنية",
                 ["نص التفسير للتجربة الحتمية يكرر مرتين بلا فروق في الأحرف أو الكلمات المكتوبة."])]
    first = ingest_mod.ingest_turath_jsonl(store, recs)
    second = ingest_mod.ingest_turath_jsonl(store, recs)
    assert first["passages"] > 0
    assert second["passages"] == 0
    assert second["books"] == 1  # INSERT OR REPLACE, no new book row
    assert second["duplicates_skipped"] == first["passages"]


def test_turath_layer_isolation_readonly_claims(store):
    # turath ingest must never touch claims / argument_relations of the
    # منقحة layers
    store.add_source("fikr", "fikr", "lic", "declared-open")
    store.add_claim("فكرة منسوبة للطبقة المنقحة")
    rels_before = store.conn.execute(
        "SELECT COUNT(*) c FROM argument_relations").fetchone()["c"]
    claims_before = store.conn.execute(
        "SELECT COUNT(*) c FROM claims").fetchone()["c"]
    ingest_mod.ingest_turath_jsonl(
        store, [_rec("444", "12__العقائد", ["نص عقدي للتجربة يتجاوز الحد الأدنى المطلوب للقسمة الموثقة في الاختبار المحدد هنا."])])
    claims_after = store.conn.execute(
        "SELECT COUNT(*) c FROM claims").fetchone()["c"]
    rels_after = store.conn.execute(
        "SELECT COUNT(*) c FROM argument_relations").fetchone()["c"]
    assert claims_after == claims_before
    assert rels_after == rels_before


def test_record_size_matches_pages():
    rec = _rec("555", "cat", ["أ" * 100, "ب" * 50])
    assert record_size(rec) == 150


def test_select_turath_core_budgeted_breadth_deterministic():
    cats = ["cat1", "cat2", "cat3"]
    recs = []
    for c in cats:
        for n in range(3):
            recs.append(_rec(f"{c}-{n}", c, ["x" * (1000 - 100 * n)],
                             title=f"كتاب {c} رقم {n}"))
    # a huge book that can never fit
    recs.append(_rec("huge", "cat3", ["y" * 500_000]))

    r1 = select_turath_core(recs, set(), budget_chars=3_500)
    r2 = select_turath_core(recs, set(), budget_chars=3_500)
    assert json.dumps(r1[0], ensure_ascii=False) == json.dumps(r2[0], ensure_ascii=False)
    selected, report = r1
    # round-robin breadth over size-sorted lists: round 1 takes the smallest
    # (800) of each category = 2400; round 2 takes 900 only while the budget
    # (3500) allows -> 3300 spent, cat1's next (1000) does not fit
    assert report["budget_spent_chars"] == 3_300
    assert {r["cat"] for r in selected} == set(cats)
    assert len(selected) == 4
    sizes = [record_size(r) for r in selected]
    assert sizes == [800, 800, 800, 900]
    assert report["records_selected"] == 4
    assert report["categories_covered"] == 3


def test_select_turath_core_skips_existing_and_unfit(store):
    recs = [_rec("777", "catA", ["z" * 500]),
            _rec("888", "catB", ["z" * 9000])]
    ingest_mod.ingest_turath_jsonl(store, [recs[0]])
    existing = {r["book_id"] for r in store.conn.execute("SELECT book_id FROM books")}
    selected, report = select_turath_core(recs, existing, budget_chars=2_000,
                                          spent_chars=0)
    # turath-777 already present -> only turath-888 remains and it does not fit
    assert selected == []
    assert report["records_skipped_existing"] == 1
    assert report["budget_spent_chars"] == 0


def test_select_turath_core_budget_is_layer_total_not_per_run():
    # the budget is the TOTAL of the committed layer: a completed build
    # re-run selects nothing (v1.16.0 fix — a per-run allowance kept
    # growing the committed DB on every re-run).  ``budget_spent_chars`` is
    # the cumulative layer spend; ``budget_new_chars`` is this run's spend.
    recs = [_rec("a1", "catA", ["x" * 1000]),
            _rec("a2", "catA", ["x" * 800])]
    sel1, rep1 = select_turath_core(recs, set(), budget_chars=2_500,
                                    spent_chars=0)
    assert len(sel1) == 2
    assert rep1["budget_new_chars"] <= 2_500
    existing = {ingest_mod.turath_book_id(r["id"]) for r in sel1}
    spent_total = rep1["budget_spent_chars"]
    sel2, rep2 = select_turath_core(recs, existing, budget_chars=2_500,
                                    spent_chars=spent_total)
    assert sel2 == []
    assert rep2["budget_spent_chars"] == spent_total
    assert rep2["budget_new_chars"] == 0
    assert rep2["budget_spent_before_chars"] == spent_total


def test_readonly_cli_never_persists_vectors(tmp_path, monkeypatch):
    # regression (v1.16.0): a CLI search on the committed DB used to
    # re-persist passage_vec and grow the file past GitHub's limit
    from ryko_lm.knowledge.__main__ import main as cli_main

    db = tmp_path / "ro.sqlite"
    ingest_mod.ingest_turath_jsonl(
        KnowledgeStore(str(db)),
        [_rec("666", "catRO", ["نص التجربة للتحقق من عدم استمرار المتجهات بعد البحث القرائي فقط دون كتابة."])])
    before = db.stat().st_size
    import sqlite3

    with sqlite3.connect(db) as conn:
        vec_before = conn.execute("SELECT COUNT(*) FROM passage_vec").fetchone()[0]

    rc = cli_main(["search", "--db", str(db), "نص التجربة", "-k", "2"])
    assert rc == 0
    rc2 = cli_main(["stats", "--db", str(db)])
    assert rc2 == 0
    with sqlite3.connect(db) as conn:
        vec_after = conn.execute("SELECT COUNT(*) FROM passage_vec").fetchone()[0]
    assert vec_before == 0
    assert vec_after == 0
    assert db.stat().st_size == before


def test_manifest_deterministic_and_documented():
    from ryko_lm.knowledge.turath import MANIFEST_KIND, PROTOCOL

    rows = [{"book_id": "turath-999", "title": "كتاب الوثيقة",
             "category": "تراث/catC", "raw_pages": 1, "chars": 400}]
    rep = {"policy": "budgeted-breadth", "budget_chars": 400,
           "budget_spent_chars": 400, "budget_new_chars": 400,
           "budget_spent_before_chars": 0, "records_available": 1,
           "records_skipped_existing": 0, "records_selected": 1,
           "categories_available": 1, "categories_covered": 1, "rounds": 1}
    m1 = build_manifest(rows, rep, {"module": "test"})
    m2 = build_manifest(list(reversed(rows)), rep, {"module": "test"})
    assert json.dumps(m1, ensure_ascii=False, sort_keys=True) == \
        json.dumps(m2, ensure_ascii=False, sort_keys=True)
    assert m1["kind"] == MANIFEST_KIND == "ryko-turath-manifest"
    assert m1["protocol"] == PROTOCOL == "ryko-turath/1"
    assert m1["claims_policy"].startswith("none")
    assert m1["books"][0]["book_id"] == "turath-999"
    assert m1["books"][0]["dataset_id"] == "shamela-999"
    assert m1["books"][0]["category"].startswith("تراث/")
    assert m1["counts"] == {"books": 1, "chars": 400}
