"""Accuracy — Knowledge Core: store ops, ranking, citation, audit (45 tests).

Everything runs against throwaway :memory: stores (the committed repo DB
is exercised by the API module).  The BM25/dense/hybrid expectations pin
the documented search contract: deterministic orderings, provenance on
every hit, and the ryko-cite/1 verdict gates.
"""

from __future__ import annotations

import json

import pytest
from comprehensive_helpers import SYNTHETIC_PASSAGES

from ryko_lm.knowledge.citation import (
    AUDIT_ATTRIBUTED,
    AUDIT_PARTIAL,
    AUDIT_UNATTRIBUTED,
    PROTOCOL,
    audit_answer,
    verify_quote,
)
from ryko_lm.knowledge.schema import (
    VERIFICATION_UNVERIFIED,
    VERIFICATION_VERIFIED,
)
from ryko_lm.knowledge.store import KnowledgeStore


@pytest.fixture()
def store():
    s = KnowledgeStore(":memory:")
    s.add_source("synth", "Synthetic Bench", "MIT", "declared-open")
    yield s
    s.close()


@pytest.fixture()
def seeded(store):
    store.add_book("bk1", "synth", "كتاب الاختبارات", category="تجريبي")
    store.add_book("bk2", "synth", "كتاب ثانٍ", category="أدلة")
    for i, text in enumerate(SYNTHETIC_PASSAGES):
        store.add_passage("bk1" if i % 2 == 0 else "bk2", text,
                          page_num=i // 3, seq=i % 3)
    return store


# ------------------------------------------------------------ store (15)
def test_stats_shape(store):
    st = store.stats()
    assert st["schema_version"] == 1
    assert st["sources"] == 1 and st["passages"] == 0 and st["books"] == 0


def test_unknown_rights_rejected(store):
    with pytest.raises(ValueError):
        store.add_source("x", "X", "MIT", "super-open")


def test_passage_insert_and_dedup(seeded):
    _pid, created = seeded.add_passage("bk1", SYNTHETIC_PASSAGES[0], 0, 0)
    assert created is False  # same normalized text in same book -> dedup
    _, created2 = seeded.add_passage("bk1", " " + SYNTHETIC_PASSAGES[0] + " ", 1, 0)
    assert created2 is False  # whitespace-only difference still dedups


def test_passage_empty_rejected(store):
    with pytest.raises(ValueError):
        store.add_passage("bk1", "   ")


def test_seq_auto_increments(seeded):
    s1, _ = seeded.add_passage("bk3", "نص أول في الخلية")
    s2, _ = seeded.add_passage("bk3", "نص ثانٍ في الخلية")
    assert s1 != s2


def test_seq_collision_bumped(seeded):
    seeded.add_passage("bk4", "نص مشغول الخلية", page_num=1, seq=0)
    _pid, created = seeded.add_passage("bk4", "نص متدخل", page_num=1, seq=0)
    assert created is True


def test_passage_id_prefix_stable(seeded):
    pid, _ = seeded.add_passage("bk5", "نص لفحص الهوية")
    assert pid.startswith("p-")


def test_claim_id_deterministic(store):
    a = store.add_claim("الزكاة ركن", claim_type="شرع")
    b = store.add_claim("الزكاة ركن", claim_type="شرع")
    assert a == b


def test_relation_closed_vocabulary(store):
    cid = store.add_claim("مصدر الحكم")
    rid = store.add_claim("مرتبط")
    with pytest.raises(ValueError):
        store.add_relation(cid, rid, "invents")


def test_relation_id_deterministic(store):
    c1, c2 = store.add_claim("الحكم الأول"), store.add_claim("الحكم الثاني")
    r1 = store.add_relation(c1, c2, "supports")
    r2 = store.add_relation(c1, c2, "supports")
    assert r1 == r2


def test_set_verification_validates_status(seeded):
    pid = next(seeded.iter_passages())["passage_id"]
    with pytest.raises(ValueError):
        seeded.set_verification(pid, "magic")


def test_set_verification_round_trip(seeded):
    pid = next(seeded.iter_passages())["passage_id"]
    seeded.set_verification(pid, VERIFICATION_VERIFIED)
    assert seeded.passage(pid)["verification_status"] == VERIFICATION_VERIFIED


def test_iter_passages_deterministic(seeded):
    rows = list(seeded.iter_passages())
    again = list(seeded.iter_passages())
    assert [r["passage_id"] for r in rows] == [r["passage_id"] for r in again]


def test_iter_books_ordered(seeded):
    titles = [b["title_ar"] for b in seeded.iter_books()]
    assert titles == sorted(titles)


def test_hit_dict_contract(seeded):
    hit = seeded.search("الزكاة", k=1, mode="hybrid")[0]
    d = hit.as_dict()
    for key in ("passage_id", "book_id", "title_ar", "source_id",
                "page_num", "seq", "text", "score", "bm25_rank", "dense_rank"):
        assert key in d


# ------------------------------------------------------ ranking (12)
def test_bm25_relevance_top1(seeded):
    hits = seeded.search("الزكاة", k=3, mode="bm25")
    assert hits and "الزكاة" in hits[0].text


def test_bm25_k_respected(seeded):
    assert len(seeded.search("الزكاة", k=2, mode="bm25")) <= 2


def test_bm25_deterministic_across_stores(seeded):
    other = KnowledgeStore(":memory:")
    other.add_source("synth", "Synthetic Bench", "MIT", "declared-open")
    other.add_book("bk1", "synth", "كتاب الاختبارات")
    other.add_book("bk2", "synth", "كتاب ثانٍ")
    for i, text in enumerate(SYNTHETIC_PASSAGES):
        other.add_passage("bk1" if i % 2 == 0 else "bk2", text,
                          page_num=i // 3, seq=i % 3)
    a = [h.passage_id for h in seeded.search("الصلاة والطهارة", k=5, mode="bm25")]
    b = [h.passage_id for h in other.search("الصلاة والطهارة", k=5, mode="bm25")]
    other.close()
    assert a == b


def test_book_filter_slices(seeded):
    hits = seeded.search("الزكاة", k=10, mode="bm25", book_id="bk1")
    assert hits and all(h.book_id == "bk1" for h in hits)


def test_book_filter_unknown_empty(seeded):
    assert seeded.search("الزكاة", k=5, book_id="ghost") == []


def test_empty_query_returns_nothing(seeded):
    assert seeded.search("123456", k=5, mode="bm25") == []


def test_diacritics_query_matches_bare_text(seeded):
    hits = seeded.search("الزَّكَاةُ", k=3, mode="bm25")
    assert hits and "الزكاة" in hits[0].text


def test_scores_non_negative(seeded):
    for mode in ("bm25", "dense", "hybrid"):
        assert all(h.score >= 0 for h in seeded.search("الإسلام", k=5, mode=mode))


def test_dense_mode_returns_hits(seeded):
    hits = seeded.search("الصلاة عمود الدين", k=3, mode="dense")
    assert hits and all(h.dense_rank is not None for h in hits)


def test_hybrid_returns_hits(seeded):
    hits = seeded.search("أركان الإسلام الخمسة", k=3, mode="hybrid")
    assert hits


def test_hybrid_overlaps_bm25_top(seeded):
    bm = {h.passage_id for h in seeded.search("الزكاة", k=3, mode="bm25")}
    hy = {h.passage_id for h in seeded.search("الزكاة", k=3, mode="hybrid")}
    assert bm & hy


def test_unknown_mode_raises(seeded):
    with pytest.raises(ValueError):
        seeded.search("الزكاة", mode="fuzzy")


# ------------------------------------------------------ citation (10)
def test_cite_verbatim(seeded):
    quote = SYNTHETIC_PASSAGES[0]
    v = verify_quote(seeded, quote, write_status=False)
    assert v.status == VERIFICATION_VERIFIED and v.score == 1.0


def test_cite_verbatim_with_diacritics(seeded):
    v = verify_quote(seeded, "الزَّكَاةُ فريضة من فرائض الإسلام", write_status=False)
    assert v.status == VERIFICATION_VERIFIED


def test_cite_protocol_fields(seeded):
    v = verify_quote(seeded, SYNTHETIC_PASSAGES[0], write_status=False)
    assert v.protocol == PROTOCOL == "ryko-cite/1"
    assert len(v.quote_sha256) == 64
    assert v.passage_id and v.book_id


def test_cite_partial_paraphrase(seeded):
    # same content words, reordered, not a literal substring
    v = verify_quote(seeded, "في المال الإسلام فريضة من فرائض الزكاة للفقراء",
                     write_status=False)
    assert v.status == "partial" and 0.82 <= v.score < 1.0


def test_cite_token_full_cover_is_never_verified(seeded):
    # every quote token exists in a passage but the string is NOT a
    # substring -> must stay partial (v1.18.0 containment cap)
    v = verify_quote(seeded, "الإسلام فريضة الزكاة المال للفقراء من فرائض",
                     write_status=False)
    assert v.status == "partial" and v.score <= 0.9999


def test_cite_not_found(seeded):
    v = verify_quote(seeded, "العجاج الشفاف يكتب الشفرات العليا",
                     write_status=False)
    assert v.status == "not_found" and v.passage_id is None


def test_cite_empty_quote_raises(seeded):
    # diacritics-only quotes normalize to the empty string -> refused
    with pytest.raises(ValueError):
        verify_quote(seeded, "ًٌٍَُِّْ", write_status=False)


def test_cite_write_status_false_keeps_store(seeded):
    pid = next(seeded.iter_passages())["passage_id"]
    verify_quote(seeded, SYNTHETIC_PASSAGES[0], write_status=False)
    assert seeded.passage(pid)["verification_status"] == VERIFICATION_UNVERIFIED


def test_cite_write_status_true_upgrades(seeded):
    verify_quote(seeded, SYNTHETIC_PASSAGES[0], write_status=True)
    pid = seeded.search("الزكاة فريضة", k=1, mode="bm25")[0].passage_id
    assert seeded.passage(pid)["verification_status"] == VERIFICATION_VERIFIED


def test_cite_verdict_json_round_trip(seeded):
    v = verify_quote(seeded, SYNTHETIC_PASSAGES[0], write_status=False)
    parsed = json.loads(v.as_json())
    assert parsed["status"] == VERIFICATION_VERIFIED


# ---------------------------------------------------- audit (8)
def test_audit_attributed(seeded):
    hits = seeded.search("الزكاة", k=5, mode="hybrid")
    answer = SYNTHETIC_PASSAGES[0]
    a = audit_answer(answer, hits)
    assert a.status == AUDIT_ATTRIBUTED and a.attributed_ratio == 1.0


def test_audit_unattributed(seeded):
    hits = seeded.search("الزكاة", k=5, mode="hybrid")
    a = audit_answer("الشفرات العليا تعمل الآن. العجاج الشفاف يلمع.", hits)
    assert a.status == AUDIT_UNATTRIBUTED


def test_audit_partial(seeded):
    hits = seeded.search("الزكاة", k=5, mode="hybrid")
    a = audit_answer("في المال الإسلام فريضة من فرائض الزكاة للفقراء.", hits)
    assert a.status == AUDIT_PARTIAL


def test_audit_sentence_limit(seeded):
    hits = seeded.search("الزكاة", k=5, mode="hybrid")
    long_answer = ". ".join([SYNTHETIC_PASSAGES[0]] * 3)
    a = audit_answer(long_answer, hits, max_sentences=1)
    assert a.sentences_audited == 1


def test_audit_split_on_punctuation(seeded):
    hits = seeded.search("الزكاة", k=5, mode="hybrid")
    answer = (SYNTHETIC_PASSAGES[0] + "\n" + "جملة قصيرة جداً هنا وليست مؤهلة.")
    a = audit_answer(answer, hits)
    assert a.verdicts[0].status == VERIFICATION_VERIFIED


def test_audit_accepts_plain_strings(seeded):
    a = audit_answer(SYNTHETIC_PASSAGES[0], [SYNTHETIC_PASSAGES[0]])
    assert a.status == AUDIT_ATTRIBUTED


def test_audit_best_is_highest(seeded):
    hits = seeded.search("الزكاة", k=5, mode="hybrid")
    answer = "جملة منثورة بلا سند هنا. " + SYNTHETIC_PASSAGES[0]
    a = audit_answer(answer, hits)
    assert a.best.status == VERIFICATION_VERIFIED


def test_audit_dict_contract(seeded):
    hits = seeded.search("الزكاة", k=5, mode="hybrid")
    d = audit_answer(SYNTHETIC_PASSAGES[0], hits).as_dict()
    assert {"protocol", "sentences_audited", "verdicts", "attributed_ratio",
            "status", "best", "latency_ms"} <= set(d)
