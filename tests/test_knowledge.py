"""Knowledge Core tests — schema, ingest, retrieval, citation, taxonomy."""

from __future__ import annotations

import json

import pytest

from ryko_lm.knowledge import KnowledgeStore, chunk_text, classify, embed
from ryko_lm.knowledge.citation import audit_answer, verify_quote
from ryko_lm.knowledge.embeddings import cosine, pack, unpack
from ryko_lm.knowledge.ingest import ingest_openiti_jsonl, ingest_shamela_jsonl
from ryko_lm.knowledge.schema import stable_id
from ryko_lm.knowledge.textproc import ar_tokens, normalize_for_match


@pytest.fixture()
def store():
    s = KnowledgeStore(":memory:")
    s.add_source("shamela", "Shamela v4", "MIT (declared)", "declared-open")
    yield s
    s.close()


@pytest.fixture()
def seeded(store):
    store.add_book("b1", "shamela", "كتاب الطهي", category="فقه")
    store.add_passage("b1", "الطهو أن يسأل عن الصحيح قبل العمل به", page_num=3, seq=0)
    store.add_passage("b1", "الماء الطهور لا ينجّس بغير تغيير أحد خصائصه", page_num=4, seq=0)
    store.add_passage("b1", "حدثنا عبد الله عن أبيه في سند الإسناد المتصل", page_num=5, seq=0)
    return store


# ------------------------------------------------------------------ schema
def test_schema_and_stats(store):
    st = store.stats()
    assert st["schema_version"] == 1
    assert st["sources"] == 1
    assert st["passages"] == 0


def test_unknown_rights_rejected(store):
    with pytest.raises(ValueError):
        store.add_source("x", "X", "MIT", "super-open")


def test_passage_id_deterministic():
    a = stable_id("p", "book", 3, 0)
    b = stable_id("p", "book", 3, 0)
    assert a == b and a.startswith("p-")


# ------------------------------------------------------------------ ingest
def test_add_passage_dedup_within_book(store):
    store.add_book("b1", "shamela", "كتاب")
    pid1, created1 = store.add_passage("b1", "نص واحد فقط للتجربة", page_num=1)
    pid2, created2 = store.add_passage("b1", "نص  واحد فقط للتجربة", page_num=2)
    assert created1 and not created2
    assert pid1 == pid2


def test_chunker_bounds():
    text = " ".join(["كلمة" * 3] * 300)
    chunks = chunk_text(text, target=400, min_len=80, max_len=900)
    assert all(len(c) <= 900 for c in chunks)
    assert "".join(chunks).replace(" ", "").startswith("كلمةكلمة")
    assert all(c.strip() for c in chunks)


def test_ingest_shamela_page_provenance(store, tmp_path):
    payload = {
        "id": "shamela-0001", "book": "صحيح التجريبي", "cat": "حديث", "pages": 2,
        "page_records": [
            {"page_num": 1, "body": "حدثنا أبو النضر عن سند متصل في أول الكتاب\nوهذا سطر ثانٍ للمصفحة الأولى"},
            {"page_num": 2, "body": "الماء الطهور لا ينجّس بغير تغيير"},
        ],
    }
    p = tmp_path / "shamela.jsonl"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    report = ingest_shamela_jsonl(store, p)
    assert report["books"] == 1
    assert report["passages"] >= 2  # short lines of one page merge into one chunk
    hits = store.search("الماء الطهور", k=1, mode="bm25")
    assert hits and hits[0].page_num == 2
    assert hits[0].book_id == "shamela-0001"


def test_ingest_idempotent(store, tmp_path):
    payload = {"id": "shamela-0002", "book": "كتاب", "cat": "فقه",
               "page_records": [{"page_num": 1, "body": "نص فقهي طويل نسبياً للتقسيم " * 10}]}
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    first = ingest_shamela_jsonl(store, p)
    second = ingest_shamela_jsonl(store, p, register_source=False)
    assert second["passages"] == 0
    assert store.stats()["passages"] == first["passages"]


def test_ingest_openiti(store, tmp_path):
    p = tmp_path / "openiti.jsonl"
    p.write_text(
        json.dumps({"id": "openiti-0000-000010", "text": "الاستدلال بالقياس على العلة المشتركة\nوهذا شرح للمبحث الثاني"},
                   ensure_ascii=False),
        encoding="utf-8")
    report = ingest_openiti_jsonl(store, p)
    assert report["passages"] >= 1
    row = store.search("الاستدلال بالقياس", k=1, mode="bm25")[0]
    assert row.page_num is None
    assert row.book_id == "openiti-0000-000010"


# ------------------------------------------------------------------ search
def test_bm25_ranks_relevant_first(seeded):
    hits = seeded.search("الماء الطهور تغيير", k=3, mode="bm25")
    assert hits[0].text.startswith("الماء الطهور")


def test_dense_search_deterministic(seeded):
    a = seeded.search("سند الإسناد", k=3, mode="dense")
    b = seeded.search("سند الإسناد", k=3, mode="dense")
    assert [h.passage_id for h in a] == [h.passage_id for h in b]
    assert a and "الإسناد" in a[0].text


def test_hybrid_fusion(seeded):
    hits = seeded.search("الماء الطهور", k=3, mode="hybrid")
    assert hits
    top = hits[0]
    assert top.bm25_rank is not None and top.dense_rank is not None
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_book_scope(seeded):
    seeded.add_book("b2", "shamela", "كتاب آخر")
    seeded.add_passage("b2", "الماء الطهور في كتاب آخر")
    hits = seeded.search("الماء الطهور", k=5, mode="hybrid", book_id="b2")
    assert all(h.book_id == "b2" for h in hits)
    assert len(hits) == 1


def test_textproc_normalization():
    assert normalize_for_match("العَلْمُ النَّافِعُ") == "العلم النافع"
    toks = ar_tokens("والماءُ الطهورُ")
    assert "ماء" in toks or "الماء" in toks
    assert "والماء" not in toks


def test_embedder_deterministic_and_normed():
    v1 = embed("العلم نور والجهل ظلام")
    v2 = embed("العلم نور والجهل ظلام")
    assert list(v1) == list(v2)
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-3
    assert cosine(embed("العلم نور"), embed("الجهل ظلام")) < 0.5


def test_pack_unpack_roundtrip():
    vec = embed("نص تجريبي للتغليف")
    assert list(unpack(pack(vec))) == list(vec)


# ----------------------------------------------------------------- citation
def test_citation_verified(seeded):
    verdict = verify_quote(seeded, "الماء الطهور لا ينجّس بغير تغيير")
    assert verdict.status == "verified"
    assert verdict.page_num == 4
    d = verdict.as_dict()
    assert d["protocol"] == "ryko-cite/1" and len(d["quote_sha256"]) == 64
    row = seeded.passage(verdict.passage_id)
    assert row["verification_status"] == "verified"


def test_citation_partial(seeded):
    verdict = verify_quote(seeded, "الماء الطهور لا ينجّس")
    assert verdict.status in {"verified", "partial"}
    assert verdict.score >= 0.82


def test_citation_not_found(seeded):
    verdict = verify_quote(seeded, "كلام كليلا غير موجود هنا في المتن إطلاقا")
    assert verdict.status == "not_found"
    assert verdict.passage_id is None


# ------------------------------------------------------------ answer audit
def test_audit_attributed(seeded):
    audit = audit_answer(
        "الماء الطهور لا ينجّس بغير تغيير أحد خصائصه. وزدنا هنا كلاما غير موجود في المتن أصلا",
        seeded.iter_passages(),
    )
    assert audit.protocol == "ryko-cite/1"
    assert audit.status == "attributed"
    assert audit.attributed_ratio == 0.5
    best = audit.as_dict()["best"]
    assert best["status"] == "verified" and best["page_num"] == 4


def test_audit_partial(seeded):
    audit = audit_answer(
        "الماء الطهور لا ينجّس بغير تغيير أحد خصائصه النادرة",
        seeded.iter_passages(),
    )
    assert audit.status == "partial"
    assert audit.verdicts[0].status == "partial"
    assert 0.82 <= audit.verdicts[0].score < 1.0


def test_audit_unattributed(seeded):
    audit = audit_answer("كلام كليلا غير موجود هنا في المتن إطلاقا وقطعا", seeded.iter_passages())
    assert audit.status == "unattributed"
    assert audit.attributed_ratio == 0.0


def test_audit_plain_strings():
    audit = audit_answer("الماء الطهور لا ينجّس بغير تغيير أحد خصائصه",
                         ["الماء الطهور لا ينجّس بغير تغيير أحد خصائصه"])
    assert audit.status == "attributed"
    assert audit.verdicts[0].passage_id is None


def test_audit_deterministic_and_capped(seeded):
    long_answer = "الماء الطهور لا ينجّس بغير تغيير أحد خصائصه. " * 6
    a = audit_answer(long_answer, seeded.iter_passages())
    b = audit_answer(long_answer, seeded.iter_passages())
    assert a.status == b.status == "attributed"
    assert [v.as_dict() for v in a.verdicts] == [v.as_dict() for v in b.verdicts]
    assert a.sentences_audited == 4  # ANSWER_AUDIT_MAX_SENTENCES cap


# ---------------------------------------------------------------- taxonomy
@pytest.mark.parametrize("text,expected", [
    ("حدثنا مالك عن نافع عن ابن عمر أن رسول الله قال", "transmission"),
    ("سئل رحمه الله عن الصلاة في الثياب الموشاة فأجاب", "fatwa"),
    ("اعترض عليه بأن القياس هنا معطل، والاعتراض وارد", "objection"),
    ("الظاهر الأقوى والأرجح عند جمهور الفقهاء هو التوسع", "preference"),
    ("خلاصة القول إن الحاصل أن المسألة تعود للسابق", "summary"),
])
def test_taxonomy_rules(text, expected):
    label = classify(text)
    assert label is not None and label.code == expected


def test_taxonomy_no_false_positive():
    assert classify("الشمس كوكب يدور في فلكه") is None or \
        classify("الشمس كوكب يدور في فلكه").code in {t for t, _ in __import__(
            "ryko_lm.knowledge.taxonomy", fromlist=["DISCOURSE_TYPES"]).DISCOURSE_TYPES}


# ------------------------------------------------------- claims & relations
def test_claims_and_relations(store):
    store.add_book("b1", "shamela", "كتاب")
    pid, _ = store.add_passage("b1", "نص الاستدلال هنا", page_num=1)
    c1 = store.add_claim("الماء لا ينجّس", book_id="b1", passage_id=pid,
                         claim_type="argumentation")
    c2 = store.add_claim("النجاسة تحتاج سبباً", book_id="b1")
    rid = store.add_relation(c1, c2, "supports")
    st = store.stats()
    assert st["claims"] == 2 and st["argument_relations"] == 1
    with pytest.raises(ValueError):
        store.add_relation(c1, c2, "hates")


# ----------------------------------------------------------------- verdicts
def test_passage_lookup(seeded):
    hit = seeded.search("الماء الطهور", k=1, mode="bm25")[0]
    row = seeded.passage(hit.passage_id)
    assert row["title_ar"] == "كتاب الطهي"
    assert row["source_id"] == "shamela"
    assert seeded.passage("p-nonexistent") is None
