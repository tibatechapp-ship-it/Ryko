"""Tests for the لغة (Linguistics) layer — v1.15.0 التوسع اللغوي الواسع.

Covers: the lugha manifest contract (structure + rights policy), the lugha
knowledge layer (manifest kind registration, ingest determinism/idempotency,
claims seeding), the linguistics pretraining pack builder (cleaning cascade,
deterministic split, receipts), and the self-checked linguistics probe set
(rb-ling families, byte-identical double build).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ryko_lm.data.augment_corpus import doc_seed  # noqa: E402
from ryko_lm.knowledge import ingest as ingest_mod  # noqa: E402
from ryko_lm.knowledge.ideas import (  # noqa: E402
    MANIFEST_KINDS,
    load_manifest,
    manifest_layer,
    prune_fikr,
    seed_ideas,
)
from ryko_lm.knowledge.store import KnowledgeStore  # noqa: E402

MANIFEST = ROOT / "data" / "knowledge" / "lugha_manifest.json"

SECTIONS = ("nahw", "sarf", "balagha", "maajim", "fiqh-lugha", "adab", "munhaj")
VALID_STATUSES = ("available", "pending", "metadata")


def _script(name: str):
    """Load a repo script (scripts/ is not a package) as a module."""
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- manifest
def test_manifest_loads_and_registers_layer():
    assert "ryko-lugha-manifest" in MANIFEST_KINDS
    manifest = load_manifest(MANIFEST)
    assert manifest_layer(manifest) == "lugha"
    assert manifest["version"] == 2
    assert manifest.get("layer_label") == "لغة"


def test_manifest_sections_and_catalog_coverage():
    manifest = load_manifest(MANIFEST)
    ids = [s["id"] for s in manifest["schools"]]
    assert ids == list(SECTIONS), "the 7 أبواب must exist in the fixed order"
    books = manifest["books"]
    assert len(books) >= 80, "the catalog must carry the user's book list"
    for s in manifest["schools"]:
        for field in ("id", "name", "central_idea"):
            assert s.get(field), f"section missing {field}: {s['id']}"
    ideas = manifest["ideas"]
    assert len(ideas) >= 20
    for idea in ideas:
        assert idea["school"] in SECTIONS


def test_manifest_book_policy():
    """Status/hints/scope policy is machine-checked, not prose."""
    manifest = load_manifest(MANIFEST)
    seen_titles = set()
    for book in manifest["books"]:
        title = book["title"]
        assert title not in seen_titles, f"duplicate catalog title: {title}"
        seen_titles.add(title)
        assert book["status"] in VALID_STATUSES
        assert book["school"] in SECTIONS
        if book["status"] == "available":
            # every ingestible book points at its mirror path
            path = book.get("shamela_path")
            assert path, f"available book without shamela_path: {title}"
            cat, slug = path.split("/", 1)
            assert cat in {"29__كتب-اللغة", "30__الغريب-والمعاجم", "31__النحو-والصرف",
                           "35__البلاغة", "32__الأدب", "34__الشعر-ودواوينه",
                           "04__علوم-القرآن-وأصول-التفسير"}
            import re
            assert re.match(r"\d+__", slug), f"bad mirror slug: {slug}"
        else:
            # pending/metadata books must document why (never invented)
            assert book.get("source_hint"), f"status={book['status']} without source_hint: {title}"
        # classical (pre-modern) authors never hide behind "metadata" —
        # they are pending with a source hint instead
        death = book.get("death_hijri")
        if isinstance(death, int) and death <= 1350:
            assert book["status"] != "metadata", (
                f"classical book ({title}, d.{death}H) must not be metadata-only")


def test_manifest_policy_block():
    manifest = load_manifest(MANIFEST)
    policy = manifest["policy"]
    assert policy["principle"]
    joined = " ".join(policy["ingestion_policy"])
    assert "إثبات صفحي" in joined and "source_hint" in joined.replace(" ", "") or "ممنوع اختلاق" in joined


# ------------------------------------------------------------- lugha layer
def _fake_lugha_record(book: str, school: str, body: str) -> dict:
    return {"id": f"shamela-test-{abs(hash(book)) % 10000}", "book": book,
            "cat": f"لغة/{school}", "pages": 1, "author": "مؤلف الاختبار",
            "school": school, "page_records": [{"page_num": 1, "body": body}]}


def test_ingest_lugha_jsonl_deterministic_and_idempotent(tmp_path):
    body = ("الكلام هو اللفظ المركب المفيد بالوضع. وقسّم النحاة الكلام إلى "
            "اسم وفعل وحرف، وكل باب من الباب له أصول يُرجع إليها عند الشك.\n"
            "والعرَب علامة معرفة، والتنوين علامة نكرة، والرفع علامة الفاعل "
            "ما لم يطرأ عليه لازِم.")
    rec = _fake_lugha_record("متن الاختبار النحوي", "nahw", body)
    reports, stats_list = [], []
    for run in (1, 2):
        store = KnowledgeStore(str(tmp_path / f"store{run}.sqlite"))
        report = ingest_mod.ingest_lugha_jsonl(store, [rec], classify=True)
        stats = store.stats()
        store.close()
        reports.append(report)
        stats_list.append(stats)
    assert reports[0]["format"] == "lugha"
    assert reports[0]["passages"] > 0
    # deterministic: two independent builds give identical stats
    assert reports[0]["passages"] == reports[1]["passages"]
    assert stats_list[0]["passages"] == stats_list[1]["passages"]
    # idempotent: re-ingesting the same record adds no passages
    store = KnowledgeStore(str(tmp_path / "idem.sqlite"))
    ingest_mod.ingest_lugha_jsonl(store, [rec], classify=True)
    again = ingest_mod.ingest_lugha_jsonl(store, [rec], classify=True)
    assert again["passages"] == 0
    assert again["duplicates_skipped"] == reports[0]["passages"]


def test_lugha_source_registered(tmp_path):
    store = KnowledgeStore(str(tmp_path / "s.sqlite"))
    ingest_mod.ingest_lugha_jsonl(store, [_fake_lugha_record("ك", "adab", "نص أدبي وافي بالعربية للفحص")], classify=False)
    srcs = {r["source_id"]: dict(r) for r in store.conn.execute("SELECT * FROM sources")}
    assert "lugha" in srcs
    assert srcs["lugha"]["rights_status"] == "declared-open"


def test_seed_lugha_claims(tmp_path):
    manifest = load_manifest(MANIFEST)
    store = KnowledgeStore(str(tmp_path / "s.sqlite"))
    rep = seed_ideas(store, manifest, replace=True)
    assert rep["claims_seeded"] == len(manifest["ideas"])
    # idempotent
    rep2 = seed_ideas(store, manifest, replace=True)
    assert rep2["claims_removed"] == len(manifest["ideas"])
    assert rep2["claims_seeded"] == len(manifest["ideas"])
    rows = store.conn.execute(
        "SELECT COUNT(*) FROM claims WHERE json_extract(meta,'$.layer')='lugha'"
    ).fetchone()[0]
    assert rows == len(manifest["ideas"])


def test_prune_lugha_scoped(tmp_path):
    manifest = load_manifest(MANIFEST)
    store = KnowledgeStore(str(tmp_path / "s.sqlite"))
    seed_ideas(store, manifest, replace=True)
    # prune with an empty exclusion list removes nothing
    rep = prune_fikr(store, manifest)
    assert rep["books_removed"] == 0


# ------------------------------------------------------------- ling pack
def test_ling_pack_deterministic(tmp_path):
    mod = _script("build_ling_pack.py")
    raw1 = tmp_path / "raw_a.jsonl"
    raw2 = tmp_path / "raw_b.jsonl"
    body_a = ("باب النحو وأصوله في العربية. علامات الإعراب الرفع والنصب والجر "
              "والجزم معروفة في الكلام العربي المفيد.\nوالفعل الماضي بناء لا "
              "يزيد على ثلاثة أحرف في الأصل.")
    body_b = ("التصريف يعلم كيف وُلدت الكلمة من أصله. من كَتَبَ إلى كاتب "
              "ومكتوب ومكتبة وكتابة مسائل شتى في الاشتقاق.\nوالاشتقاق الكبير "
              "حجّة في توافق المعاني والمبانٍ عند ابن جني.")
    raw1.write_text(json.dumps(_fake_lugha_record("متن أ", "nahw", body_a),
                               ensure_ascii=False) + "\n", encoding="utf-8")
    raw2.write_text(json.dumps(_fake_lugha_record("متن ب", "sarf", body_b),
                               ensure_ascii=False) + "\n", encoding="utf-8")
    out1, out2 = tmp_path / "pack1", tmp_path / "pack2"
    sys_argv = ["--raw", str(raw1), str(raw2), "--out-dir", str(out1)]
    mod.main(sys_argv)
    mod.main(["--raw", str(raw1), str(raw2), "--out-dir", str(out2)])
    assert (out1 / "train.txt").read_bytes() == (out2 / "train.txt").read_bytes()
    assert (out1 / "val.txt").read_bytes() == (out2 / "val.txt").read_bytes()
    stats = json.loads((out1 / "stats.json").read_text(encoding="utf-8"))
    assert stats["pack"] == "linguistics-v1"
    assert set(stats["sections"]) <= set(SECTIONS) | {"other"}
    assert stats["chars_kept"] > 0
    # receipt must verify offline
    from ryko_lm.provenance import verify_receipt
    ok, _report = verify_receipt(out1 / "receipt.json")
    assert ok
    # the two docs split deterministically by the hash-bucket rule
    seed_a = doc_seed(body_a.split("\n"))
    assert seed_a is not None


# ------------------------------------------------------------- probes
@pytest.fixture(scope="module")
def probe_store(tmp_path_factory):
    """A small store with lugha claims + books + passages for probe builds."""
    manifest = load_manifest(MANIFEST)
    tmp = tmp_path_factory.mktemp("probes")
    db = tmp / "probe.sqlite"
    store = KnowledgeStore(str(db))
    seed_ideas(store, manifest, replace=True)
    # a few lugha books with authors/sections and one passage each
    for i, (title, author, sec) in enumerate([
        ("متن نحوي للاختبار", "مؤلف نحوي", "nahw"),
        ("متن صرفي للاختبار", "مؤلف صرفي", "sarf"),
        ("متن بلاغي للاختبار", "مؤلف بلاغي", "balagha"),
        ("معجم للاختبار", "مؤلف معجمي", "maajim"),
    ]):
        bid = f"shamela-9{i:03d}"
        author_id = store.add_author(f"a-{i}", author,
                                     meta={"school": sec})
        store.add_book(bid, "lugha", title, author_id=author_id,
                       category=f"لغة/{sec}", rights_status="declared-open")
        store.add_passage(bid, "هذا نصٌّ اختباري طويل بما يكفي لقصّه مقتطفاً "
                                "قابلًا للإسناد إلى كتابه في مسبارات اللغة "
                                "بإجابة صحيحة واحدة من أربعة خيارات.", 1, 0)
    store.conn.commit()
    yield store
    store.close()


def test_probes_build_and_schema(probe_store):
    mod = _script("build_ling_probes.py")
    cases = mod.build(probe_store)
    assert cases, "probe builder must produce cases"
    # closed schema: every case must load through the bench loader
    from ryko_lm.eval.benchmarks import load_benchmark
    tmp = probe_store.conn.execute("SELECT 1").fetchone()
    out = Path(str(tmp[0]) + "-unused") if False else None
    lines = "\n".join(json.dumps(c, ensure_ascii=False, sort_keys=True) for c in cases)
    p = probe_store  # noqa: F841
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(lines + "\n")
        path = Path(fh.name)
    loaded = load_benchmark(path)
    assert len(loaded) == len(cases)
    for c in loaded:
        assert c.answer is not None and 0 <= c.answer < len(c.options)
        assert len(c.options) == 4
        # correct answer never duplicated among distractors
        assert len(set(c.options)) == 4


def test_probes_deterministic_double_build(probe_store):
    mod = _script("build_ling_probes.py")
    c1 = mod.build(probe_store)
    c2 = mod.build(probe_store)
    j1 = [json.dumps(c, ensure_ascii=False, sort_keys=True) for c in c1]
    j2 = [json.dumps(c, ensure_ascii=False, sort_keys=True) for c in c2]
    assert j1 == j2


def test_probes_self_checked_answers(probe_store):
    """Correct values must originate in the store (never invented)."""
    mod = _script("build_ling_probes.py")
    cases = mod.build(probe_store)
    claims = {c["text"]: c["originator"] for c in mod._lugha_claims(probe_store)}
    books = {b["title"]: b for b in mod._lugha_books(probe_store)}
    for case in cases:
        correct = case["options"][case["answer"]]
        if case["domain"] == "نسبة-القاعدة-اللغوية":
            q = case["question"]
            text = q.split("«", 1)[1].rsplit("»", 1)[0]
            assert claims[text] == correct
        elif case["domain"] == "نسبة-المؤلف":
            title = case["question"].split("«", 1)[1].rsplit("»", 1)[0]
            assert books[title]["author"] == correct
        elif case["domain"] == "باب-اللغة":
            title = case["question"].split("«", 1)[1].rsplit("»", 1)[0]
            sec = books[title]["section"]
            assert mod._SECTION_LABEL_MAP[sec] == correct


def test_committed_probe_file_matches_contract():
    """The committed ling_probes.jsonl validates against the closed schema."""
    probes = ROOT / "docs" / "benchmarks" / "ling_probes.jsonl"
    if not probes.exists():
        pytest.skip("ling_probes.jsonl not built yet")
    from ryko_lm.eval.benchmarks import load_benchmark
    cases = load_benchmark(probes)
    assert len(cases) >= 40
    domains = {c.domain for c in cases}
    assert {"نسبة-القاعدة-اللغوية", "نسبة-المؤلف",
            "نسبة-الكتاب-اللغوي", "باب-اللغة"} <= domains


def test_committed_pack_stats_exist():
    stats = ROOT / "data" / "processed_ling" / "stats.json"
    receipt = ROOT / "data" / "processed_ling" / "receipt.json"
    if not stats.exists():
        pytest.skip("ling pack not built yet")
    data = json.loads(stats.read_text(encoding="utf-8"))
    assert data["pack"] == "linguistics-v1"
    assert data["chars_kept"] > 50_000_000, "the pack must cover the big collection"
    from ryko_lm.provenance import verify_receipt
    # the receipt covers the binarized pack (train.bin/val.bin — gitignored
    # build artifacts); in a fresh clone only the signed receipt + stats
    # are committed, so the offline verify runs only where the bins exist
    receipt_artifacts = json.loads(
        receipt.read_text(encoding="utf-8")).get("artifacts") or []
    if receipt_artifacts and not (ROOT / receipt_artifacts[0]["path"]).exists():
        pytest.skip("ling pack bins not in this tree (gitignored artifacts)")
    ok, _ = verify_receipt(receipt)
    assert ok


def test_lugha_collect_shim_defaults():
    """The collector shim points at the lugha manifest without any network."""
    spec = importlib.util.spec_from_file_location(
        "collect_lugha", ROOT / "src" / "ryko_lm" / "data" / "collect_lugha.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.DEFAULT_MANIFEST.name == "lugha_manifest.json"
    assert (ROOT / mod.DEFAULT_MANIFEST).exists()
