"""DPO round-2 bundle tests (v1.23.0) — frozen-prefix byte identity,
closed-catalog linguistic distractors, leakage fatality, determinism,
and the real committed-seed contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from conftest import ROOT

from ryko_lm.knowledge.store import KnowledgeStore
from ryko_lm.training import sft_prep
from ryko_lm.training.sft_prep import build_dpo_r2

from test_ryko1000_eval import WORDS, _case  # noqa: F401

R1_PATH = ROOT / "data" / "training" / "ryko1000_dpo_train.jsonl"
R2_PATH = ROOT / "data" / "training" / "ryko1000_dpo_train_r2.jsonl"
SEED_PATH = ROOT / "docs" / "benchmarks" / "ryko1000_seed.jsonl"
REAL_STORE = ROOT / "data" / "knowledge" / "ryko.sqlite"


# ------------------------------------------------------------ synthetic
@pytest.fixture(scope="module")
def ling_store_path(tmp_path_factory) -> Path:
    """Synthetic store with a two-author/two-book catalog."""
    path = tmp_path_factory.mktemp("ling") / "store.sqlite"
    s = KnowledgeStore(str(path))
    s.add_source("shamela", "Shamela", "MIT", "declared-open")
    s.add_author("a1", "الشاطبي", 790)
    s.add_author("a2", "ابن جني", 392)
    for b in range(2):
        s.add_book(f"bk{b}", "shamela",
                   f"كتاب {WORDS[b]} المبين رقم {b}",
                   author_id=f"a{b + 1}", category="لغة")
        for i in range(3):
            s.add_passage(f"bk{b}",
                          f"باب {WORDS[i]} المبين رقم {b} نص عربي "
                          f"{WORDS[b + 3]} للمتابعة {WORDS[i]}",
                          page_num=i, seq=i)
    s.close()
    return path


def _ling_case(sid: KnowledgeStore, cid: str, family: str, pattern: str,
               expected: str, own_author: str = "a1",
               own_book: str = "bk0") -> dict:
    pid = next(sid.iter_passages())["passage_id"]
    probes = [{"kind": "wrong_attribution",
               "probe": f"نسبة المقطع إلى غير مثبت ({pattern})",
               "must": "reject"}]
    meta = {"ling_pattern": pattern,
            "author_id": own_author, "book_id": own_book}
    if pattern == "rule":
        meta = {"ling_pattern": "rule"}
    return _case(sid, cid, family, "mcq", pid,
                 question=f"سؤال لغوي ({pattern}) رقم {cid}؟",
                 options=[expected, "بديل أول", "بديل ثانٍ", "بديل ثالث"],
                 correct=expected,
                 probes=probes, meta=meta)


@pytest.fixture()
def ling_seed(ling_store_path, tmp_path) -> Path:
    """15 linguistic MCQ cases (5 per family: author/book/continuation x2/
    rule) + 1 core MCQ + 1 dev must-answer abstain row + 1 test row."""
    sid = KnowledgeStore(str(ling_store_path), persist_vectors=False)
    titles = [b["title_ar"] for b in sid.iter_books()]
    cases = []
    rules = {"irab": "منشأ أ — الأصل الأول",
             "sarf": "منشأ ب — الأصل الثاني",
             "balagha": "منشأ ج — الأصل الثالث"}
    for fam in ("irab", "sarf", "balagha"):
        cases.append(_ling_case(sid, f"ry10-{fam}-001", fam, "author",
                                "الشاطبي"))
        cases.append(_ling_case(sid, f"ry10-{fam}-002", fam, "book",
                                titles[0], own_book="bk0"))
        cases.append(_ling_case(sid, f"ry10-{fam}-003", fam,
                                "continuation", "متابعة أولى",
                                own_book="bk0"))
        cases.append(_ling_case(sid, f"ry10-{fam}-004", fam,
                                "continuation", "متابعة ثانية",
                                own_book="bk1"))
        cases.append(_ling_case(sid, f"ry10-{fam}-005", fam, "rule",
                                rules[fam]))
    cases.append(_case(sid, "ry10-fahm-001", "fahm", "mcq",
                       next(sid.iter_passages())["passage_id"],
                       question="سؤال غير لغوي؟",
                       options=["أ", "ب", "ج", "د"], correct="أ",
                       probes=[{"kind": "wrong_attribution",
                                "probe": "x", "must": "reject"}]))
    dev = [_case(sid, "ry10-yaqin-090", "yaqin", "abstain",
                 next(sid.iter_passages())["passage_id"],
                 question="سؤال يجب الإجابة عنه؟", should_answer=True,
                 expected="منطلق مثبت", options=["أ", "ب"],
                 criterion="abstain-must-answer",
                 probes=[{"kind": "wrong_attribution", "probe": "x",
                          "must": "reject"}])]
    test = [_case(sid, "ry10-irab-090", "irab", "mcq",
                  next(sid.iter_passages())["passage_id"],
                  question="سؤال محجوز؟", options=["أ", "ب", "ج", "د"],
                  correct="ب", probes=[])]
    sid.close()
    split = tmp_path / "split"
    split.mkdir()
    (split / "ryko1000_dev.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in dev) + "\n",
        encoding="utf-8")
    (split / "ryko1000_test.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in test) + "\n",
        encoding="utf-8")
    (split / "ryko1000_train.jsonl").write_text("", encoding="utf-8")
    seed_path = tmp_path / "seed.jsonl"
    seed_path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n",
        encoding="utf-8")
    return seed_path


@pytest.fixture()
def r1_file(ling_seed, tmp_path) -> Path:
    rows = [
        {"schema": sft_prep.SCHEMA_DPO, "family": "fahm",
         "probe_kind": "splice", "must": "not_found",
         "provenance": {"case_id": "ry10-fahm-001"},
         "id": "dpo-ry10-fahm-001-splice",
         "prompt": "وثّق الاقتباس الآتي:\n«س»\nالحكم:",
         "chosen": "الاقتباس مركّب — يُرفض ويُمتنع.",
         "rejected": "مُوثَّق حرفياً في «س»."},
        {"schema": sft_prep.SCHEMA_DPO, "family": "yaqin",
         "probe_kind": "over_refusal", "must": "answer",
         "provenance": {"case_id": "ry10-yaqin-001"},
         "id": "dpo-ry10-yaqin-001-over_refusal",
         "prompt": "السؤال: س؟\nالجواب:", "chosen": " جواب",
         "rejected": sft_prep.DPO_REFUSE},
    ]
    p = tmp_path / "r1.jsonl"
    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")
    return p


def _pins(seed_path: Path, r1_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sft_prep, "SEED_SHA_R2",
                        sft_prep._sha256(seed_path))
    monkeypatch.setattr(sft_prep, "ROUND1_SHA_R2",
                        sft_prep._sha256(r1_path))


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------- tests
def test_r2_counts_templates_and_distractors(
        ling_store_path, ling_seed, r1_file, tmp_path, monkeypatch):  # noqa: F401
    _pins(ling_seed, r1_file, monkeypatch)
    m = build_dpo_r2(ling_seed, tmp_path / "split", ling_store_path,
                     r1_file, tmp_path / "out")
    pairs = _rows(tmp_path / "out" / "ryko1000_dpo_train_r2.jsonl")
    sid = KnowledgeStore(str(ling_store_path), persist_vectors=False)
    titles = [b["title_ar"] for b in sid.iter_books()]
    sid.close()
    # 2 r1 rows + 15 ling probes + 1 dev must-answer pair
    assert m["files"]["train"]["n"] == len(pairs) == 18
    assert m["counts"]["new_pairs"] == 16
    kinds = {p["probe_kind"] for p in pairs}
    assert kinds == {"splice", "over_refusal", "wrong_attribution"}
    ling_wa = [p for p in pairs[2:]
               if p["probe_kind"] == "wrong_attribution"]
    ling_or = [p for p in pairs[2:]
               if p["probe_kind"] == "over_refusal"]
    assert len(ling_wa) == 15 and len(ling_or) == 1  # dev only
    assert m["counts"]["by_family"] == {
        "balagha": 5, "irab": 5, "sarf": 5, "yaqin": 1}
    # templates: MCQ-canonical prompt, chosen/rejected shapes
    for p in ling_wa + ling_or:
        assert p["prompt"].startswith("السؤال: ")
        assert p["prompt"].endswith("\nالجواب:")
        assert p["chosen"].startswith(" ")
        assert p["chosen"] != p["rejected"]
    # closed-catalog distractors, per pattern (id-keyed)
    pid_ = {p["id"]: p for p in pairs[2:]}
    assert pid_["dpo-ry10-irab-001-wrong_attribution"]["rejected"] \
        == " ابن جني"
    assert pid_["dpo-ry10-irab-002-wrong_attribution"]["rejected"] \
        == " " + titles[1]
    assert pid_["dpo-ry10-irab-002-wrong_attribution"]["chosen"] \
        == " " + titles[0]
    assert pid_["dpo-ry10-irab-003-wrong_attribution"]["rejected"] \
        == " متابعة ثانية"
    assert pid_["dpo-ry10-irab-005-wrong_attribution"]["rejected"] \
        == " منشأ ب — الأصل الثاني"
    # over-refusal counter-pair: the dev must-answer row only
    assert all(p["rejected"] == sft_prep.DPO_REFUSE for p in ling_or)
    # the dev pair carries its السند
    dev_pair = next(p for p in pairs[2:]
                    if p["provenance"].get("split") == "dev")
    assert "السند:" in dev_pair["chosen"]
    # ids unique, provenance carries case_id + ling_pattern
    ids = [p["id"] for p in pairs]
    assert len(ids) == len(set(ids))
    assert all(p["provenance"]["case_id"].startswith("ry10-")
               for p in pairs)
    assert m["leakage_check"]["leaked"] == 0


def test_r2_prefix_byte_identity(ling_store_path, ling_seed, r1_file,
                                 tmp_path, monkeypatch):  # noqa: F401
    _pins(ling_seed, r1_file, monkeypatch)
    build_dpo_r2(ling_seed, tmp_path / "split", ling_store_path,
                 r1_file, tmp_path / "out")
    r1 = r1_file.read_bytes()
    r2 = (tmp_path / "out" / "ryko1000_dpo_train_r2.jsonl").read_bytes()
    assert r2[:len(r1)] == r1


def test_r2_determinism(ling_store_path, ling_seed, r1_file,
                        tmp_path, monkeypatch):  # noqa: F401
    _pins(ling_seed, r1_file, monkeypatch)
    m1 = build_dpo_r2(ling_seed, tmp_path / "split", ling_store_path,
                      r1_file, tmp_path / "out1")
    m2 = build_dpo_r2(ling_seed, tmp_path / "split", ling_store_path,
                      r1_file, tmp_path / "out2")
    assert m1["files"]["train"]["sha256"] == m2["files"]["train"]["sha256"]


def test_r2_leakage_fatal(ling_store_path, ling_seed, r1_file,
                          tmp_path, monkeypatch):  # noqa: F401
    # a ling case_id that lands in the test bucket must abort the build
    # before any write
    rows = [_rows(ling_seed)[0]]  # ry10-irab-001 — a real new-pair source
    split = tmp_path / "split"
    (split / "ryko1000_test.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")
    _pins(ling_seed, r1_file, monkeypatch)
    with pytest.raises(ValueError, match="leakage"):
        build_dpo_r2(ling_seed, split, ling_store_path, r1_file,
                     tmp_path / "out")
    assert not (tmp_path / "out" / "ryko1000_dpo_train_r2.jsonl").exists()


def test_r2_pin_drift_fail_loud(ling_store_path, ling_seed, r1_file,
                                tmp_path, monkeypatch):  # noqa: F401
    _pins(ling_seed, r1_file, monkeypatch)
    drifted = tmp_path / "seed_drifted.jsonl"
    drifted.write_text(ling_seed.read_text(encoding="utf-8").replace(
        "سؤال لغوي", "سؤال مُعدَّل"), encoding="utf-8")
    with pytest.raises(ValueError, match="seed drift"):
        build_dpo_r2(drifted, tmp_path / "split", ling_store_path,
                     r1_file, tmp_path / "out")
    with pytest.raises(ValueError, match="round-1 pairs drift"):
        build_dpo_r2(ling_seed, tmp_path / "split", ling_store_path,
                     drifted, tmp_path / "out")
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------- real contract
requires_real_bundle = pytest.mark.skipif(
    not (R1_PATH.exists() and SEED_PATH.exists() and REAL_STORE.exists()),
    reason="real round-1 bundle / seed / committed store not present")


@pytest.mark.skipif(
    not R2_PATH.exists(),
    reason="round-2 bundle not built yet")
def test_r2_real_bundle_contract():
    """The committed round-2 bundle satisfies its own manifest and the
    frozen-prefix contract against the real round-1 file."""
    m = json.loads((ROOT / "data" / "training" / "dpo_manifest_r2.json")
                   .read_text(encoding="utf-8"))
    r1 = R1_PATH.read_bytes()
    r2b = R2_PATH.read_bytes()
    assert r2b[:len(r1)] == r1
    assert hashlib.sha256(r2b).hexdigest() == \
        m["files"]["train"]["sha256"]
    rows = _rows(R2_PATH)
    assert len(rows) == m["files"]["train"]["n"] == 1094
    assert m["counts"]["round1_pairs"] == 792
    assert m["counts"]["new_pairs"] == 302
    assert m["counts"]["by_probe_kind"] == {
        "over_refusal": 2, "wrong_attribution": 300}
    assert m["sources"]["round1"]["sha256"] == \
        hashlib.sha256(r1).hexdigest()
    assert m["leakage_check"]["leaked"] == 0


@requires_real_bundle
def test_r2_real_builder_reproduces_bundle(tmp_path):
    """Rebuilding from the REAL committed inputs reproduces the committed
    bundle byte-exactly (pins are the real constants — no monkeypatch)."""
    m = build_dpo_r2(SEED_PATH,
                     ROOT / "data" / "training" / "ryko1000_split",
                     REAL_STORE, R1_PATH, tmp_path / "out")
    assert m["files"]["train"]["sha256"] == \
        json.loads((ROOT / "data" / "training" / "dpo_manifest_r2.json")
                   .read_text(encoding="utf-8"))["files"]["train"]["sha256"]
