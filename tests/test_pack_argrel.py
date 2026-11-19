"""pack_argrel tests — determinism, per-relation split, FATAL leakage."""

from __future__ import annotations

import json

import pytest

from ryko_lm.knowledge.relations import RELATION_AR
from ryko_lm.knowledge.store import KnowledgeStore
from ryko_lm.training.pack_argrel import (
    SCHEMA,
    build,
    leakage_check,
    pair_variants,
    split_relations,
)


# ---------------------------------------------------------------- fixtures
_CLAIM_TEXTS = {
    "c1": "القول الأول للاختبار الممتد بما يكفي لعتبة العشرين حرفاً",
    "c2": "القول الثاني للاختبار الممتد بما يكفي لعتبة العشرين حرفاً",
    "c3": "القول الثالث للاختبار الممتد بما يكفي لعتبة العشرين حرفاً",
    "c4": "القول الرابع للاختبار الممتد بما يكفي لعتبة العشرين حرفاً",
    "c5": "القول الخامس للاختبار الممتد بما يكفي لعتبة العشرين حرفاً",
    "c6": "القول السادس للاختبار الممتد بما يكفي لعتبة العشرين حرفاً",
}
_REL_SPEC = [
    ("c1", "c2", "supports"), ("c2", "c3", "summarizes"),
    ("c3", "c4", "supports"), ("c4", "c5", "responds"),
    ("c5", "c6", "cites"), ("c1", "c6", "paraphrases"),
]


def _mini_store_path(tmp_path) -> str:
    """A tiny file-backed store with 6 documented relations."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "mini.sqlite"
    store = KnowledgeStore(str(path), persist_vectors=False)
    ids = {k: store.add_claim(t, meta={"layer": "fikr",
                                       "originator": "منشئ الاختبار"})
           for k, t in _CLAIM_TEXTS.items()}
    for i, (a, b, rel) in enumerate(_REL_SPEC):
        store.add_relation(ids[a], ids[b], rel,
                           evidence=f"سند الاختبار رقم {i + 1}.",
                           meta={"layer": "fikr",
                                 "src_originator": "منشئ الاختبار الأول",
                                 "dst_originator": "منشئ الاختبار الثاني"})
    store.close()
    return str(path)


# ----------------------------------------------------------- pair variants
def test_pair_variants_fixed_order_and_bench_gate_contract() -> None:
    rel = {
        "relation_id": "r-test00001", "relation": "supports", "layer": "fikr",
        "src": {"claim_id": "c-a", "text": "قول المصدر الممتد بما يكفي هنا",
                "originator": "الأصل الموثق"},
        "dst": {"claim_id": "c-b", "text": "قول الهدف الممتد بما يكفي هنا",
                "originator": "الهدف الموثق"},
        "evidence": "سند موثق.",
    }
    rows, skipped = pair_variants(rel)
    assert skipped == []
    assert [r["id"].rsplit("-", 1)[1] for r in rows] == ["v1", "v2", "v3"]
    # v1 IS the gate conditional: completion = " " + RELATION_AR verb
    assert rows[0]["completion"] == " " + RELATION_AR["supports"]
    assert rows[0]["prompt"].startswith("السؤال: القول الأول: «")
    assert rows[0]["prompt"].endswith(
        "ما علاقة القول الأول بالثاني كما هي موثقة في مزرعة العلاقات؟\nالجواب:")
    # v2 names both documented sides + the evidence line
    assert "الأصل: الأصل الموثق" in rows[1]["completion"]
    assert "الهدف: الهدف الموثق" in rows[1]["completion"]
    assert "السند: سند موثق." in rows[1]["completion"]


def test_pair_variants_skips_degenerate_loudly() -> None:
    rel = {"relation_id": "r-test00002", "relation": "supports", "layer": "fikr",
           "src": {"claim_id": "c-a", "text": "قصير", "originator": ""},
           "dst": {"claim_id": "c-b",
                   "text": "القول الثاني الممتد بما يكفي لعتبة العشرين حرفاً",
                   "originator": ""},
           "evidence": ""}
    rows, skipped = pair_variants(rel)
    assert rows == []
    assert skipped == ["v1_bench_gate", "v2_open_grounding", "v3_classify"]


# ------------------------------------------------------------------- split
def test_split_relations_deterministic_disjoint_full_cover() -> None:
    ids = [f"r-{i:010d}" for i in range(35)]
    tr1, va1 = split_relations(ids, split_seed=1337, train_share=0.85)
    tr2, va2 = split_relations(ids, split_seed=1337, train_share=0.85)
    assert (tr1, va1) == (tr2, va2)
    assert not (set(tr1) & set(va1))          # never straddles
    assert sorted(tr1 + va1) == sorted(set(ids))
    assert len(va1) == 5 and len(tr1) == 30   # 35 * 0.85


# ----------------------------------------------------------------- leakage
def test_leakage_check_is_fatal() -> None:
    row = {"id": "argrel-r-x-v1", "prompt": "س؟", "completion": " ج",
           "provenance": {"relation_id": "r-x"}}
    with pytest.raises(ValueError):
        leakage_check([row], [dict(row)])
    # a shared rendered prompt across splits is equally fatal
    a = {**row, "id": "argrel-r-x-v1", "provenance": {"relation_id": "r-x"}}
    b = {**row, "id": "argrel-r-y-v1", "provenance": {"relation_id": "r-y"}}
    with pytest.raises(ValueError):
        leakage_check([a], [b])


# ------------------------------------------------------------------- build
def test_build_deterministic_and_relation_disjoint(tmp_path) -> None:
    path1, path2 = _mini_store_path(tmp_path), _mini_store_path(tmp_path / "b")
    m1 = build(path1, tmp_path / "out1")
    m2 = build(path2, tmp_path / "out2")
    assert m1["schema"] == SCHEMA
    assert m1["relations_total"] == 6
    assert (m1["relations_train"], m1["relations_val"]) == (5, 1)
    assert (m1["pairs_train"], m1["pairs_val"]) == (15, 3)
    for side in ("train", "val"):
        assert m1["files"][side]["sha256"] == m2["files"][side]["sha256"]
    assert m1["leakage_check"]["leaked_relations"] == 0
    assert m1["leakage_check"]["leaked_pair_ids"] == 0
    assert m1["leakage_check"]["leaked_prompts"] == 0
    assert m1["leakage_check"]["fatal"] is True
    by_side: dict[str, set] = {}
    for side in ("train", "val"):
        path = tmp_path / "out1" / f"argrel_sft_{side}.jsonl"
        recs = [json.loads(l) for l in
                path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert all(r["split"] == side for r in recs)
        by_side[side] = {r["provenance"]["relation_id"] for r in recs}
    assert not (by_side["train"] & by_side["val"]), \
        "a relation's phrasings must never straddle the splits"
