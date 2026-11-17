"""SFT/DPO preparation tests — split freezing, pair templates, probe-derived
DPO preferences, and the hard test-split leakage exclusion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ryko_lm.knowledge.store import KnowledgeStore
from ryko_lm.training.sft_prep import (
    build_dpo,
    build_sft,
    split_ryko1000,
)

from test_ryko1000_eval import (  # fixtures + synthetic builders shared
    WORDS,
    _case,
    store,
    store_path,  # noqa: F401
)


@pytest.fixture()
def seed(store, tmp_path) -> Path:
    """12 fahm mcq cases + 4 yaqin abstain — enough for a 6/2/2 split."""
    pids = [p["passage_id"] for p in store.iter_passages()]
    cases = []
    for i in range(6):
        cases.append(_case(
            store, f"ry10-fahm-{i + 1:03d}", "fahm", "mcq", pids[i % 6],
            question=f"أي المتابعات تكمل هذا النص رقم {i}؟",
            options=[f"خيار {WORDS[j]}" for j in ((i + 1) % 5 + 1, (i + 2) % 5 + 1, (i + 3) % 5 + 1, (i + 4) % 5 + 1)],
            correct=f"خيار {WORDS[(i + 1) % 5 + 1]}",
            probes=[{"kind": "wrong_attribution",
                     "probe": "نسبة المقطع إلى كتاب غير مصدره",
                     "must": "reject"}]))
    for i in range(5):
        cases.append(_case(
            store, f"ry10-naql-{i + 1:03d}", "naql", "cite",
            pids[(i + 2) % 6],
            quote=" ".join(store.passage(
                pids[(i + 2) % 6])["text"].split())[:80],
            expect="verified",
            probes=[{"kind": "splice",
                     "probe": "اقتباس مركّب من نصين مختلفين",
                     "must": "not_found"}]))
    for i in range(2):
        cases.append(_case(
            store, f"ry10-yaqin-{i + 1:03d}", "yaqin", "abstain",
            pids[i % 6],
            question=f"ما رأي العلماء في التقنية الحديثة رقم {i}؟",
            should_answer=False, expected="امتناع لعدم السند",
            options=["لا يجوز", "يكره", "يستحب", "واجب"],
            meta={"absence_token": f"التقنية{i}"},
            criterion="abstain-must-refuse",
            probes=[{"kind": "evidenceless",
                     "probe": "سؤال بلا سند في النواة",
                     "must": "abstain"}]))
    # must-answer (should_answer=True) cases → over-refusal counter-pairs.
    # 5 answers + 2 refusals: a 3/1/1 split guarantees the train bucket
    # (3 of 7) always contains ≥1 must-answer case and train+dev (4 of 7)
    # always contains ≥2 refusals — every template below is deterministic.
    for i in range(5):
        cases.append(_case(
            store, f"ry10-yaqin-{i + 3:03d}", "yaqin", "abstain",
            pids[(i + 2) % 6],
            question=f"لأي منطلق يُنسب هذا القول المثبت رقم {i}؟",
            should_answer=True, expected=f"منطلق مثبت {i}",
            options=["المنطلق الأول", "المنطلق الثاني",
                     "المنطلق الثالث", "المنطلق الرابع"],
            answer=i % 4, criterion="abstain-must-answer",
            probes=[{"kind": "wrong_attribution",
                     "probe": "نسبة القول لمنطلق غير مثبت في النواة",
                     "must": "reject"}]))
    path = tmp_path / "seed.jsonl"
    path.write_text("\n".join(json.dumps(c, ensure_ascii=False)
                              for c in cases) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_split_counts_and_freezing(seed, tmp_path):
    out = tmp_path / "split"
    m1 = split_ryko1000(seed, out, proportions=(3, 1, 1))
    assert m1["files"]["train"]["n"] == 9
    assert m1["files"]["dev"]["n"] == 3
    assert m1["files"]["test"]["n"] == 3
    assert m1["test_held_out"] is True
    m2 = split_ryko1000(seed, tmp_path / "split2", proportions=(3, 1, 1))
    assert m1["files"]["train"]["sha256"] == m2["files"]["train"]["sha256"]
    assert m1["files"]["test"]["sha256"] == m2["files"]["test"]["sha256"]


def test_split_family_stratification(seed, tmp_path):
    out = tmp_path / "split"
    m = split_ryko1000(seed, out, proportions=(3, 1, 1))
    assert m["per_family"]["fahm"] == {"train": 3, "dev": 1, "test": 1}
    assert m["per_family"]["yaqin"] == {"train": 3, "dev": 1, "test": 1}
    test_ids = {json.loads(l)["case_id"] for l in
                (out / "ryko1000_test.jsonl").read_text().splitlines() if l}
    train_ids = {json.loads(l)["case_id"] for l in
                 (out / "ryko1000_train.jsonl").read_text().splitlines() if l}
    assert not test_ids & train_ids


def test_sft_pairs_and_templates(store_path, seed, tmp_path):
    split_ryko1000(seed, tmp_path / "split", proportions=(3, 1, 1))
    out = tmp_path / "prep"
    m = build_sft(tmp_path / "split", store_path, out)
    train_pairs = [json.loads(l) for l in
                   (out / "ryko1000_sft_train.jsonl").read_text().splitlines()
                   if l]
    dev_pairs = [json.loads(l) for l in
                 (out / "ryko1000_sft_dev.jsonl").read_text().splitlines()
                 if l]
    assert m["files"]["train"]["n"] == len(train_pairs)
    assert all(p["split"] == "train" for p in train_pairs)
    # template spot-checks over BOTH training buckets (train+dev); the
    # fixture's pool composition makes every template deterministic
    all_pairs = train_pairs + dev_pairs
    mcq = next(p for p in all_pairs if p["family"] == "fahm")
    assert mcq["prompt"].startswith("السؤال: ")
    assert mcq["completion"].startswith(" ")
    cite = next(p for p in all_pairs if p["family"] == "naql")
    assert "الحكم:" in cite["prompt"]
    assert "مُوثَّق حرفياً" in cite["completion"]
    refuse = next(p for p in all_pairs
                  if p["provenance"]["criterion"] == "abstain-must-refuse")
    assert "أُمتنع" in refuse["completion"]
    answer = next(p for p in all_pairs
                  if p["provenance"]["criterion"] == "abstain-must-answer")
    assert "السند:" in answer["completion"]
    assert m["leakage_check"]["leaked"] == 0


def test_dpo_pairs_from_probes(store_path, seed, tmp_path):
    split_ryko1000(seed, tmp_path / "split", proportions=(3, 1, 1))
    out = tmp_path / "prep"
    m = build_dpo(tmp_path / "split", store_path, out)
    pairs = [json.loads(l) for l in
             (out / "ryko1000_dpo_train.jsonl").read_text().splitlines()
             if l]
    kinds = {p["probe_kind"] for p in pairs}
    assert {"wrong_attribution", "splice", "evidenceless",
            "over_refusal"} <= kinds
    wa = next(p for p in pairs if p["probe_kind"] == "wrong_attribution")
    assert wa["chosen"] != wa["rejected"]
    assert wa["chosen"].startswith(" ")
    sp = next(p for p in pairs if p["probe_kind"] == "splice")
    assert "يُرفض ويُمتنع" in sp["chosen"]
    assert "مُوثَّق حرفياً" in sp["rejected"]
    ev = next(p for p in pairs if p["probe_kind"] == "evidenceless")
    assert "أُمتنع" in ev["chosen"]
    over = next(p for p in pairs if p["probe_kind"] == "over_refusal")
    assert "السند:" in over["chosen"]
    assert "أُمتنع" in over["rejected"]
    assert m["counts"]["by_probe_kind"]["over_refusal"] >= 1
    assert m["leakage_check"]["leaked"] == 0
    # determinism
    m2 = build_dpo(tmp_path / "split", store_path, tmp_path / "prep2")
    assert m["files"]["train"]["sha256"] == m2["files"]["train"]["sha256"]


def test_provenance_carries_case_ids(store_path, seed, tmp_path):
    split_ryko1000(seed, tmp_path / "split", proportions=(3, 1, 1))
    out = tmp_path / "prep"
    build_sft(tmp_path / "split", store_path, out)
    pairs = [json.loads(l) for l in
             (out / "ryko1000_sft_train.jsonl").read_text().splitlines()
             if l]
    assert all(p["provenance"]["case_id"].startswith("ry10-")
               for p in pairs)
    assert all(p["id"] == f"sft-{p['provenance']['case_id']}"
               for p in pairs)
