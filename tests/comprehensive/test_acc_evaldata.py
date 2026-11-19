"""Accuracy — RYKO-1000 split integrity (16 tests).

The evaluation corpus is the honesty benchmark of the campaign: the
train/dev/test split must be disjoint, complete, and structurally in sync
with its manifest — no leakage, no missing cases, no silent truncation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "data" / "training" / "ryko1000_split"

REQUIRED_KEYS = {"case_id", "family", "kind", "domain", "expect", "context",
                 "answer", "expected_answer", "acceptance"}


def _load(name: str) -> list[dict]:
    return [json.loads(line) for line in (SPLIT / name).read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def splits():
    return {
        "train": _load("ryko1000_train.jsonl"),
        "dev": _load("ryko1000_dev.jsonl"),
        "test": _load("ryko1000_test.jsonl"),
    }


@pytest.fixture(scope="module")
def manifest():
    return json.loads((SPLIT / "split_manifest.json").read_text(encoding="utf-8"))


def test_split_sizes_1000(splits):
    assert sum(len(v) for v in splits.values()) == 1000
    assert len(splits["train"]) == 850 and len(splits["dev"]) == 70 and len(splits["test"]) == 80


def test_case_ids_globally_unique(splits):
    ids = [c["case_id"] for v in splits.values() for c in v]
    assert len(ids) == 1000 and len(set(ids)) == 1000


def test_no_train_dev_leakage(splits):
    dev_ids = {c["case_id"] for c in splits["dev"]}
    train_ids = {c["case_id"] for c in splits["train"]}
    assert not (dev_ids & train_ids)


def test_no_train_test_leakage(splits):
    test_ids = {c["case_id"] for c in splits["test"]}
    train_ids = {c["case_id"] for c in splits["train"]}
    assert not (test_ids & train_ids)


def test_no_dev_test_leakage(splits):
    dev_ids = {c["case_id"] for c in splits["dev"]}
    test_ids = {c["case_id"] for c in splits["test"]}
    assert not (dev_ids & test_ids)


def test_manifest_proportions(splits, manifest):
    proportions = manifest["proportions"]
    assert proportions == [85, 7, 8]
    total = sum(len(v) for v in splits.values())
    for prop, key in zip(proportions, ("train", "dev", "test")):
        expected = round(total * prop / 100)
        assert abs(len(splits[key]) - expected) <= 1


def test_manifest_per_family_matches(splits, manifest):
    families = {c["family"] for v in splits.values() for c in v}
    assert set(manifest["per_family"]) == families
    for fam, counts in manifest["per_family"].items():
        for key in ("train", "dev", "test"):
            actual = sum(1 for c in splits[key] if c["family"] == fam)
            assert actual == counts[key], (fam, key)


def test_every_case_has_required_keys(splits):
    for v in splits.values():
        for c in v:
            missing = REQUIRED_KEYS - set(c)
            assert not missing, (c.get("case_id"), missing)


def test_expect_in_closed_vocabulary(splits):
    valid = {"verified", "not_found", None}  # None = open-ended generation
    for v in splits.values():
        for c in v:
            assert c["expect"] in valid, (c["case_id"], c["expect"])


def test_acceptance_criterion_parseable(splits):
    for v in splits.values():
        for c in v:
            acc = c["acceptance"]
            assert isinstance(acc, dict) and "criterion" in acc
            assert isinstance(acc["criterion"], str) and acc["criterion"]


def test_domain_and_expected_answer_nonempty(splits):
    for v in splits.values():
        for c in v:
            assert c["domain"].strip(), c["case_id"]
            assert c["expected_answer"].strip(), c["case_id"]


def test_failure_probes_shape(splits):
    for v in splits.values():
        for c in v:
            for probe in c.get("failure_probes", []):
                assert "kind" in probe and "probe" in probe and "must" in probe


def test_families_cover_all_manifest_entries(splits, manifest):
    train_fams = {c["family"] for c in splits["train"]}
    assert set(manifest["per_family"]) <= train_fams
    assert len(set(manifest["per_family"])) == 10


def test_split_seed_recorded(manifest):
    assert isinstance(manifest["split_seed"], int) and manifest["split_seed"] > 0


def test_schema_version_recorded(manifest):
    assert manifest["schema"].startswith("ryko1000-split/")
