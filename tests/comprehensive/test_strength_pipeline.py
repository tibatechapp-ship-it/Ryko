"""Strength — pipeline integrity gates (15 tests).

The build's own honesty infrastructure, exercised end-to-end: provenance
receipts verify, manifests parse, the committed benchmark files are
well-formed, the secret gates pass, and the CI wiring actually matches
what the repo ships.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ryko_lm.provenance import verify_receipt

ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          timeout=600, check=False)


# ------------------------------------------------------------- receipts
def test_serving_checkpoint_receipt_verifies():
    ok, report = verify_receipt(ROOT / "artifacts" / "dpo_round2_small" / "final_receipt.json")
    assert ok, [r for r in report if not r["ok"]]


def test_bundle_pack_receipt_verifies():
    ok, report = verify_receipt(ROOT / "data" / "training" / "bundle_pack" / "receipt.json")
    assert ok, [r for r in report if not r["ok"]]


def test_argrel_pack_receipt_verifies():
    ok, report = verify_receipt(ROOT / "data" / "training" / "argrel_pack" / "receipt.json")
    assert ok, [r for r in report if not r["ok"]]


def test_all_receipts_audit_exit_zero():
    # the fixed repo-copy audit: untracked build artifacts are SKIP, never FAIL
    r = _run([sys.executable, "scripts/verify_all_receipts.py"])
    assert r.returncode == 0, r.stdout[-500:]
    assert "0 FAIL" in r.stdout


# ------------------------------------------------------------ manifests
@pytest.mark.parametrize("manifest", [
    "data/training/sft_manifest.json",
    "data/training/ling_sft_manifest.json",
    "data/training/dpo_manifest.json",
    "data/training/dpo_manifest_r2.json",
    "data/training/bundle_pack_manifest.json",
    "data/training/argrel_pack_manifest.json",
])
def test_training_manifests_parse(manifest):
    blob = json.loads((ROOT / manifest).read_text(encoding="utf-8"))
    assert isinstance(blob, dict) and blob


def test_palmx_bench_lines_parse():
    lines = (ROOT / "data" / "bench" / "palmx_ic.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 50
    for line in lines:
        assert "case_id" in json.loads(line)


def test_tokenizer_meta_agrees_with_artifact():
    meta = json.loads((ROOT / "artifacts" / "tokenizer" / "tokenizer_meta.json").read_text(encoding="utf-8"))
    assert meta["vocab_size"] == 32768
    assert meta["algo"] == "ByteLevelBPE"
    assert meta["special_tokens"] == ["<pad>", "<bos>", "<eos>", "<unk>"]


# ------------------------------------------------------------ secret gates
def test_local_secret_gate_clean():
    r = _run([sys.executable, "scripts/secret_scan.py", "--tree", "."])
    assert r.returncode == 0, r.stdout[-300:]
    assert "critical=0" in r.stdout


def test_gitleaks_baseline_contains_only_sha256_findings():
    baseline = json.loads((ROOT / "gitleaks-baseline.json").read_text(encoding="utf-8"))
    assert len(baseline) == 5
    for finding in baseline:
        assert finding["RuleID"] == "generic-api-key"
        # matches start after the rule keyword boundary — no leading quote
        assert finding["Match"].startswith("tokenizer_sha256")


def test_ci_workflow_valid_and_baseline_wired():
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    steps = [s for job in ci["jobs"].values() for s in job.get("steps", [])]
    run_cmds = " ".join(s.get("run", "") for s in steps)
    assert "--baseline-path gitleaks-baseline.json" in run_cmds
    assert any("pytest -q" in s.get("run", "") for s in steps)


# ------------------------------------------------------------ build glue
def test_makefile_exposes_core_targets():
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("test:", "train-smoke:", "tokenizer:", "binarize:"):
        assert target in text


def test_pyproject_version_matches_changelog():
    import tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert version in changelog


def test_docs_consistency_routing_mentions_serving_point():
    routing = (ROOT / "docs" / "ROUTING.md").read_text(encoding="utf-8")
    assert "dpo_round2_small" in routing
