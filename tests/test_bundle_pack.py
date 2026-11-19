"""Bundle v1.26.0 asset contracts — determinism, receipts, probe-free data.

Covers the combined-treatment assets added in v1.26.0 (linguistics SFT pack,
bundle pack, bundle DPO preference pairs): the same discipline the argrel
assets got in v1.25.0 — byte-rebuildable determinism, fatal probe-leakage
gate re-checked at test time, and manifest/sha256 coherence.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "training"
PROBES = ROOT / "docs" / "benchmarks" / "ling_probes.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm(text: str) -> str:
    return re.sub(r"[\s\u064B-\u0652\u0670«»(){}.,،؛:!?\-—]+", "", text)


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _probe_exclusions() -> dict:
    claim_ids, book_ids, passage_ids, quotes = set(), set(), set(), set()
    for case in _load(PROBES):
        meta = case.get("meta", {})
        claim_ids.add(meta.get("claim_id"))
        book_ids.add(meta.get("book_id"))
        passage_ids.add(meta.get("passage_id"))
        for quoted in re.findall(r"«(.+?)»", case.get("question", ""), re.S):
            quotes.add(_norm(quoted))
    return {"claim_ids": {x for x in claim_ids if x},
            "book_ids": {x for x in book_ids if x},
            "passage_ids": {x for x in passage_ids if x},
            "quotes": quotes,
            "title_norms": {q for q in quotes if len(q) < 25}}


def _assert_probe_free(rows: list[dict]) -> None:
    excl = _probe_exclusions()
    for row in rows:
        prompt = row.get("prompt", "")
        np_ = _norm(prompt)
        quoted = [_norm(q) for q in re.findall(r"«(.+?)»", prompt, re.S)]
        for q in excl["quotes"]:
            if not q:
                continue
            if len(q) >= 25:
                assert q not in np_, f"probe quote leak in {row.get('id')}"
            else:
                assert not any(q == nq for nq in quoted), \
                    f"probe title leak in {row.get('id')}"
        prov = json.dumps(row.get("provenance", {}), ensure_ascii=False)
        for key in ("claim_ids", "book_ids", "passage_ids"):
            for pid in excl[key]:
                assert pid not in prov, f"probe id leak in {row.get('id')}"


def test_ling_sft_determinism_and_probe_free():
    """Rebuilding the ling SFT pack must be byte-identical and probe-free."""
    train = _load(TRAIN / "ling_sft_train.jsonl")
    val = _load(TRAIN / "ling_sft_val.jsonl")
    manifest = json.loads((TRAIN / "ling_sft_manifest.json").read_text(
        encoding="utf-8"))
    assert manifest["rows"]["train"] == len(train) == 102
    assert manifest["rows"]["val"] == len(val) == 14
    assert manifest["probe_exclusions"]["books_excluded"] == 33
    assert manifest["probe_exclusions"]["claims_excluded"] == 16
    assert (manifest["sources"]["probes_sha256"]
            == _sha256(PROBES))
    _assert_probe_free(train + val)
    # Byte-rebuild determinism via the committed builder.
    out = ROOT / "artifacts" / "smoke" / "ling_sft_rebuild"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "scripts/build_ling_sft.py", "--out-dir", str(out)]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True,
                   env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"
                        + ":/usr/local/bin"})
    assert _sha256(out / "ling_sft_train.jsonl") \
        == _sha256(TRAIN / "ling_sft_train.jsonl")
    assert _sha256(out / "ling_sft_val.jsonl") \
        == _sha256(TRAIN / "ling_sft_val.jsonl")


def test_bundle_pack_manifest_and_leakage():
    """The bundle pack manifest must match the committed rows; rows must be
    probe-free and cover all four documented components."""
    manifest = json.loads((TRAIN / "bundle_pack_manifest.json").read_text(
        encoding="utf-8"))
    train = _load(TRAIN / "bundle_sft_train.jsonl")
    val = _load(TRAIN / "bundle_sft_val.jsonl")
    assert manifest["rows_train"] == len(train) == 740
    assert manifest["rows_val"] == len(val) == 29
    for key in ("train", "val"):
        assert manifest["files"][key]["sha256"] == _sha256(
            TRAIN / ("bundle_sft_train.jsonl" if key == "train"
                     else "bundle_sft_val.jsonl"))
    comps = manifest["components"]
    assert comps["argrel"]["train"] == 90
    assert comps["ling-sft"]["train"] == 102
    assert comps["weakest-families"]["train"] == 255
    assert comps["ling-replay"]["train"] == 293
    assert comps["ling-replay"]["excluded_probe_overlap"] == 7
    assert manifest["leakage_check"]["probe_leaks"] == 0
    _assert_probe_free(train + val)
    pm = json.loads((TRAIN / "bundle_pack" / "pack_manifest.json").read_text(
        encoding="utf-8"))
    assert pm["files"]["train"]["tokens"] == 55695
    receipt = json.loads((TRAIN / "bundle_pack" / "receipt.json").read_text(
        encoding="utf-8"))
    assert receipt.get("kind") == "ryko-bundle-pack-receipt"


def test_bundle_dpo_pairs_probe_free_and_wellformed():
    """DPO preference files: probe-free, chosen != rejected, argrel prefs
    swap only the leading relation verb."""
    for name in ("bundle_dpo_pairs_ling.jsonl",
                 "bundle_dpo_pairs_mixed.jsonl"):
        rows = _load(TRAIN / name)
        assert rows, name
        _assert_probe_free(rows)
        for row in rows:
            assert row["chosen"] != row["rejected"], row["id"]
            assert row["prompt"].strip(), row["id"]
    dman = json.loads((TRAIN / "bundle_dpo_manifest.json").read_text(
        encoding="utf-8"))
    assert dman["files"]["mixed"]["n"] == 1393
    assert dman["files"]["ling"]["n"] == 1288
    assert dman["kept_r2"]["excluded_probe_overlap"] == 7
    assert dman["leakage_check"]["probe_leaks"] == 0
    # Argrel defensive prefs: rejected differs only in the leading verb
    # (verbs may be multi-word — match the closed vocabulary by prefix).
    verbs = ("يدعم", "يعترض", "يرد على", "ينقل عن", "يرجّح",
             "يلخّص", "يعيد الصياغة", "يناقض")

    def _split_verb(text: str) -> tuple[str, str]:
        t = text.strip()
        for v in sorted(verbs, key=len, reverse=True):
            if t.startswith(v):
                return v, t[len(v):]
        raise AssertionError(f"no closed-vocabulary verb prefix in: {t[:40]}")

    for row in _load(TRAIN / "bundle_dpo_pairs_mixed.jsonl"):
        if not row["id"].startswith("pref-argrel-"):
            continue
        gold_v, gold_tail = _split_verb(row["chosen"])
        wrong_v, wrong_tail = _split_verb(row["rejected"])
        assert gold_v != wrong_v, row["id"]
        assert gold_tail == wrong_tail, row["id"]


def test_bundle_experiments_ledger_index():
    """Every recorded experiment file is present and hash-listed."""
    index = json.loads((ROOT / "docs" / "results" / "bundle_experiments"
                        / "index.json").read_text(encoding="utf-8"))
    for name, sha in index["sha256"].items():
        path = ROOT / "docs" / "results" / "bundle_experiments" / name
        assert path.exists(), name
        assert _sha256(path) == sha, name
    ry10 = json.loads((ROOT / "docs" / "results" / "bundle_experiments"
                       / "bundle_sft_r2_480_ryko1000_parts.json").read_text(
                           encoding="utf-8"))
    rates = {k: v["rate"] for k, v in ry10["per_family"].items()}
    assert rates["istidlal"] == 0.60 and rates["mustalahat"] == 0.51 \
        and rates["tarjih"] == 0.44
