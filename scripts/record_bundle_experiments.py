#!/usr/bin/env python3
"""Record the v1.26.0 combined-treatment experiments (دفتر المعالجات).

Copies the measured result JSONs of the seven controlled bundle experiments
into docs/results/bundle_experiments/ with a stable, documented naming —
the measurement ledger behind docs/results/bundle_treatment.md.

Run from the repo root:
    python scripts/record_bundle_experiments.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "bundle_experiments"

# experiment id -> (source json, docstring)
EXPERIMENTS = {
    # 1 — bundle SFT round-1: 82 steps (~3 epochs), light dose
    "bundle_sft_r1_82": "/tmp/bench_bundle.json",
    "bundle_sft_r1_82_ling": "/tmp/ling_bundle.json",
    # 2 — bundle SFT round-2: 480 steps (~17.7 epochs), full dose
    "bundle_sft_r2_480": "/tmp/bench_r2.json",
    "bundle_sft_r2_480_ling": "/tmp/ling_r2.json",
    "bundle_sft_r2_480_ryko1000_parts": None,  # assembled from slice runs
    # 3 — full committed r2 DPO replay on top of bundle SFT
    "dpo_r2_replay": "/tmp/bench_dpo.json",
    "dpo_r2_replay_ling": "/tmp/ling_dpo.json",
    # 4 — ling-only DPO on top of the committed argrel shard
    "dpo_ling_on_argrel": "/tmp/bench_planC.json",
    "dpo_ling_on_argrel_ling": "/tmp/ling_planC.json",
    # 5 — mixed DPO (argrel prefs defense) on top of bundle SFT
    "dpo_mixed_on_bundle": "/tmp/bench_planD.json",
    "dpo_mixed_on_bundle_ling": "/tmp/ling_planD.json",
    # 6 — full-mixed DPO (r2 full + ling x3 + argrel prefs) on argrel shard
    "dpo_e_on_argrel": "/tmp/bench_planE.json",
    "dpo_e_on_argrel_ling": "/tmp/ling_planE.json",
    # 7 — WiSE-FT soup alpha=0.5 (serving <-> argrel shard)
    "soup_a05": "/tmp/soup_bench.json",
    "soup_a05_ling": "/tmp/soup_ling.json",
}

RY10_SLICES = {
    "mustalahat": "/tmp/ry10_b_a.json",
    "istidlal": "/tmp/ry10_b_a.json",
    "tarjih": "/tmp/ry10_b_e.json",
}

# Reference numbers measured on the committed points (quoted verbatim from
# the committed result docs; not re-measured here).
REFERENCES = {
    "serving_dpo_round2": {"mcq": 39.62, "hajaj": 3.3, "rb_ling": 44.23,
                           "correct_abstain": 1.0, "wrong_abstain": 0.5625,
                           "cite": 1.0, "palmx": 29.05,
                           "doc": "docs/results/benchmarks_round2.md + "
                                  "ling_probes_round2.json + palmx_ic_v4.md"},
    "argrel_shard_v125": {"mcq": 55.35, "hajaj": 76.67, "rb_ling": 30.77,
                          "correct_abstain": 1.0, "wrong_abstain": 0.4375,
                          "cite": 1.0, "palmx": 28.24,
                          "doc": "docs/results/argrel_proof.md + "
                                 "argrel_proof_ling.json + argrel_proof_palmx.json"},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for name, src in EXPERIMENTS.items():
        if src is None:
            continue
        dst = OUT / f"{name}.json"
        shutil.copyfile(Path(src), dst)
        written.append(dst.name)

    # Assemble the RYKO-1000 weakest-family slice record.
    ry10 = {"schema": "ryko-bundle-experiment/1",
            "experiment": "bundle_sft_r2_480_ryko1000_parts",
            "mode": "base", "checkpoint": "artifacts/proof_bundle_small/final.pt",
            "note": "worklist slices (skip-engine) — istidlal/mustalahat at "
                    "100:300, tarjih at 500:600 (worklist order differs from "
                    "the raw seed order)",
            "per_family": {}}
    for fam, src in RY10_SLICES.items():
        d = json.load(open(src))
        mcq = d["base"]["mcq"]
        rate = mcq["per_family"][fam]["rate"]
        ry10["per_family"][fam] = {"n": mcq["per_family"][fam]["n"],
                                   "rate": rate}
    (OUT / "bundle_sft_r2_480_ryko1000_parts.json").write_text(
        json.dumps(ry10, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    written.append("bundle_sft_r2_480_ryko1000_parts.json")

    (OUT / "references.json").write_text(
        json.dumps(REFERENCES, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    written.append("references.json")

    index = {"schema": "ryko-bundle-experiments-index/1",
             "files": written,
             "sha256": {w: _sha256(OUT / w) for w in sorted(written)}}
    (OUT / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[experiments] wrote {len(written) + 1} files -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
