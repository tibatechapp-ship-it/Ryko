# -*- coding: utf-8 -*-
"""argrel proof honesty analysis: which rb-rel bench cases target relations
seen in the proof pack's TRAIN split (memorization zone) vs VAL-held-out
(generalization zone) vs untouched by the pack. Compact output only."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = Path(__file__).resolve().parents[1]


def pair_relid(p):
    prov = p.get("provenance")
    if isinstance(prov, dict):
        for k in ("relation_id", "relation", "id"):
            if k in prov:
                return prov[k]
    pid = str(p.get("id", ""))
    if "|" in pid:
        return pid.split("|")[0]
    return pid


def main():
    train_rel, val_rel = set(), set()
    for path, bucket in (("data/training/argrel_sft_train.jsonl", train_rel),
                         ("data/training/argrel_sft_val.jsonl", val_rel)):
        for line in open(REPO / path, encoding="utf-8"):
            bucket.add(str(pair_relid(json.loads(line))))
    print(f"pack: train_relations={len(train_rel)} val_relations={len(val_rel)}")

    cases = [json.loads(l) for l in open(REPO / "docs" / "benchmarks" / "ryko_bench_seed.jsonl", encoding="utf-8")]
    arg = [c for c in cases if c["kind"] == "mcq" and str(c.get("case_id", "")).startswith("rb-rel-")]
    zones = {"train": [], "val": [], "absent": []}
    for c in arg:
        rid = str(c["meta"]["relation_id"])
        if rid in train_rel:
            zones["train"].append(c)
        elif rid in val_rel:
            zones["val"].append(c)
        else:
            zones["absent"].append(c)
    print(f"rb-rel zones: train={len(zones['train'])} val={len(zones['val'])} absent={len(zones['absent'])}")

    from ryko_lm.eval.benchmarks import load_benchmark, run_mcq
    from ryko_lm.model.ryko_model import RykoLM
    from ryko_lm.tokenizer.loader import RykoTokenizer

    model = RykoLM.load(REPO / "artifacts" / "proof_argrel_small" / "final.pt", device="cpu")
    tok = RykoTokenizer.from_file(REPO / "artifacts" / "tokenizer" / "ryko-tokenizer.json")
    case_objs = [c for c in load_benchmark(REPO / "docs" / "benchmarks" / "ryko_bench_seed.jsonl")
                 if c.case_id.startswith("rb-rel-")]
    rows = run_mcq(model, tok, case_objs, "cpu")
    by_case = {r["case_id"]: r for r in rows}
    for zone, cs in zones.items():
        if not cs:
            print(f"zone {zone}: n=0")
            continue
        correct = sum(1 for c in cs if by_case[c["case_id"]]["correct"])
        ids = [c["case_id"] for c in cs if not by_case[c["case_id"]]["correct"]]
        print(f"zone {zone}: n={len(cs)} acc={correct/len(cs):.4f} wrong={ids}")


if __name__ == "__main__":
    main()
