"""Deduce the stage-A val_ratio used by the original v5 build.

Facts used:
- stage A cleaned shards 0-1 into 1,782 docs (verified dup count) and split
  with random.Random(1337).shuffle over the tmp doc order;
- the current val.txt holds the 17 stage-A val docs (ratio 0.01) FIRST (in tmp
  order), then the augment-diverted docs (ratio 0.004 buckets);
- binarize token accounting is additive per document (BOS + lines + EOS).

For any ratio r <= 0.01 the candidate val docs are the first int(1782*r)
entries of the permutation — a subset of the current 17 val docs. So encoding
every doc once and summing subsets reproduces the exact binarize accounting.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ryko_lm.data.binarize import _encode_file  # noqa: E402
from ryko_lm.tokenizer.loader import RykoTokenizer  # noqa: E402


def fp_int(line: str) -> int:
    folded = re.sub(r"[^\u0600-\u06FFA-Za-z]", "", line)
    return int.from_bytes(hashlib.sha256(folded.encode("utf-8")).digest()[:8], "big")


def read_docs(path: Path) -> list[str]:
    docs: list[str] = []
    cur: list[str] = []
    with path.open(encoding="utf-8") as src:
        for line in src:
            if line.strip():
                cur.append(line.rstrip("\n"))
            elif cur:
                docs.append("\n".join(cur))
                cur = []
    if cur:
        docs.append("\n".join(cur))
    return docs


def main() -> int:
    val_path = Path("data/processed_v5/val.txt")
    tok_path = Path("artifacts/tokenizer/ryko-tokenizer.json")
    tok = RykoTokenizer.from_file(tok_path)

    docs = read_docs(val_path)
    print(f"val.txt: {len(docs)} docs total")
    stage_a = docs[:17]
    diverted = docs[17:]

    # verify the diverted docs match the 0.004 bucket (doc_seed % 250 == 0)
    bad = [i for i, d in enumerate(diverted) if fp_int(d.split("\n")[0]) % 250 != 0]
    print(f"diverted docs: {len(diverted)}, not-in-250-bucket: {len(bad)}")

    # per-doc token counts (exact binarize accounting)
    tmp = Path("/tmp/val_doc_probe.txt")
    counts: list[int] = []
    for i, d in enumerate(stage_a + diverted):
        tmp.write_text(d + "\n\n", encoding="utf-8")
        out = Path("/tmp/val_doc_probe.bin")
        n = _encode_file(tok, tmp, out)
        counts.append(n)
    counts_a, counts_d = counts[:17], counts[17:]
    print(f"stage-A val tokens: {sum(counts_a):,}; diverted tokens: {sum(counts_d):,}")

    # permutation of the 1,782 stage-A docs
    P = list(range(1782))
    random.Random(1337).shuffle(P)
    val_ids_01 = sorted(P[:17])  # tmp positions of my 17 val docs, in tmp order

    target = 3_683_960
    div_tokens = sum(counts_d)
    print(f"\nsearching n (stage-A val docs) for total {target:,} "
          f"(with diverted {div_tokens:,}):")
    found = []
    for n in range(1, 18):
        cand_ids = sorted(P[:n])
        # map tmp position -> my doc index
        my_idx = [val_ids_01.index(p) for p in cand_ids]
        total = sum(counts_a[i] for i in my_idx) + div_tokens
        mark = ""
        if total == target:
            found.append(n); mark = "  <=== EXACT MATCH"
        print(f"  n={n:2d}: {total:,} (delta {total - target:+,}){mark}")
    print(f"\nexact matches: {found or 'NONE'}")
    # also: stage-A only (no diverted)
    for n in range(1, 18):
        cand_ids = sorted(P[:n])
        my_idx = [val_ids_01.index(p) for p in cand_ids]
        total = sum(counts_a[i] for i in my_idx)
        if total == target:
            print(f"stage-A-only exact match at n={n}")
    tmp.unlink(missing_ok=True)
    Path("/tmp/val_doc_probe.bin").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
