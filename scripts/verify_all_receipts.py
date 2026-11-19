#!/usr/bin/env python3
"""Empirical audit: verify every *receipt*.json under Ryko artifacts/ and data/.

Walks the two trees for files matching *receipt*.json, loads each through
ryko_lm.provenance.verify_receipt and prints one line per receipt:

    path -> PASS/FAIL   (failed artifact entries listed inline)

Exit code 0 only if every receipt verifies fully.
"""
from __future__ import annotations

import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Ryko", "src"))

from ryko_lm.provenance import verify_receipt  # noqa: E402

RYKO = os.path.join(ROOT, "Ryko")


def main() -> int:
    patterns = [
        os.path.join(RYKO, "artifacts", "**", "*receipt*.json"),
        os.path.join(RYKO, "data", "**", "*receipt*.json"),
    ]
    paths = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})
    if not paths:
        print("no receipts found")
        return 1

    n_pass = n_fail = 0
    for rp in paths:
        rel = os.path.relpath(rp, RYKO)
        try:
            ok, report = verify_receipt(rp)
        except Exception as exc:  # malformed receipt JSON etc.
            print(f"{rel} -> FAIL   (unreadable: {exc})")
            n_fail += 1
            continue
        failed = [f'{r["check"]} ({r["detail"]})' for r in report if not r["ok"]]
        if ok:
            n_entries = len(report) - 1  # minus receipt_sha256 self-check
            print(f"{rel} -> PASS   ({n_entries} artifact entries verified)")
            n_pass += 1
        else:
            print(f"{rel} -> FAIL   failed entries: {', '.join(failed) if failed else '(receipt_sha256)'}")
            n_fail += 1

    print(f"\nsummary: {n_pass} PASS / {n_fail} FAIL over {len(paths)} receipts")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
