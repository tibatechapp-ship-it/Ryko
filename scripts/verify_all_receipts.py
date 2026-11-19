#!/usr/bin/env python3
"""Empirical audit: verify every *receipt*.json under Ryko artifacts/ and data/.

Walks the two trees for files matching *receipt*.json, loads each through
ryko_lm.provenance.verify_receipt and prints one line per receipt:

    path -> PASS/FAIL/SKIP   (failed artifact entries listed inline)

Failure classification
    - entry hash mismatch        -> FAIL  (real corruption)
    - entry missing from disk AND
      the path is gitignored     -> SKIP  (intentionally untracked build
                                          artifact; receipt kept for lineage)
    - entry missing but tracked  -> FAIL  (someone deleted a committed file)

Exit code 0 only when there are zero FAIL lines.  Pass --strict to also fail
on SKIP (i.e. require every referenced artifact on disk).
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ryko_lm.provenance import verify_receipt

# This copy lives inside the repository, so the repo root IS the Ryko tree.
RYKO = ROOT


def _git_ignored(rel: str) -> bool:
    """True if rel is ignored by .gitignore (deliberately untracked)."""
    try:
        r = subprocess.run(
            ["git", "check-ignore", "-q", "--", rel],
            cwd=RYKO,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,  # returncode IS the signal here
        )
        return r.returncode == 0
    except FileNotFoundError:
        return False  # git unavailable -> be conservative, treat as FAIL


def main() -> int:
    strict = "--strict" in sys.argv[1:]
    patterns = [
        os.path.join(RYKO, "artifacts", "**", "*receipt*.json"),
        os.path.join(RYKO, "data", "**", "*receipt*.json"),
    ]
    paths = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})
    if not paths:
        print("no receipts found")
        return 1

    n_pass = n_fail = n_skip = 0
    for rp in paths:
        rel = os.path.relpath(rp, RYKO)
        try:
            ok, report = verify_receipt(rp)
        except Exception as exc:  # noqa: BLE001 — malformed receipt JSON etc.
            print(f"{rel} -> FAIL   (unreadable: {exc})")
            n_fail += 1
            continue

        bad = [r for r in report if not r["ok"]]
        skipped = [r for r in bad if "missing" in r["detail"] and _git_ignored(r["check"])]
        hard = [r for r in bad if r not in skipped]

        if ok:
            n_entries = len(report) - 1  # minus receipt_sha256 self-check
            print(f"{rel} -> PASS   ({n_entries} artifact entries verified)")
            n_pass += 1
        elif not hard and skipped:
            names = ", ".join(f'{r["check"]}' for r in skipped)
            print(f"{rel} -> SKIP   (untracked build artifacts, lineage kept: {names})")
            n_skip += 1
        else:
            failed = [f'{r["check"]} ({r["detail"]})' for r in hard]
            print(f"{rel} -> FAIL   failed entries: {', '.join(failed) if failed else '(receipt_sha256)'}")
            n_fail += 1

    print(f"\nsummary: {n_pass} PASS / {n_skip} SKIP / {n_fail} FAIL over {len(paths)} receipts")
    if n_fail:
        return 1
    if strict and n_skip:
        print("strict mode: SKIP counts as failure")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
