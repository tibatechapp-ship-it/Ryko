"""Signed provenance receipts for RYKO artifacts.

Adopted from the ZYKO project's *offline-verifiable receipt* pattern
(github.com/tibatechapp-ship-it/zyko): every pipeline stage can emit a
receipt whose fingerprints anyone can re-compute offline — no trust, only
hashes.  ZYKO uses receipts to make routing decisions auditable; RYKO uses
them to make corpus and checkpoint lineage auditable.

A receipt is a plain JSON document::

    {
      "kind": "ryko-corpus-receipt",
      "created_utc": "2026-11-15T09:00:00Z",
      "artifacts": [
        {"path": "data/processed_v3/train.bin",
         "sha256": "...", "bytes": 492279920}
      ],
      "meta": {"tokens": 246139960, "sources": {...}},
      "receipt_sha256": "..."          # hash of everything above
    }

``verify_receipt`` re-computes every artifact hash and the receipt hash and
returns a per-file pass/fail report.

Usage::

    python -m ryko_lm.provenance make data/processed_v3 \
        --kind corpus --meta '{"tokens": 246139960}' \
        --out data/processed_v3/receipt.json
    python -m ryko_lm.provenance verify data/processed_v3/receipt.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        while True:
            chunk = src.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fingerprint_dir(directory: Path, patterns: tuple[str, ...] = ("*.bin", "*.txt", "*.json")) -> list[dict]:
    """Hash every artifact under ``directory`` (sorted, stable).

    Receipt files themselves (``receipt.json`` / ``*.receipt.json``) are
    EXCLUDED: a receipt must never fingerprint itself (or a previous
    receipt) — the self-entry made any re-signing fail by construction,
    since the written receipt pointed at the hash of the receipt it was
    about to replace (v1.22.0 fix, tested).
    """
    entries: list[dict] = []
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            if not path.is_file():
                continue
            if path.name == "receipt.json" or path.name.endswith(".receipt.json"):
                continue
            entries.append({
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
    return entries


def make_receipt(target: Path, kind: str = "generic", meta: dict | None = None) -> dict:
    """Build a receipt for a file or a directory of artifacts."""
    target = Path(target)
    if target.is_dir():
        artifacts = fingerprint_dir(target)
    else:
        artifacts = [{"path": str(target), "sha256": sha256_file(target),
                      "bytes": target.stat().st_size}]
    receipt = {
        "kind": f"ryko-{kind}-receipt",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts": artifacts,
        "meta": meta or {},
    }
    payload = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return receipt


def verify_receipt(receipt_path: Path) -> tuple[bool, list[dict]]:
    """Re-compute all hashes; return (all_ok, per-artifact report)."""
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    payload = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    expected = receipt["receipt_sha256"]
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    report = [{"check": "receipt_sha256", "ok": actual == expected,
               "detail": "intact" if actual == expected else "receipt was modified"}]
    all_ok = actual == expected
    root = Path(receipt_path).parent
    for artifact in receipt.get("artifacts", []):
        path = Path(artifact["path"])
        if not path.exists():
            # receipts may live next to the artifacts or anywhere: try relative
            alt = root / path.name
            path = alt if alt.exists() else path
        if not path.exists():
            report.append({"check": artifact["path"], "ok": False, "detail": "missing"})
            all_ok = False
            continue
        digest = sha256_file(path)
        ok = digest == artifact["sha256"]
        report.append({"check": str(path), "ok": ok,
                       "detail": "match" if ok else "hash mismatch"})
        all_ok = all_ok and ok
    return all_ok, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RYKO provenance receipts (ZYKO pattern).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_make = sub.add_parser("make", help="create a receipt for a file/dir")
    p_make.add_argument("target", type=Path)
    p_make.add_argument("--kind", default="generic")
    p_make.add_argument("--meta", default="{}", help="JSON metadata attached to the receipt")
    p_make.add_argument("--out", type=Path, required=True)

    p_ver = sub.add_parser("verify", help="verify a receipt offline")
    p_ver.add_argument("receipt", type=Path)

    args = parser.parse_args(argv)

    if args.cmd == "make":
        receipt = make_receipt(args.target, kind=args.kind, meta=json.loads(args.meta))
        args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[receipt] {len(receipt['artifacts'])} artifacts -> {args.out}")
        print(f"[receipt] sha256={receipt['receipt_sha256'][:16]}...")
        return 0

    ok, report = verify_receipt(args.receipt)
    for row in report:
        print(f"[{'PASS' if row['ok'] else 'FAIL'}] {row['check']}: {row['detail']}")
    print("[receipt] verification:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
