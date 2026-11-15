"""Secret gate — ZYKO-pattern local secret scan for RYKO.

Adapted from ``zyko.utils.secret_scan`` (github.com/tibatechapp-ship-it/zyko).
The CI runs gitleaks *after* a push; this gate runs **before** (local
developer loop, pre-push), so a leaked credential never reaches GitHub in
the first place.

Policies (all explicit, nothing implicit):

* one pattern table — GitHub PAT values (critical) and textual credential
  assignments (medium);
* exemption by explicit in-line marker ``ryko:secret-ok`` (tests/docs only);
* evidence is a sha16 fingerprint — the proof verifies without leaking the
  value;
* ``--diff A..B`` scans only added lines (removing a secret is not a crime);
* exit 1 on critical hits only; medium hits are logged, not blocking;
* byte-deterministic output — same scan, same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

EXEMPT_MARKER = "ryko:secret-ok"

GITHUB_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b")
CRED_ASSIGN_RE = re.compile(
    r"(?i)\b(password|passwd|secret|token|api[_-]?key|access[_-]?key)\b\s*[:=]\s*"
    r"[\"']?[^\s\"']{6,}"
)

_SKIP_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".pytest_cache",
})
_SCAN_SUFFIXES = (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
                  ".cfg", ".ini", ".sh", ".js", ".ts", ".html", ".css")


@dataclass(frozen=True)
class SecretHit:
    name: str
    severity: str        # critical | medium
    line_no: int
    evidence_sha16: str
    message_ar: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "severity": self.severity,
            "line_no": self.line_no,
            "evidence_sha16": self.evidence_sha16,
            "message_ar": self.message_ar,
        }


def _hit(name: str, severity: str, line_no: int, value: str, message_ar: str) -> SecretHit:
    return SecretHit(name, severity, line_no,
                     hashlib.sha256(value.encode()).hexdigest()[:16], message_ar)


def scan_text(text: str) -> tuple[SecretHit, ...]:
    """Line-level scan; lines carrying the exemption marker are skipped."""
    hits: list[SecretHit] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if EXEMPT_MARKER in line:
            continue
        for m in GITHUB_TOKEN_RE.finditer(line):
            hits.append(_hit("github-token", "critical", line_no, m.group(0),
                             "قيمة توكن مضمّنة في السطر"))
        for m in CRED_ASSIGN_RE.finditer(line):
            hits.append(_hit("cred-assignment", "medium", line_no, m.group(0),
                             "إسناد بيانات اعتماد نصياً في السطر"))
    return tuple(hits)


def scan_tree(root: str | Path, *, max_bytes: int = 2_000_000) -> dict:
    """Whole-tree scan under explicit ceilings; deterministic order."""
    root = Path(root)
    scanned = 0
    findings: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix not in _SCAN_SUFFIXES:
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        scanned += 1
        for hit in scan_text(text):
            entry = hit.as_dict()
            entry["path"] = str(path.relative_to(root))
            findings.append(entry)
    return {"scanned_files": scanned, "hits": findings,
            "critical": sum(1 for f in findings if f["severity"] == "critical"),
            "medium": sum(1 for f in findings if f["severity"] == "medium")}


def added_lines(diff_text: str) -> str:
    """Extract added lines ('+' rows) from a unified diff — deletion is free."""
    out: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            out.append(line[1:])
    return "\n".join(out)


def scan_diff(rev_range: str, cwd: str | Path = ".") -> dict:
    proc = subprocess.run(
        ["git", "diff", rev_range],
        cwd=str(cwd), capture_output=True, text=True, check=True,
    )
    text = added_lines(proc.stdout)
    hits = scan_text(text)
    return {
        "mode": f"diff {rev_range}",
        "hits": [h.as_dict() for h in hits],
        "critical": sum(1 for h in hits if h.severity == "critical"),
        "medium": sum(1 for h in hits if h.severity == "medium"),
    }


def scan_to_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ryko_secret_scan",
        description="RYKO local secret gate (ZYKO pattern) — runs before push.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tree", type=Path, help="scan a directory tree")
    group.add_argument("--diff", type=str, help="scan added lines of a git rev range")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args(argv)

    payload = (scan_diff(args.diff) if args.diff else scan_tree(args.tree))
    if args.json:
        print(scan_to_json(payload))
    else:
        for hit in payload["hits"]:
            print(f"[{hit['severity'].upper():8s}] {hit['path']}:{hit['line_no']} "
                  f"{hit['name']} sha16={hit['evidence_sha16']}")
        print(f"[secret-scan] scanned={payload.get('scanned_files', 'diff')} "
              f"critical={payload['critical']} medium={payload['medium']}")
    return 1 if payload["critical"] else 0


if __name__ == "__main__":
    sys.exit(main())
