"""Secret gate tests — ZYKO-pattern local protection.

Synthetic credentials below are built by *concatenation* and carry the
``ryko:secret-ok`` exemption marker on their source lines: the gate's
documented contract for test fixtures.
"""

from __future__ import annotations

from ryko_lm.security.secret_scan import (
    EXEMPT_MARKER,
    added_lines,
    scan_text,
    scan_to_json,
    scan_tree,
)

_SYNTH = "A1b2C3d4E5f6G7h8I9j0KkLm"  # ryko:secret-ok (synthetic fixture, half-token)


def test_github_token_detected_with_fingerprint_only():
    token = "ghp_" + _SYNTH  # ryko:secret-ok (concatenation avoids a literal)
    hits = scan_text(f"curl -H 'Authorization: Bearer {token}' https://api")
    assert hits and hits[0].name == "github-token"
    assert hits[0].severity == "critical"
    assert len(hits[0].evidence_sha16) == 16
    assert token not in hits[0].evidence_sha16


def test_cred_assignment_medium():
    hits = scan_text("my password = hunter2secret")  # ryko:secret-ok (fixture)
    assert any(h.name == "cred-assignment" and h.severity == "medium" for h in hits)


def test_clean_text_passes():
    assert scan_text("العلم نور والجهل ظلام = معادلة الفائدة") == ()


def test_exempt_marker_skips_line():
    line = f"token = 'ghp_' + '{_SYNTH}'  # {EXEMPT_MARKER}"
    assert scan_text(line) == ()


def test_added_lines_from_diff():
    token = "ghp_" + _SYNTH
    diff = "\n".join([
        "diff --git a/a.py b/a.py",
        "--- a/a.py",
        "+++ b/a.py",
        "@@ -1,2 +1,3 @@",
        " keep = 1",
        f'+leak = "{token}"',
        "-removed_secret = 'ghp_' + 'zzzz'",
    ])
    added = added_lines(diff)
    assert "leak" in added
    assert "removed_secret" not in added
    hits = scan_text(added)
    assert any(h.name == "github-token" for h in hits)


def test_scan_tree_deterministic_and_skips_git(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "doc.md").write_text("safe arabic text", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "leak.py").write_text("token = ghp_" + "Q" * 30, encoding="utf-8")
    first = scan_tree(tmp_path)
    second = scan_tree(tmp_path)
    assert scan_to_json(first) == scan_to_json(second)
    assert first["critical"] == 0
    assert first["scanned_files"] == 1
