"""Collect the Linguistics layer (طبقة اللغة) books from the Shamela v4 mirror.

A thin, defaults-first shim over :mod:`ryko_lm.data.collect_fikr` — the
linguistics manifest (``data/knowledge/lugha_manifest.json``, kind
``ryko-lugha-manifest``) is resolved against the same mirror listing and
fetched with page-level provenance, so the Knowledge Core can ingest the
language classics (نحو/صرف/بلاغة/معاجم/لسانيات/أدب/مقررات) under the
``lugha`` source.

Resumable via ``data/raw/lugha.progress.json``.  The big dictionaries
(لسان العرب، تاج العروس، تهذيب اللغة، العين …) carry ``size_class`` in the
manifest; pass ``--max-file-mb`` generously for the full pass and let the
build receipt document whatever the byte budget deferred.

Usage:
    python -m ryko_lm.data.collect_lugha --max-file-mb 30
        (matn/medium pass — small texts first)
    python -m ryko_lm.data.collect_lugha --max-file-mb 200
        (full pass incl. xlarge dictionaries; resumable — done books skip)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ryko_lm.data.collect_fikr import list_page_files, resolve_paths
from ryko_lm.data.collect_fikr import fetch_book, main as _collect_main

DEFAULT_MANIFEST = Path("data/knowledge/lugha_manifest.json")


def resolve_scope(manifest_path: Path) -> dict:
    """Load the lugha manifest and report resolution against the mirror."""
    manifest = manifest_path.read_text(encoding="utf-8")
    from ryko_lm.knowledge.ideas import load_manifest

    m = load_manifest(str(manifest_path) if manifest_path.is_absolute()
                      else manifest_path)
    listing = list_page_files()
    resolved, unresolved = resolve_paths(m, listing)
    return {
        "kind": m["kind"],
        "layer": m.get("layer_label", "لغة"),
        "sections": len(m["schools"]),
        "books_total": len(m["books"]),
        "resolved": len(resolved),
        "pending": len(unresolved),
    }


if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if not any(a == "--manifest" for a in argv):
        argv = ["--manifest", str(DEFAULT_MANIFEST), *argv]
    sys.exit(_collect_main(argv))
