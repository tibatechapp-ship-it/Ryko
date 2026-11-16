"""Surgically remove the last N blank-line-separated docs from a large file.

Used to undo an augment_corpus append whose input selection turned out to be
wrong: the appended docs sit contiguously at the tail, so dropping exactly the
last N docs restores the pre-augment byte state (verified by re-counting).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CHUNK = 8 << 20


def find_cut_offset(path: Path, n_docs: int) -> int:
    """Byte offset where the (N)-th-from-last doc starts."""
    size = path.stat().st_size
    with path.open("rb") as f:
        # walk backwards, tracking segment (doc) boundaries by b"\n\n"
        tail = b""
        docs_seen = 0
        pos = size
        cut_in_tail = None
        buf = tail
        while pos > 0 and cut_in_tail is None:
            step = min(CHUNK, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + tail
            # count complete docs in this buffer: split on \n\n
            parts = buf.split(b"\n\n")
            # if this is not the final chunk, the last part may be an
            # incomplete doc — re-attach it to the next (earlier) buffer
            if pos > 0:
                tail = parts.pop()
            else:
                # at file start: leading empty piece if buffer starts with \n\n
                if parts and parts[0] == b"":
                    parts.pop(0)
            # docs in this buffer, counted from the END of the file
            for piece in reversed(parts):
                if piece == b"" and pos > 0:
                    continue
                docs_seen += 1
                if docs_seen == n_docs:
                    # absolute offset of this doc's start
                    idx = buf.rfind(piece)
                    # walk back to the true start: rfind may match inside an
                    # earlier doc; find the FIRST occurrence boundary properly
                    # by scanning split offsets
                    off = 0
                    for p in parts:
                        if p is piece:
                            break
                        off += len(p) + 2
                    cut_in_tail = pos + off
                    break
        if cut_in_tail is None:
            raise SystemExit(f"file has fewer than {n_docs} docs")
        return cut_in_tail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--docs", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cut = find_cut_offset(args.file, args.docs)
    size = args.file.stat().st_size
    print(f"cut at byte {cut:,} of {size:,} (removing {size - cut:,} bytes)")
    if not args.dry_run:
        with args.file.open("r+b") as f:
            f.truncate(cut)
        print("truncated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
