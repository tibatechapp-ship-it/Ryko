"""Augment an existing processed corpus with new raw sources (streaming).

Re-cleaning the full corpus would load every document in RAM; instead this
script streams:

1.  fingerprint every paragraph of the existing train/val split into a global
    dedup set (truncated hashes as ints — compact even for 100M+ chars),
2.  run the *standard* cleaning cascade from :mod:`ryko_lm.data.clean_corpus`
    over each new raw JSONL input, deduplicate against the global set, and
    write kept documents to a temp ``train_new`` file — a deterministic hash
    fraction of documents is diverted to ``val_new``,
3.  concatenate ``old + new`` into the target directory (``train.txt`` /
    ``val.txt``) and emit ``stats.json``.

The output layout is byte-compatible with ``clean_corpus`` (paragraph per
line, blank line between documents), so ``binarize`` works unchanged.

Usage:
    python -m ryko_lm.data.augment_corpus \
        --existing-dir data/processed_v2 \
        --inputs data/raw/openiti.jsonl data/raw/shamela.jsonl \
        --out-dir data/processed_v3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from .clean_corpus import CleanOptions, _clean_line

FP_FOLD_RE = None  # imported lazily to mirror clean_corpus folding


def _fold(line: str) -> str:
    import re
    return re.sub(r"[^\u0600-\u06FFA-Za-z]", "", line)


def fp_int(line: str) -> int:
    """Truncated paragraph fingerprint (first 8 bytes of sha256)."""
    return int.from_bytes(hashlib.sha256(_fold(line).encode("utf-8")).digest()[:8], "big")


def load_existing_fingerprints(existing_dir: Path) -> tuple[set[int], int]:
    seen: set[int] = set()
    count = 0
    for name in ("train.txt", "val.txt"):
        path = existing_dir / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as src:
            for line in src:
                line = line.rstrip("\n")
                if not line:
                    continue
                seen.add(fp_int(line))
                count += 1
    return seen, count


def doc_seed(paragraphs: list[str]) -> int:
    return fp_int(paragraphs[0]) if paragraphs else 0


def stream_clean_file(path: Path, opts: CleanOptions, seen: set[int]):
    """Yield ``(paragraphs, dropped_duplicates)`` docs from a raw JSONL file."""
    with path.open(encoding="utf-8") as src:
        for raw in src:
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text = record.get("text") or ""
            paragraphs: list[str] = []
            for block in text.split("\n\n") if "\n\n" in text else [text]:
                for sub in block.splitlines():
                    if not sub.strip():
                        continue
                    cleaned = _clean_line(sub, opts)
                    if cleaned is not None:
                        paragraphs.append(cleaned)
            if not paragraphs:
                continue
            unique: list[str] = []
            dropped = 0
            for para in paragraphs:
                h = fp_int(para)
                if h in seen:
                    dropped += 1
                    continue
                seen.add(h)
                unique.append(para)
            if unique:
                yield unique, dropped


def copy_stream(src: Path, dst_fileobj) -> None:
    with src.open("rb") as fsrc:
        shutil.copyfileobj(fsrc, dst_fileobj, length=1 << 20)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Augment a processed corpus (streaming).")
    parser.add_argument("--existing-dir", type=Path, required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.004,
                        help="fraction of NEW documents diverted to val")
    parser.add_argument("--min-line-len", type=int, default=40)
    parser.add_argument("--min-arabic-letters", type=int, default=20)
    parser.add_argument("--min-arabic-ratio", type=float, default=0.55)
    args = parser.parse_args(argv)

    opts = CleanOptions(
        min_line_len=args.min_line_len,
        min_arabic_letters=args.min_arabic_letters,
        min_arabic_ratio=args.min_arabic_ratio,
        remove_diacritics=True,
        unify_alef=False,
        unify_yaa=False,
        taa_marbuta_to_haa=False,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_new = args.out_dir / "train_new.txt"
    val_new = args.out_dir / "val_new.txt"

    print("[augment] fingerprinting existing corpus ...", flush=True)
    seen, existing_paras = load_existing_fingerprints(args.existing_dir)
    print(f"[augment] existing paragraphs fingerprinted: {existing_paras:,}", flush=True)

    per_source: dict[str, dict] = {}
    val_bucket = max(1, int(1 / args.val_ratio))
    # NOTE: both temp sinks are opened ONCE across all inputs — reopening per
    # input in "w" mode would wipe the previous source's data.
    with train_new.open("w", encoding="utf-8") as tr, \
            val_new.open("w", encoding="utf-8") as va:
        for path in args.inputs:
            docs_kept, chars_kept, dup = 0, 0, 0
            for paragraphs, dropped in stream_clean_file(path, opts, seen):
                body = "\n".join(paragraphs) + "\n\n"
                chars_kept += sum(len(p) for p in paragraphs)
                if doc_seed(paragraphs) % val_bucket == 0:
                    va.write(body)
                else:
                    tr.write(body)
                docs_kept += 1
                dup += dropped
            per_source[path.name] = {"docs": docs_kept, "chars": chars_kept,
                                     "dup_paragraphs": dup}
            print(f"[augment] {path.name}: {docs_kept} docs, "
                  f"{chars_kept / 1e6:.1f}M chars, dup_dropped={dup}", flush=True)

    # concatenate old + new
    train_out = args.out_dir / "train.txt"
    val_out = args.out_dir / "val.txt"
    with train_out.open("wb") as out:
        copy_stream(args.existing_dir / "train.txt", out)
        copy_stream(train_new, out)
    with val_out.open("wb") as out:
        copy_stream(args.existing_dir / "val.txt", out)
        copy_stream(val_new, out)
    train_new.unlink()
    val_new.unlink()

    old_chars = sum(len(l) for l in (args.existing_dir / "train.txt").open(encoding="utf-8"))
    stats = {
        "existing_dir": str(args.existing_dir),
        "existing_paragraphs": existing_paras,
        "new_sources": per_source,
        "val_ratio_new": args.val_ratio,
        "chars_existing_train": old_chars,
    }
    (args.out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {train_out} ({train_out.stat().st_size / 1e6:.0f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
