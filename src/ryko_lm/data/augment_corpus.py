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

With ``--in-place`` the concatenation step appends the new paragraphs to the
existing ``train.txt``/``val.txt`` files directly, so the disk peak is only
the *new* paragraphs instead of a full second copy of the corpus — the old
and new directories may then be the same directory.

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
import shutil as _shutil
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


def load_existing_fingerprints(existing_dir: Path) -> tuple[set[int], int, int]:
    seen: set[int] = set()
    count = 0
    chars = 0
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
                chars += len(line)
    return seen, count, chars


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


def _require_free(inputs: list[Path], factor: float = 1.5) -> None:
    """Abort early if the filesystem cannot hold the cleaned new content.

    The cleaned output is typically <= the raw input bytes (mARkdown scaffolding
    removed), so requiring ~1.5x the input size keeps a safety margin without
    blocking legitimate runs.
    """
    need = sum(p.stat().st_size for p in inputs if p.exists()) * factor
    usage = _shutil.disk_usage(
        inputs[0].parent if inputs else Path(".")
    )
    if usage.free < need:
        raise SystemExit(
            f"[error] low disk: need ~{need / 1e9:.2f} GB free "
            f"(1.5x inputs), have {usage.free / 1e9:.2f} GB — "
            "free space first or split the inputs"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Augment a processed corpus (streaming).")
    parser.add_argument("--existing-dir", type=Path, required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--in-place", action="store_true",
                        help="append new paragraphs to the existing train/val "
                             "files instead of copying them into a new dir "
                             "(disk peak = new content only); --out-dir must "
                             "already hold train.txt/val.txt")
    parser.add_argument("--val-ratio", type=float, default=0.004,
                        help="fraction of NEW documents diverted to val")
    parser.add_argument("--no-new-val", action="store_true",
                        help="freeze the held-out ruler: every NEW document "
                             "goes to train and val.txt is left byte-identical "
                             "(v1.15.0 — the standard practice of keeping the "
                             "eval set fixed when the train set grows)")
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
    _require_free(args.inputs)
    train_new = args.out_dir / "train_new.txt"
    val_new = args.out_dir / "val_new.txt"

    # in-place mode streams the cleaned paragraphs directly into the final
    # train/val files (append mode): no second copy of the corpus is ever
    # written, so the disk peak equals the *new* content only.  Byte output is
    # identical to the temp-file + append path.
    if args.in_place and not ((args.out_dir / "train.txt").exists()
                              and (args.out_dir / "val.txt").exists()):
        print("[error] --in-place needs train.txt/val.txt in --out-dir", flush=True)
        return 2
    if args.in_place:
        train_sink = (args.out_dir / "train.txt").open("a", encoding="utf-8")
        val_sink = (args.out_dir / "val.txt").open("a", encoding="utf-8")
    else:
        train_sink = train_new.open("w", encoding="utf-8")
        val_sink = val_new.open("w", encoding="utf-8")

    print("[augment] fingerprinting existing corpus ...", flush=True)
    seen, existing_paras, existing_chars = load_existing_fingerprints(args.existing_dir)
    print(f"[augment] existing paragraphs fingerprinted: {existing_paras:,}", flush=True)

    per_source: dict[str, dict] = {}
    val_bucket = max(1, int(1 / args.val_ratio))
    diverted = 0
    # NOTE: both sinks are opened ONCE across all inputs — reopening per
    # input in "w" mode would wipe the previous source's data.
    with train_sink as tr, val_sink as va:
        for path in args.inputs:
            docs_kept, chars_kept, dup = 0, 0, 0
            for paragraphs, dropped in stream_clean_file(path, opts, seen):
                body = "\n".join(paragraphs) + "\n\n"
                chars_kept += sum(len(p) for p in paragraphs)
                if not args.no_new_val and doc_seed(paragraphs) % val_bucket == 0:
                    va.write(body)
                    diverted += 1
                else:
                    tr.write(body)
                docs_kept += 1
                dup += dropped
            per_source[path.name] = {"docs": docs_kept, "chars": chars_kept,
                                     "dup_paragraphs": dup}
            print(f"[augment] {path.name}: {docs_kept} docs, "
                  f"{chars_kept / 1e6:.1f}M chars, dup_dropped={dup}", flush=True)
    if args.no_new_val:
        print(f"[augment] --no-new-val: held-out ruler frozen, diverted=0", flush=True)

    # concatenate old + new
    train_out = args.out_dir / "train.txt"
    val_out = args.out_dir / "val.txt"
    old_chars = existing_chars
    if args.in_place:
        # appended directly during cleaning — nothing to concatenate
        pass
    else:
        if not (args.existing_dir / "train.txt").exists() or \
                not (args.existing_dir / "val.txt").exists():
            print("[error] copy mode needs train.txt/val.txt in --existing-dir", flush=True)
            return 2
        with train_out.open("wb") as out:
            copy_stream(args.existing_dir / "train.txt", out)
            copy_stream(train_new, out)
        with val_out.open("wb") as out:
            copy_stream(args.existing_dir / "val.txt", out)
            copy_stream(val_new, out)
    if not args.in_place:
        train_new.unlink()
        val_new.unlink()

    stats = {
        "mode": "in-place" if args.in_place else "copy",
        "existing_dir": str(args.existing_dir),
        "existing_paragraphs": existing_paras,
        "new_sources": per_source,
        "val_ratio_new": args.val_ratio,
        "no_new_val": bool(args.no_new_val),
        "chars_existing_train": old_chars,
    }
    (args.out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {train_out} ({train_out.stat().st_size / 1e6:.0f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
