"""Clean and normalize the raw Arabic corpus into a training-ready text file.

Streaming pipeline (v1.6.0+): the previous version loaded every document of
every source into RAM before dedup/split, which OOM-killed on small boxes once
the corpus passed ~1 GB of raw text.  The rewrite is two-pass streaming with a
constant memory footprint:

Pass 1 — ``clean_streaming``
    1. Unicode NFKC — folds Arabic presentation forms (ﻻ -> لا) and legacy
       glyphs into their canonical code points.
    2. Strip markup: HTML tags, URLs, e-mails, wiki residue (``[[..]]`` kept
       as inner text, ``[1]``-style citations, file/image namespaces).
    3. Arabic orthography cleanup: tatweel (U+0640) and diacritics
       (U+064B..U+065F, U+0670, U+06D6..U+06ED) removed; aggressive
       unifications (alef/yaa/taa) optional and off by default.
    4. Line filtering: minimum length, minimum Arabic-letter ratio.
    5. Exact-duplicate removal at paragraph level — fingerprints are stored as
       int64 (first 8 bytes of sha256 of the punctuation-folded text) inside a
       plain ``set``; ~3M paragraphs cost ~150-200 MB instead of the >1.5 GB
       the old list-of-lists representation needed.
    Cleaned documents stream straight to a temporary file, one paragraph per
    line, a blank line separating documents.

Pass 2 — ``split_streaming``
    Documents are byte-offset indexed in the temp file, the index is shuffled
    with the seeded RNG (identical semantics to shuffling the doc list), and
    train/val splits stream back out by seeking.

Output: one paragraph per line; a blank line separates documents.

Usage:
    python -m ryko_lm.data.clean_corpus \
        --inputs data/raw/wiki.jsonl data/raw/tashkeela.jsonl \
        --out-dir data/processed_v4 --val-ratio 0.01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Arabic character machinery
# ---------------------------------------------------------------------------
TATWEEL = "\u0640"
DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
WIKI_CITE_RE = re.compile(r"\[\d+\]|\[أ?\]|\[م\]|\[بحاجة لمصدر\]")
WIKI_LINK_RE = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
WIKI_FILE_RE = re.compile(r"\[\[(ملف|File|تصنيف|Category|صورة):[^\]]+\]\]")
MULTISPACE_RE = re.compile(r"[ \t\u00a0]+")
EMPTY_PARENS_RE = re.compile(r"\(\s*\)|\[\s*\]")


def normalize_arabic(
    text: str,
    remove_diacritics: bool = True,
    unify_alef: bool = False,
    unify_yaa: bool = False,
    taa_marbuta_to_haa: bool = False,
) -> str:
    """Apply the Ryko Arabic normalization cascade to a text string."""
    text = unicodedata.normalize("NFKC", text)
    text = DIACRITICS_RE.sub("", text) if remove_diacritics else text
    text = text.replace(TATWEEL, "")
    if unify_alef:
        text = re.sub("[\u0623\u0625\u0622]", "\u0627", text)
    if unify_yaa:
        text = text.replace("\u0649", "\u064A")
    if taa_marbuta_to_haa:
        text = text.replace("\u0629", "\u0647")
    return text


class CleanOptions:
    def __init__(self, min_line_len: int, min_arabic_letters: int, min_arabic_ratio: float,
                 remove_diacritics: bool, unify_alef: bool, unify_yaa: bool,
                 taa_marbuta_to_haa: bool) -> None:
        self.min_line_len = min_line_len
        self.min_arabic_letters = min_arabic_letters
        self.min_arabic_ratio = min_arabic_ratio
        self.remove_diacritics = remove_diacritics
        self.unify_alef = unify_alef
        self.unify_yaa = unify_yaa
        self.taa_marbuta_to_haa = taa_marbuta_to_haa


def _clean_line(line: str, opts: CleanOptions) -> str | None:
    line = HTML_RE.sub(" ", line)
    line = URL_RE.sub(" ", line)
    line = EMAIL_RE.sub(" ", line)
    line = WIKI_FILE_RE.sub(" ", line)
    line = WIKI_LINK_RE.sub(r"\1", line)
    line = WIKI_CITE_RE.sub("", line)
    line = EMPTY_PARENS_RE.sub("", line)
    line = normalize_arabic(
        line,
        remove_diacritics=opts.remove_diacritics,
        unify_alef=opts.unify_alef,
        unify_yaa=opts.unify_yaa,
        taa_marbuta_to_haa=opts.taa_marbuta_to_haa,
    )
    line = MULTISPACE_RE.sub(" ", line).strip()
    if len(line) < opts.min_line_len:
        return None
    arabic = len(ARABIC_LETTER_RE.findall(line))
    if arabic < opts.min_arabic_letters:
        return None
    if arabic / len(line) < opts.min_arabic_ratio:
        return None
    return line


def _fingerprint(line: str) -> str:
    """Full sha256 hex of a paragraph after punctuation/whitespace fold."""
    folded = re.sub(r"[^\u0600-\u06FFA-Za-z]", "", line)
    return hashlib.sha256(folded.encode("utf-8")).hexdigest()


def _fingerprint64(line: str) -> int:
    """Memory-cheap fingerprint: first 8 bytes of the sha256 as an int64.

    Collision probability for ~10M distinct paragraphs against a 64-bit space
    is < 3e-6 — acceptable for corpus dedup.
    """
    folded = re.sub(r"[^\u0600-\u06FFA-Za-z]", "", line)
    return int.from_bytes(hashlib.sha256(folded.encode("utf-8")).digest()[:8], "big")


def _clean_document(text: str, opts: CleanOptions) -> list[str]:
    """Clean one raw document into a list of kept paragraph lines (no dedup)."""
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        for line in block.split("\n"):
            cleaned = _clean_line(line, opts)
            if cleaned is not None:
                paragraphs.append(cleaned)
    return paragraphs


def clean_streaming(
    inputs: list[Path],
    opts: CleanOptions,
    tmp_path: Path,
) -> dict:
    """Pass 1: clean every JSONL input, dedup paragraphs, stream docs to tmp.

    Returns the stats dict (docs/paragraphs/chars/dropped) so the caller can
    merge it into ``stats.json``.
    """
    seen: set[int] = set()
    n_docs = 0
    n_paras = 0
    dropped = 0
    kept_chars = 0
    per_source: dict[str, int] = {}
    t0 = time.time()

    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as out:
        for path in inputs:
            before = n_docs
            with path.open(encoding="utf-8") as src:
                for raw in src:
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    text = record.get("text") or ""
                    paragraphs = _clean_document(text, opts)
                    if not paragraphs:
                        continue
                    unique: list[str] = []
                    for para in paragraphs:
                        fp = _fingerprint64(para)
                        if fp in seen:
                            dropped += 1
                            continue
                        seen.add(fp)
                        unique.append(para)
                    if not unique:
                        continue
                    out.write("\n".join(unique) + "\n\n")
                    n_docs += 1
                    n_paras += len(unique)
                    kept_chars += sum(len(p) for p in unique)
            per_source[path.name] = n_docs - before
            print(
                f"[clean] {path.name}: +{n_docs - before} docs "
                f"(total {n_docs}, {kept_chars / 1e6:.1f}M chars, "
                f"dup_paras={dropped}) in {time.time() - t0:.0f}s",
                flush=True,
            )

    return {
        "mode": "streaming",
        "inputs": per_source,
        "documents_total": n_docs,
        "paragraphs_kept": n_paras,
        "paragraphs_dropped_duplicate": dropped,
        "chars_kept": kept_chars,
        "seconds": round(time.time() - t0, 1),
    }


def _index_documents(tmp_path: Path) -> list[tuple[int, int]]:
    """Byte-offset (start, end) index of blank-line-separated documents."""
    offsets: list[tuple[int, int]] = []
    pos = 0
    doc_start: int | None = None
    doc_end = 0
    with tmp_path.open("rb") as f:
        for line in f:
            if line.strip():
                if doc_start is None:
                    doc_start = pos
                doc_end = pos + len(line)
            else:
                if doc_start is not None:
                    offsets.append((doc_start, doc_end))
                    doc_start = None
            pos += len(line)
    if doc_start is not None:
        offsets.append((doc_start, doc_end))
    return offsets


def split_streaming(
    tmp_path: Path,
    train_path: Path,
    val_path: Path,
    val_ratio: float,
    seed: int,
) -> dict:
    """Pass 2: seeded shuffle of the document index, then stream both splits.

    Semantics match the legacy in-memory path: ``rng.shuffle(docs)`` then
    first ``val_ratio`` slice becomes validation.
    """
    offsets = _index_documents(tmp_path)
    order = list(range(len(offsets)))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(len(order) * val_ratio))
    val_ids = set(order[:n_val])

    train_chars = val_chars = 0
    train_docs = val_docs = 0
    with tmp_path.open("rb") as src, \
            train_path.open("w", encoding="utf-8") as f_train, \
            val_path.open("w", encoding="utf-8") as f_val:
        for doc_id, (start, end) in enumerate(offsets):
            src.seek(start)
            text = src.read(end - start).decode("utf-8")
            if not text.endswith("\n"):
                text += "\n"
            if doc_id in val_ids:
                f_val.write(text + "\n")
                val_docs += 1
                val_chars += len(text)
            else:
                f_train.write(text + "\n")
                train_docs += 1
                train_chars += len(text)

    return {
        "documents_train": train_docs,
        "documents_val": val_docs,
        "chars_train": train_chars,
        "chars_val": val_chars,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean raw Arabic JSONL corpus (streaming).")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed_v4"))
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--min-line-len", type=int, default=40)
    parser.add_argument("--min-arabic-letters", type=int, default=20)
    parser.add_argument("--min-arabic-ratio", type=float, default=0.55)
    parser.add_argument("--keep-diacritics", action="store_true")
    parser.add_argument("--unify-alef", action="store_true")
    parser.add_argument("--unify-yaa", action="store_true")
    parser.add_argument("--taa-marbuta-to-haa", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--keep-tmp", action="store_true",
                        help="keep the intermediate all-docs file (debugging)")
    args = parser.parse_args(argv)

    opts = CleanOptions(
        min_line_len=args.min_line_len,
        min_arabic_letters=args.min_arabic_letters,
        min_arabic_ratio=args.min_arabic_ratio,
        remove_diacritics=not args.keep_diacritics,
        unify_alef=args.unify_alef,
        unify_yaa=args.unify_yaa,
        taa_marbuta_to_haa=args.taa_marbuta_to_haa,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = args.out_dir / "_all_docs.tmp.txt"

    stats = clean_streaming(args.inputs, opts, tmp_path)
    print(
        f"[pass1 done] {stats['documents_total']} docs, "
        f"{stats['chars_kept'] / 1e6:.1f}M chars, dup_paras="
        f"{stats['paragraphs_dropped_duplicate']} — splitting ...",
        flush=True,
    )
    stats.update(split_streaming(
        tmp_path,
        args.out_dir / "train.txt",
        args.out_dir / "val.txt",
        args.val_ratio,
        args.seed,
    ))

    (args.out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[done] train={stats['documents_train']} docs ({stats['chars_train'] / 1e6:.1f}M chars) "
        f"val={stats['documents_val']} docs ({stats['chars_val'] / 1e6:.1f}M chars)",
        flush=True,
    )
    if not args.keep_tmp:
        tmp_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
