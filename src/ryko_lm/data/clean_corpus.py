"""Clean and normalize the raw Arabic corpus into a training-ready text file.

Pipeline (each step is reported in the stats sidecar):

1.  Unicode NFKC — folds Arabic presentation forms (ﻻ -> لا) and legacy glyphs
    into their canonical code points.
2.  Strip markup: HTML tags, URLs, e-mails, wiki residue (``[[..]]`` kept as
    inner text, ``[1]``-style citations, file/image namespaces).
3.  Arabic orthography cleanup:
      - remove tatweel            (ـ U+0640)
      - remove diacritics         (harakat U+064B..U+065F, dagger alif U+0670,
                                   Quranic annotation U+06D6..U+06ED)
    Aggressive unifications (alef variants, taa marbuta, alif maqsura) are
    *optional* and off by default: they compress the corpus but erase
    orthographic information.  See docs/ARABIC_TOKENIZATION.md.
4.  Line filtering: minimum length, minimum Arabic-letter ratio, drop lines
    dominated by Latin/digits/symbols.
5.  Exact-duplicate removal at paragraph level (hash on normalized text).
6.  Split train/val (paragraphs, seeded shuffle).

Output: one paragraph per line; a blank line separates documents.
Usage:
    python -m ryko_lm.data.clean_corpus \
        --inputs data/raw/wiki.jsonl data/raw/tashkeela.jsonl \
        --out-dir data/processed --val-ratio 0.01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
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


def _clean_line(line: str, opts: "CleanOptions") -> str | None:
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
    """Hash a paragraph after aggressive whitespace/punctuation-insensitive fold."""
    folded = re.sub(r"[^\u0600-\u06FFA-Za-z]", "", line)
    return hashlib.sha256(folded.encode("utf-8")).hexdigest()


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


def clean_file(path: Path, opts: CleanOptions) -> list[list[str]]:
    """Clean one JSONL input file; return a list of documents (lists of paragraphs)."""
    docs: list[list[str]] = []
    with path.open(encoding="utf-8") as src:
        for raw in src:
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text = record.get("text") or ""
            paragraphs: list[str] = []
            for block in re.split(r"\n\s*\n", text):
                lines = [l for l in (block.split("\n")) if l.strip()]
                for line in lines:
                    cleaned = _clean_line(line, opts)
                    if cleaned is not None:
                        paragraphs.append(cleaned)
            if paragraphs:
                docs.append(paragraphs)
    return docs


def deduplicate(documents: list[list[str]], seen: set[str]) -> tuple[list[list[str]], int]:
    """Drop duplicate paragraphs (mutates ``seen``) and duplicate documents."""
    kept_docs: list[list[str]] = []
    dropped = 0
    for paragraphs in documents:
        unique_paras: list[str] = []
        para_hashes: list[str] = []
        for para in paragraphs:
            fp = _fingerprint(para)
            if fp in seen:
                dropped += 1
                continue
            seen.add(fp)
            unique_paras.append(para)
            para_hashes.append(fp)
        if len(unique_paras) >= 1:
            kept_docs.append(unique_paras)
    return kept_docs, dropped


def write_split(documents: list[list[str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as out:
        for paragraphs in documents:
            out.write("\n".join(paragraphs) + "\n\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean raw Arabic JSONL corpus.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--min-line-len", type=int, default=40)
    parser.add_argument("--min-arabic-letters", type=int, default=20)
    parser.add_argument("--min-arabic-ratio", type=float, default=0.55)
    parser.add_argument("--keep-diacritics", action="store_true")
    parser.add_argument("--unify-alef", action="store_true")
    parser.add_argument("--unify-yaa", action="store_true")
    parser.add_argument("--taa-marbuta-to-haa", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
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

    raw_chars = 0
    all_docs: list[list[str]] = []
    per_source: dict[str, int] = {}
    for path in args.inputs:
        docs = clean_file(path, opts)
        per_source[path.name] = len(docs)
        raw_chars += sum(len(p) for doc in docs for p in doc)
        all_docs.extend(docs)
        print(f"[clean] {path.name}: {len(docs)} documents", flush=True)

    seen: set[str] = set()
    all_docs, dropped_paras = deduplicate(all_docs, seen)

    rng = random.Random(args.seed)
    rng.shuffle(all_docs)
    n_val = max(1, int(len(all_docs) * args.val_ratio))
    val_docs, train_docs = all_docs[:n_val], all_docs[n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_split(train_docs, args.out_dir / "train.txt")
    write_split(val_docs, args.out_dir / "val.txt")

    train_chars = sum(len(p) for doc in train_docs for p in doc)
    val_chars = sum(len(p) for doc in val_docs for p in doc)
    stats = {
        "inputs": per_source,
        "documents_total": len(all_docs),
        "documents_train": len(train_docs),
        "documents_val": len(val_docs),
        "paragraphs_dropped_duplicate": dropped_paras,
        "chars_raw_kept": raw_chars,
        "chars_train": train_chars,
        "chars_val": val_chars,
        "options": vars(opts),
    }
    (args.out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[done] train={len(train_docs)} docs ({train_chars / 1e6:.1f}M chars) "
        f"val={len(val_docs)} docs ({val_chars / 1e3:.0f}K chars) "
        f"dup_paras={dropped_paras}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
