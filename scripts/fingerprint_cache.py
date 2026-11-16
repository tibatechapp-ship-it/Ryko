"""Fingerprint-cache + candidate-selection dup counter for corpus rebuild.

Pass 1 (--cache): stream train.txt/val.txt, write fp64 fingerprints to a raw
binary cache (8 bytes each, sorted not required — set built on load).

Pass 2 (--test): load the cache, stream a candidate raw JSONL (same cleaning
as augment_corpus), count dropped duplicates (cross + internal) exactly like
stream_clean_file would — WITHOUT writing anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ryko_lm.data.clean_corpus import CleanOptions, _clean_line  # noqa: E402
from ryko_lm.data.collect_shamela import list_page_files, select_books  # noqa: E402


def fold_fp(line: str) -> int:
    import re
    folded = re.sub(r"[^\u0600-\u06FFA-Za-z]", "", line)
    return int.from_bytes(hashlib.sha256(folded.encode("utf-8")).digest()[:8], "big")


def build_cache(paths: list[Path], cache_path: Path) -> int:
    n = 0
    with cache_path.open("wb") as out:
        for path in paths:
            with path.open(encoding="utf-8") as src:
                for line in src:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    out.write(fold_fp(line).to_bytes(8, "big"))
                    n += 1
                    if n % 5_000_000 == 0:
                        print(f"  fingerprinted {n:,}", flush=True)
    print(f"cache: {n:,} fingerprints -> {cache_path}")
    return n


def load_cache(cache_path: Path):
    """Load fingerprints as a SORTED numpy uint64 array (~265MB vs ~2GB set)."""
    import numpy as np
    arr = np.fromfile(cache_path, dtype=">u8").astype(np.uint64)
    return np.sort(arr)


def clean_doc_paragraphs(text: str, opts: CleanOptions) -> list[str]:
    paragraphs: list[str] = []
    for block in text.split("\n\n") if "\n\n" in text else [text]:
        for sub in block.splitlines():
            if not sub.strip():
                continue
            cleaned = _clean_line(sub, opts)
            if cleaned is not None:
                paragraphs.append(cleaned)
    return paragraphs


def count_dups(jsonl_path: Path, fps, opts: CleanOptions) -> dict:
    """Count dups against a sorted uint64 array — vectorized in batches.

    Semantics identical to the sequential counter: a paragraph is a duplicate
    iff its fingerprint is in the existing array OR was first seen earlier in
    this stream (first occurrence wins, later repeats count as dropped).
    """
    import numpy as np
    internal: set[int] = set()
    docs = chars = dup = 0
    batch: list[int] = []
    batch_chars = 0
    n_fps = len(fps)

    def flush() -> None:
        nonlocal dup, chars
        if not batch:
            return
        arr = np.array([fp for fp, _ in batch], dtype=np.uint64)
        lens = [ln for _, ln in batch]
        pos = np.searchsorted(fps, arr)
        pos_clipped = np.minimum(pos, n_fps - 1)
        existing = fps[pos_clipped] == arr
        dup += int(existing.sum())
        new_vals = arr[~existing]
        new_lens = [ln for ln, ex in zip(lens, existing.tolist()) if not ex]
        if new_vals.size:
            uniq, counts = np.unique(new_vals, return_counts=True)
            dup += int((counts - 1).sum())          # repeats within this batch
            # first occurrence of each unique new value:
            first_len: dict[int, int] = {}
            for v, ln in zip(new_vals.tolist(), new_lens):
                if v not in first_len:
                    first_len[v] = ln
            for v, ln in first_len.items():
                if v in internal:                    # seen in an earlier batch
                    dup += 1
                else:
                    internal.add(v)
                    chars += ln
        batch.clear()

    with jsonl_path.open(encoding="utf-8") as src:
        for raw in src:
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text = rec.get("text") or ""
            paragraphs = clean_doc_paragraphs(text, opts)
            if not paragraphs:
                continue
            docs += 1
            for para in paragraphs:
                batch.append((fold_fp(para), len(para)))
                if len(batch) >= 200_000:
                    flush()
    flush()
    return {"docs": docs, "chars": chars, "dup": dup}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--mode", choices=["cache", "test"], required=True)
    ap.add_argument("--corpus", nargs="*", type=Path, default=[])
    ap.add_argument("--shamela-select", type=str, default="4,144",
                    help="per_cat,max_books for a shamela candidate selection")
    ap.add_argument("--max-file-mb", type=float, default=100.0)
    ap.add_argument("--no-famous", action="store_true",
                    help="ignore FAMOUS_KEYWORDS in the selection")
    args = ap.parse_args(argv)

    opts = CleanOptions(40, 20, 0.55, True, False, False, False)

    if args.mode == "cache":
        build_cache(args.corpus, args.cache)
        return 0

    seen = load_cache(args.cache)
    print(f"loaded {len(seen):,} fingerprints", flush=True)
    per_cat, max_books = (int(x) for x in args.shamela_select.split(","))
    all_paths = list_page_files()
    if args.no_famous:
        import ryko_lm.data.collect_shamela as cs
        real_kw = cs.FAMOUS_KEYWORDS
        cs.FAMOUS_KEYWORDS = []
        sel = select_books(all_paths, per_cat, max_books)
        cs.FAMOUS_KEYWORDS = real_kw
    else:
        sel = select_books(all_paths, per_cat, max_books)
    print(f"selection per_cat={per_cat} max_books={max_books} cap={args.max_file_mb}MB "
          f"no_famous={args.no_famous}: {len(sel)} books", flush=True)
    tmp = Path("/tmp/candidate_shamela.jsonl")
    tmp.unlink(missing_ok=True)
    # fetch books using the committed collector (reuse its fetch machinery)
    from ryko_lm.data.collect_shamela import fetch_book, RESOLVE_BASE, _quote, FAMOUS_KEYWORDS  # noqa
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    n_docs = n_chars = n_dup = 0
    with tmp.open("w", encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = {pool.submit(fetch_book, b, int(args.max_file_mb * 1e6), False): b for b in sel}
            for fut, b in futures.items():
                rec = fut.result()
                if rec is None or "error" in rec:
                    continue
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # count after all fetched (order-independent counting)
    res = count_dups(tmp, seen, opts)
    print(f"RESULT per_cat={per_cat} max_books={max_books} cap={args.max_file_mb} "
          f"no_famous={args.no_famous}: "
          f"docs={res['docs']} chars={res['chars']/1e6:.1f}M dup={res['dup']}", flush=True)
    tmp.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
