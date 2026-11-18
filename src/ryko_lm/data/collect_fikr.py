"""Collect the Islamic-Thought (الفكر الإسلامي) key books from the Shamela v4 mirror.

Unlike :mod:`ryko_lm.data.collect_shamela` (a stratified corpus sampler), this
collector resolves an explicit, curated list of *thought* titles from
``data/knowledge/fikr_manifest.json`` against the dataset file listing and
fetches exactly those books with page-level provenance, so the Knowledge Core
can ingest them under the ``fikr`` source.

Resumable via ``data/raw/fikr.progress.json``.

Usage:
    python -m ryko_lm.data.collect_fikr --manifest data/knowledge/fikr_manifest.json \
        --out data/raw/fikr_shamela.jsonl --max-file-mb 80
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASET_API = "https://huggingface.co/api/datasets/AuthenticIlm/Shamela4_Full_DB"
RESOLVE_BASE = "https://huggingface.co/datasets/AuthenticIlm/Shamela4_Full_DB/resolve/main/"

# Shamela mirror bodies carry structural markup on heading pages
# (<span data-type="title" id=toc-N>…</span>) — strip it at collection time
# so the Knowledge Core ingests clean text (~2.6% of raw pages).
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"[ \t\u00a0]+")


def strip_markup(body: str) -> str:
    cleaned = HTML_TAG_RE.sub(" ", body)
    return MULTISPACE_RE.sub(" ", cleaned).strip()


def _safe_get(url: str, timeout: int = 120, retries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ryko-corpus/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed {url[:100]}: {last}")


def list_page_files() -> list[str]:
    meta = json.loads(_safe_get(DATASET_API))
    return [s["rfilename"] for s in meta.get("siblings", [])
            if s["rfilename"].endswith("pages.jsonl")]


def resolve_paths(manifest: dict, listing: list[str]) -> tuple[list[tuple[str, dict]], list[dict]]:
    """Match every ``available`` manifest book to its mirror path.

    Returns (resolved[(path, book_record)], unresolved[book_record]); each
    resolved book record carries the manifest's ``layer_label`` so fetched
    records are categorized under the right curated layer.
    """
    cat_label = manifest.get("layer_label", "فكر")
    resolved: list[tuple[str, dict]] = []
    unresolved: list[dict] = []
    for book in manifest.get("books", []):
        if book.get("status") != "available" or not book.get("shamela_path"):
            continue
        hits = [p for p in listing if book["shamela_path"] in p]
        if hits:
            book = dict(book)
            book["cat_label"] = cat_label
            resolved.append((sorted(hits)[0], book))
        else:
            unresolved.append(book)
    return resolved, unresolved


def fetch_book(path: str, book: dict, max_bytes: int) -> dict | None:
    try:
        blob = _safe_get(RESOLVE_BASE + urllib.request.quote(path, safe="/"), timeout=180)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:120], "path": path}
    if len(blob) > max_bytes:
        return None
    pages: list[str] = []
    page_records: list[dict] = []
    for idx, line in enumerate(blob.decode("utf-8", "replace").splitlines(), 1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        body = (rec.get("body") or "").strip()
        if not body:
            continue
        pages.append(body)
        page_num = rec.get("page_num", idx)
        try:
            page_num = int(page_num)
        except (TypeError, ValueError):
            page_num = idx
        page_records.append({"page_num": page_num, "body": strip_markup(body)})
    text = "\n".join(pages).replace("\r", "\n")
    if len(text) < 2000:
        return None
    parts = path.split("/")
    return {
        "id": f"shamela-{parts[1].split('__')[0]}",
        "book": book.get("title") or parts[1].split("__", 1)[1],
        "cat": f"{book.get('cat_label', 'فكر')}/{book['school']}" if book.get("school") else book.get("cat_label", "فكر"),
        "pages": len(pages),
        "author": book.get("author"),
        "school": book.get("school"),
        "page_records": page_records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect curated Islamic-Thought books.")
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/knowledge/fikr_manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/fikr_shamela.jsonl"))
    parser.add_argument("--progress", type=Path, default=Path("data/raw/fikr.progress.json"))
    parser.add_argument("--max-file-mb", type=float, default=80.0,
                        help="per-book size cap (تفسير المنار and إعلام الموقعين are large)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    kind = manifest.get("kind", "ryko-fikr-manifest")
    tag = {"ryko-fikr-manifest": "fikr",
           "ryko-econom-manifest": "econom",
           "ryko-lugha-manifest": "lugha"}.get(kind, kind.removeprefix("ryko-").removesuffix("-manifest"))
    print(f"[{tag}] manifest v{manifest.get('version')} — "
          f"{len(manifest['schools'])} schools, {len(manifest['ideas'])} ideas", flush=True)

    listing = list_page_files()
    resolved, unresolved = resolve_paths(manifest, listing)
    print(f"[{tag}] resolved {len(resolved)} books from the mirror, "
          f"{len(unresolved)} pending external sources", flush=True)

    # progress v2: {"done": {path: {"ok": bool, "mb": cap}}}.  Successful
    # books never re-download on resume; books that exceeded a smaller byte
    # budget (ok=False) retry automatically once a bigger cap is passed, so
    # staged passes (small matn first, then xlarge dictionaries) share one
    # progress file without duplicate records.
    done: dict[str, dict] = {}
    if args.progress.exists():
        try:
            raw_done = json.loads(args.progress.read_text())["done"]
            if isinstance(raw_done, list):  # v1 legacy format
                done = {p: {"ok": True, "mb": 0} for p in raw_done}
            else:
                done = {k: v for k, v in raw_done.items()}
        except Exception:  # noqa: BLE001
            done = {}
    max_bytes = int(args.max_file_mb * 1e6)

    def _todo() -> list[tuple[str, dict]]:
        todo = []
        for p, b in resolved:
            state = done.get(p)
            if state and state.get("ok"):
                continue
            if state and not state.get("ok") and state.get("mb", 0) >= args.max_file_mb:
                continue  # already tried at this cap or bigger — skip
            todo.append((p, b))
        return todo

    todo = _todo()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_chars, total_books, errors = 0, 0, 0
    t0 = time.time()

    with args.out.open("a", encoding="utf-8") as sink, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_book, p, b, max_bytes): p for p, b in todo}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                rec = fut.result()
            except Exception:  # noqa: BLE001
                rec = None
            if rec and "error" not in rec:
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sink.flush()
                total_books += 1
                total_chars += sum(len(p["body"]) for p in rec["page_records"])
                done[path] = {"ok": True, "mb": args.max_file_mb}
            else:
                # oversize (rec=None) or download error: record with the tried
                # cap so a bigger --max-file-mb pass retries it
                done[path] = {"ok": False, "mb": args.max_file_mb}
                if rec:
                    errors += 1
                    print(f"[{tag}] ERROR {rec['path']}: {rec['error']}", flush=True)
            # incremental checkpoint: a killed run resumes without re-downloads
            args.progress.write_text(json.dumps({"done": done}, ensure_ascii=False),
                                     encoding="utf-8")
            print(f"[{tag}] +{total_books}/{len(todo)} books, "
                  f"{total_chars / 1e6:.1f}M chars, err={errors} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    args.progress.write_text(json.dumps({"done": done}, ensure_ascii=False), encoding="utf-8")
    deferred = sum(1 for v in done.values() if not v.get("ok"))
    print(f"[done:{tag}] +{total_books} books, +{total_chars / 1e6:.1f}M chars, "
          f"errors={errors}, deferred-by-budget={deferred} -> {args.out}", flush=True)
    return 0 if errors == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
