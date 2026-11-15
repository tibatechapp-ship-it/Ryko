"""Collect selected books from the Shamela v4 corpus (HuggingFace mirror).

The ``AuthenticIlm/Shamela4_Full_DB`` dataset stores one ``pages.jsonl`` per
book (8,589 books, 41 categories).  Each page record carries ``book_id``,
``page_num`` and ``body`` — full page-level provenance for free.

This collector downloads a *stratified selection*: for every category it picks
titles matching a curated list of famous works first (Bukhari, Tabari, lisan
al-Arab, ...) and fills the remaining slots from the rest, honouring a
per-file size cap.  Records keep book id / category / title so the knowledge
layer can attribute passages later.

Resumable via ``data/raw/shamela.progress.json``.

Usage:
    python -m ryko_lm.data.collect_shamela --per-cat 4 \
        --max-file-mb 15 --out data/raw/shamela.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASET_API = "https://huggingface.co/api/datasets/AuthenticIlm/Shamela4_Full_DB"
RESOLVE_BASE = "https://huggingface.co/datasets/AuthenticIlm/Shamela4_Full_DB/resolve/main/"

# Famous works (title keywords) from the RYKO heritage map — picked first.
FAMOUS_KEYWORDS = [
    "البخاري", "صحيح-مسلم", "الترمذي", "أبو-داود", "النسائي", "ابن-ماجه",
    "موطأ", "مسند-أحمد", "الداودي", "البداية-والنهاية", "الطبري", "الكامل-في-التاريخ",
    "سير-أعلام", "تاريخ-دمشق", "لسان-العرب", "تاج-العروس", "القانون", "الحاوي",
    "المغني", "المدونة", "الأم-للشافعي", "المحلى", "الهداية", "إحياء-علوم",
    "مدارج-السالكين", "الفتوحات", "الأسفار", "تهافت", "فصل-المقال", "المقدمة-لابن-خلدون",
    "الجامع-لأحكام", "مفاتيح-الغيب", "التحرير-والتنوير", "تفسير-ابن-كثير", "القرطبي",
    "نخبة-الفكر", "تهذيب-التهذيب", "الأشباه", "مجلة-الأحكام", "الرحبية", "الآجرومية",
    "ألفية-ابن-مالك", "الشاطبية", "الجزرية", "ثلاثة-الأصول", "كشف-الشبهات",
    "دعائم-الإسلام", "جواهر-الكلام", "شرح-النيل", "الأزهار", "اختلاف-الفقهاء",
]


def _safe_get(url: str, timeout: int = 60, retries: int = 3) -> bytes:
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


def _quote(path: str) -> str:
    return urllib.request.quote(path, safe="/")


def list_page_files() -> list[str]:
    meta = json.loads(_safe_get(DATASET_API))
    return [s["rfilename"] for s in meta.get("siblings", [])
            if s["rfilename"].endswith("pages.jsonl")]


def select_books(paths: list[str], per_cat: int, max_books: int) -> list[str]:
    cats: dict[str, list[str]] = {}
    for p in paths:
        cat = p.split("/")[0]
        cats.setdefault(cat, []).append(p)

    chosen: list[str] = []
    for cat in sorted(cats):
        items = cats[cat]
        famous = [p for p in items
                  if any(kw in p for kw in FAMOUS_KEYWORDS)]
        others = [p for p in items if p not in set(famous)]
        picks = (famous[:per_cat] + others[:per_cat])[:per_cat]
        chosen.extend(picks)
    return chosen[:max_books]


def fetch_book(path: str, max_bytes: int) -> dict | None:
    try:
        blob = _safe_get(RESOLVE_BASE + _quote(path), timeout=90)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:120], "path": path}
    if len(blob) > max_bytes:
        return None
    pages: list[str] = []
    for line in blob.decode("utf-8", "replace").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        body = (rec.get("body") or "").strip()
        if body:
            pages.append(body)
    text = "\n".join(pages).replace("\r", "\n")
    if len(text) < 3000:
        return None
    parts = path.split("/")
    return {
        "id": f"shamela-{parts[1].split('__')[0]}",
        "book": parts[1].split("__", 1)[1] if "__" in parts[1] else parts[1],
        "cat": parts[0],
        "pages": len(pages),
        "text": text,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect selected Shamela books.")
    parser.add_argument("--out", type=Path, default=Path("data/raw/shamela.jsonl"))
    parser.add_argument("--progress", type=Path, default=Path("data/raw/shamela.progress.json"))
    parser.add_argument("--per-cat", type=int, default=4)
    parser.add_argument("--max-books", type=int, default=200)
    parser.add_argument("--max-file-mb", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    paths = list_page_files()
    books = select_books(paths, args.per_cat, args.max_books)
    print(f"[shamela] {len(paths)} books available, selected {len(books)}", flush=True)

    done: set[str] = set()
    if args.progress.exists():
        try:
            done = set(json.loads(args.progress.read_text())["done"])
        except Exception:  # noqa: BLE001
            done = set()
    todo = [b for b in books if b not in done]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = int(args.max_file_mb * 1e6)
    total_chars, total_books, errors = 0, 0, 0
    t0 = time.time()

    with args.out.open("a", encoding="utf-8") as sink, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_book, b, max_bytes): b for b in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            book_path = futures[fut]
            try:
                rec = fut.result()
            except Exception:  # noqa: BLE001
                rec = None
            if rec and "error" not in rec:
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_books += 1
                total_chars += len(rec["text"])
            elif rec:
                errors += 1
            done.add(book_path)
            if n % 25 == 0:
                args.progress.write_text(
                    json.dumps({"done": sorted(done)}), encoding="utf-8")
                print(f"[shamela] {n}/{len(todo)} done, "
                      f"{total_chars / 1e6:.1f}M chars, err={errors}", flush=True)

    args.progress.write_text(json.dumps({"done": sorted(done)}), encoding="utf-8")
    print(
        f"[done] +{total_books} books, +{total_chars / 1e6:.1f}M chars, "
        f"errors={errors} in {time.time() - t0:.0f}s -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
