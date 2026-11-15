"""Ingest pipelines: collector JSONL -> Knowledge Store.

Two formats are accepted (both produced by ``ryko_lm.data`` collectors):

* ``shamela`` — ``{"id": "shamela-12345", "book": "صحيح البخاري", "cat": "حديث",
  "pages": 8, "page_records": [{"page_num": 1, "body": "..."}, ...]}``
  (page-level provenance; older files without ``page_records`` fall back to
  splitting ``text`` on newlines with ``page_num=NULL``).
* ``openiti`` — ``{"id": "openiti-0000-000123", "text": "..."}`` (no page
  numbers; passages get ``page_num=NULL`` and an ordinal ``seq``).

Every ingest is idempotent: passage ids are deterministic
(``stable_id(book, page, seq)``) and in-book duplicates are skipped, so
re-running the same file is a no-op that reports the same counts.
"""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

from ryko_lm.knowledge import schema as sch
from ryko_lm.knowledge.store import KnowledgeStore

TARGET_CHUNK = 400   # chars — classical paragraphs are short
MIN_CHUNK = 80
MAX_CHUNK = 900

SHAMELA_SOURCE = {
    "source_id": "shamela",
    "name": "Shamela v4 (AuthenticIlm/Shamela4_Full_DB, HuggingFace mirror)",
    "homepage": "https://huggingface.co/datasets/AuthenticIlm/Shamela4_Full_DB",
    "license": "MIT declared by dataset publisher; editorial rights of individual editions remain with their holders",
    "rights_status": sch.RIGHTS_DECLARED_OPEN,
}

OPENITI_SOURCE = {
    "source_id": "openiti",
    "name": "OpenITI Corpus 2023.1.8 (mARkdown, HuggingFace mirror)",
    "homepage": "https://github.com/OpenITI",
    "license": "CC BY-SA 4.0 (OCR/transcription) over public-domain texts",
    "rights_status": sch.RIGHTS_PUBLIC_DOMAIN,
}

FIKR_SOURCE = {
    "source_id": "fikr",
    "name": "الفكر الإسلامي — مجموعة مختارة من كتب الفكر (مرآة الشاملة على HuggingFace، قائمة مُنَسَّقة)",
    "homepage": "https://huggingface.co/datasets/AuthenticIlm/Shamela4_Full_DB",
    "license": "MIT declared by dataset publisher; editorial rights of individual editions remain with their holders",
    "rights_status": sch.RIGHTS_DECLARED_OPEN,
}

ECONOM_SOURCE = {
    "source_id": "econom",
    "name": "الاقتصاد الإسلامي والحقوق — كتب المال والحق (مرآة الشاملة على HuggingFace، قائمة مُنَسَّقة)",
    "homepage": "https://huggingface.co/datasets/AuthenticIlm/Shamela4_Full_DB",
    "license": "MIT declared by dataset publisher; editorial rights of individual editions remain with their holders",
    "rights_status": sch.RIGHTS_DECLARED_OPEN,
}


def chunk_text(text: str, target: int = TARGET_CHUNK, min_len: int = MIN_CHUNK,
               max_len: int = MAX_CHUNK) -> list[str]:
    """Paragraph-accumulating chunker — deterministic, provenance-preserving.

    Paragraphs (lines) accumulate until ``target`` chars; a buffer is flushed
    once it reaches ``min_len`` and the next line would overshoot ``target``.
    Oversized single paragraphs are hard-split at word boundaries under
    ``max_len``.
    """
    chunks: list[str] = []
    buf = ""
    for para in (p.strip() for p in text.splitlines()):
        if not para:
            continue
        while len(para) > max_len:
            cut = para.rfind(" ", 0, max_len)
            if cut < min_len:
                cut = max_len
            head, para = para[:cut].strip(), para[cut:].strip()
            if buf:
                chunks.append(buf.strip())
                buf = ""
            chunks.append(head)
        if not para:
            continue
        candidate = f"{buf} {para}".strip() if buf else para
        if len(candidate) > target and len(buf) >= min_len:
            chunks.append(buf.strip())
            buf = para
        else:
            buf = candidate
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def ingest_shamela_jsonl(store: KnowledgeStore, jsonl_path: str | Path,
                         classify: bool = True,
                         register_source: bool = True) -> dict:
    src = SHAMELA_SOURCE
    if register_source:
        store.add_source(src["source_id"], src["name"], src["license"],
                         src["rights_status"], src["homepage"])
    books = passages = dups = chars = 0
    with Path(jsonl_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            book_id = rec["id"]
            store.add_book(book_id, src["source_id"], rec.get("book") or book_id,
                           category=rec.get("cat"),
                           rights_status=src["rights_status"],
                           meta={"raw_pages": rec.get("pages")})
            books += 1
            page_records = rec.get("page_records")
            if page_records is None:
                # legacy joined-text fallback: no page provenance
                page_records = [{"page_num": None, "body": rec.get("text", "")}]
            for page in page_records:
                body = (page.get("body") or "").strip()
                if not body:
                    continue
                page_num = page.get("page_num")
                base = store.next_seq(book_id, page_num)
                for i, chunk in enumerate(chunk_text(body)):
                    pid, created = store.add_passage(
                        book_id, chunk, page_num=page_num,
                        seq=base + i, classify=classify)
                    if created:
                        passages += 1
                        chars += len(chunk)
                    else:
                        dups += 1
    return {"format": "shamela", "books": books, "passages": passages,
            "duplicates_skipped": dups, "chars": chars}


def ingest_fikr_jsonl(store: KnowledgeStore, jsonl_path: str | Path,
                      classify: bool = True,
                      register_source: bool = True) -> dict:
    """Ingest the curated Islamic-Thought collection (``collect_fikr`` output)."""
    return _ingest_curated_jsonl(store, jsonl_path, FIKR_SOURCE, classify,
                                 register_source)


def ingest_econom_jsonl(store: KnowledgeStore, jsonl_path: str | Path,
                        classify: bool = True,
                        register_source: bool = True) -> dict:
    """Ingest the curated Economics-and-Rights collection
    (``collect_fikr --manifest data/knowledge/econom_manifest.json`` output).

    Same page-level provenance as the fikr format, but books register under
    the dedicated ``econom`` source with ``category=اقتصاد-حقوق/<section>``
    so the money-and-rights layer stays queryable as one domain.
    """
    return _ingest_curated_jsonl(store, jsonl_path, ECONOM_SOURCE, classify,
                                 register_source)


def _ingest_curated_jsonl(store: KnowledgeStore, jsonl_path: str | Path,
                          src: dict, classify: bool = True,
                          register_source: bool = True) -> dict:
    """Shared ingest for curated layers (fikr / econom): page-level
    provenance, dedicated source, author / school metadata so each layer
    stays queryable as one domain.

    ``jsonl_path`` may be a JSONL file path or an in-memory list of already
    parsed records (the ``build_*_db.py`` pipelines pass filtered records
    directly to avoid a re-read of the raw file).
    """
    if register_source:
        store.add_source(src["source_id"], src["name"], src["license"],
                         src["rights_status"], src["homepage"])
    books = passages = dups = chars = 0
    if isinstance(jsonl_path, (list, tuple)):
        lines = (json.dumps(r, ensure_ascii=False) for r in jsonl_path)
    else:
        lines = Path(jsonl_path).open(encoding="utf-8")
    with closing(lines) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            book_id = rec["id"]
            author = rec.get("author")
            author_id = None
            if author:
                author_id = sch.stable_id("a", author)
                store.add_author(author_id, author,
                                 meta={"school": rec.get("school")})
            store.add_book(book_id, src["source_id"], rec.get("book") or book_id,
                           author_id=author_id,
                           category=rec.get("cat") or "فكر",
                           rights_status=src["rights_status"],
                           meta={"raw_pages": rec.get("pages"),
                                 "author": author, "school": rec.get("school")})
            books += 1
            page_records = rec.get("page_records") or [{"page_num": None,
                                                        "body": rec.get("text", "")}]
            for page in page_records:
                body = (page.get("body") or "").strip()
                if not body:
                    continue
                page_num = page.get("page_num")
                base = store.next_seq(book_id, page_num)
                for i, chunk in enumerate(chunk_text(body)):
                    pid, created = store.add_passage(
                        book_id, chunk, page_num=page_num,
                        seq=base + i, classify=classify)
                    if created:
                        passages += 1
                        chars += len(chunk)
                    else:
                        dups += 1
    return {"format": src["source_id"], "books": books, "passages": passages,
            "duplicates_skipped": dups, "chars": chars}


def ingest_openiti_jsonl(store: KnowledgeStore, jsonl_path: str | Path,
                         classify: bool = True,
                         register_source: bool = True) -> dict:
    src = OPENITI_SOURCE
    if register_source:
        store.add_source(src["source_id"], src["name"], src["license"],
                         src["rights_status"], src["homepage"])
    books = passages = dups = chars = 0
    with Path(jsonl_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            book_id = rec["id"]
            store.add_book(book_id, src["source_id"], book_id,
                           rights_status=src["rights_status"])
            books += 1
            base = store.next_seq(book_id, None)
            for i, chunk in enumerate(chunk_text(rec.get("text", ""))):
                pid, created = store.add_passage(
                    book_id, chunk, page_num=None, seq=base + i, classify=classify)
                if created:
                    passages += 1
                    chars += len(chunk)
                else:
                    dups += 1
    return {"format": "openiti", "books": books, "passages": passages,
            "duplicates_skipped": dups, "chars": chars}
