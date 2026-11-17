"""Turath layer — the wide stratified selection of the Knowledge Core.

The layer (since v1.16.0) imports the SAME wide stratified selection the
corpus pipeline collected for v5.2 (``collect_shamela --per-cat 12
--max-books 600 --max-file-mb 3 --keep-pages``, famous-first per category)
into the Knowledge Core under the dedicated ``turath`` source, so retrieval,
cite-audit and ``/knowledge/ask`` reach far beyond the two منقحة layers.

Two tiers, same policy as fikr/econom:

* **core** — the committed store stays under GitHub's 100 MB blob limit:
  a *budgeted-breadth* selection adds books until a fixed TEXT-character
  budget is spent.  Deterministic: categories sorted, each category's books
  ordered by ``(size, id)``, one round-robin pick per category per pass —
  every category that can fit gets at least one book before any gets two.
* **full** — the local store carries every collected record; its receipt is
  committed, the ~hundreds-of-MB file itself is rebuildable
  (``make turath-collect && make turath-build-full``).

No claims are attributed in this layer: attributed ideas stay the exclusive
discipline of the two منقحة layers (fikr / econom) — the turath layer is a
retrieval-and-citation layer, and it says so in its manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

from ryko_lm.knowledge.ingest import TURATH_SOURCE, turath_book_id

MANIFEST_KIND = "ryko-turath-manifest"
PROTOCOL = "ryko-turath/1"


def record_size(rec: dict) -> int:
    """TEXT characters a wide record will contribute (page records or joined)."""
    page_records = rec.get("page_records")
    if page_records is None:
        return len(rec.get("text") or "")
    return sum(len(p.get("body") or "") for p in page_records)


def select_turath_core(records: list[dict], existing_book_ids: set[str],
                       budget_chars: int,
                       spent_chars: int = 0) -> tuple[list[dict], dict]:
    """Budgeted-breadth selection — deterministic, idempotence-aware.

    ``budget_chars`` is the TOTAL budget of the committed turath layer (a
    policy constant, not a per-run allowance): ``spent_chars`` counts TEXT
    characters already committed under the ``turath`` source, so re-running
    a completed build selects nothing.  To grow coverage in a later release,
    raise the budget — the manifest documents the constant either way.

    Books already present in the store (by namespaced id) are skipped so a
    partially-built store continues from where it stopped.  Returns
    ``(selected_records, report)`` with every policy number the manifest
    publishes.
    """
    by_cat: dict[str, list[dict]] = {}
    skipped_existing = 0
    for rec in records:
        if turath_book_id(rec.get("id", "")) in existing_book_ids:
            skipped_existing += 1
            continue
        by_cat.setdefault(rec.get("cat") or "", []).append(rec)

    cats = sorted(by_cat)
    for cat in cats:
        by_cat[cat].sort(key=lambda r: (record_size(r), turath_book_id(r.get("id", ""))))

    selected: list[dict] = []
    budget_left = budget_chars - spent_chars
    rounds = 0
    exhausted = {cat: 0 for cat in cats}
    while budget_left > 0:
        progressed = False
        for cat in cats:
            lst = by_cat[cat]
            if exhausted[cat] >= len(lst):
                continue
            rec = lst[exhausted[cat]]
            size = record_size(rec)
            if size <= budget_left:
                selected.append(rec)
                budget_left -= size
                exhausted[cat] += 1
                progressed = True
            else:
                # every remaining book of this category is even bigger —
                # the category can no longer contribute, stop visiting it
                exhausted[cat] = len(lst)
        if not progressed:
            break
        rounds += 1

    report = {
        "policy": "budgeted-breadth: category round-robin (sorted), smallest "
                  "(size, id) fitting book first — every category that fits "
                  "gets a book before any category gets a second one",
        "budget_chars": budget_chars,
        "budget_spent_chars": budget_chars - budget_left,
        "budget_new_chars": (budget_chars - budget_left) - spent_chars,
        "budget_spent_before_chars": spent_chars,
        "records_available": len(records),
        "records_skipped_existing": skipped_existing,
        "records_selected": len(selected),
        "categories_available": len(by_cat),
        "categories_covered": len({(r.get("cat") or "") for r in selected}),
        "rounds": rounds,
    }
    return selected, report


def build_manifest(book_rows: list[dict], report: dict, collector: dict) -> dict:
    """Layer manifest (kind ``ryko-turath-manifest``): policy, selection
    report, per-book provenance summary.

    ``book_rows`` come from the STORE (``SELECT … FROM books WHERE
    source_id='turath'`` joined with passage char sums) — the manifest
    reflects the *cumulative layer state*, not one run's additions, so a
    completed-build re-run rewrites it byte-identically.  Rows are sorted
    by namespaced id, making the JSON deterministic.
    """
    books = []
    for row in sorted(book_rows, key=lambda r: r["book_id"]):
        raw_id = row["book_id"]
        dataset_id = ("shamela-" + raw_id[len("turath-"):]
                      if raw_id.startswith("turath-") else raw_id)
        books.append({
            "book_id": row["book_id"],
            "dataset_id": dataset_id,
            "title": row["title"],
            "category": row["category"],
            "raw_pages": row["raw_pages"],
            "chars": row["chars"],
        })
    return {
        "kind": MANIFEST_KIND,
        "protocol": PROTOCOL,
        "source_id": TURATH_SOURCE["source_id"],
        "source_name": TURATH_SOURCE["name"],
        "homepage": TURATH_SOURCE["homepage"],
        "license": TURATH_SOURCE["license"],
        "rights_status": TURATH_SOURCE["rights_status"],
        "collector": collector,
        "selection": report,
        "claims_policy": "none — attributed ideas stay the exclusive discipline "
                         "of the منقحة layers (fikr / econom); turath is a "
                         "retrieval-and-citation layer",
        "counts": {"books": len(books),
                   "chars": sum(b["chars"] for b in books)},
        "books": books,
    }


def write_manifest(manifest: dict, out: Path) -> None:
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
