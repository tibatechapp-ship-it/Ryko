#!/usr/bin/env python3
"""Build the linguistics SFT pack (حزمة التدريب اللغوي — مكوّن الحزمة المجمّعة v1.26.0).

Mirrors the rb-ling probe families (``scripts/build_ling_probes.py``) as
SFT prompt/completion pairs built from the committed Knowledge Core, while
keeping every probe case strictly held out (leakage check is fatal):

* ``نسبة-المؤلف``  — who wrote a committed lugha book (probe ``rb-ling-author``);
* ``باب-اللغة``    — which باب a committed lugha book belongs to
  (probe ``rb-ling-section``);
* ``نسبة-القاعدة-اللغوية`` — who authored a curated linguistic rule/definition
  (probe ``rb-ling-origin``).

``نسبة-الكتاب-اللغوي`` (probe ``rb-ling-book``) needs lugha *passages*, which
are not committed under the 100MiB wall — that family stays held-out and is
trained only indirectly (author/section/replay).  This is documented in the
manifest, not hidden.

Deterministic: fixed seed, stable order — same store + same probes ⇒
byte-identical output.  Every case is self-checked against the store
(options/gold derived from stored rows, never guessed).

Usage::

    PYTHONPATH=src python scripts/build_ling_sft.py \
        --store data/knowledge/ryko.sqlite \
        --probes docs/benchmarks/ling_probes.jsonl \
        --out-dir data/training
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ryko_lm.knowledge.store import KnowledgeStore  # noqa: E402

SEED = 1337

SECTION_LABELS: tuple[tuple[str, str], ...] = (
    ("nahw", "النحو"),
    ("sarf", "الصرف"),
    ("balagha", "البلاغة"),
    ("maajim", "المعاجم"),
    ("fiqh-lugha", "فقه اللغة واللسانيات"),
    ("adab", "الأدب"),
    ("munhaj", "المقررات والمناهج"),
)
_SECTION_LABEL_MAP = dict(SECTION_LABELS)

SCHEMA = "ryko-ling-sft/1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm(text: str) -> str:
    return re.sub(r"[\s\u064B-\u0652\u0670«»(){}.,،؛:!?\-—]+", "", text)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _probe_exclusions(probes_path: Path) -> dict:
    claim_ids, book_ids, passage_ids, quotes = set(), set(), set(), set()
    for line in probes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        meta = case.get("meta", {})
        if meta.get("claim_id"):
            claim_ids.add(meta["claim_id"])
        if meta.get("book_id"):
            book_ids.add(meta["book_id"])
        if meta.get("passage_id"):
            passage_ids.add(meta["passage_id"])
        for quoted in re.findall(r"«(.+?)»", case.get("question", ""), re.S):
            quotes.add(_norm(quoted))
    return {"claim_ids": claim_ids, "book_ids": book_ids,
            "passage_ids": passage_ids, "quotes": quotes,
            "title_norms": {q for q in quotes if len(q) < 25}}


def _lugha_books(store: KnowledgeStore) -> list[dict]:
    rows = store.conn.execute(
        "SELECT b.book_id, b.title_ar, b.category, a.name_ar AS author "
        "FROM books b LEFT JOIN authors a ON a.author_id = b.author_id "
        "WHERE b.source_id='lugha'").fetchall()
    out = []
    for r in rows:
        title = " ".join((r["title_ar"] or "").split())
        if not title:
            continue
        cat = r["category"] or ""
        section = cat.rsplit("/", 1)[1] if "/" in cat else None
        out.append({"book_id": r["book_id"], "title": title,
                    "author": (r["author"] or "").strip(),
                    "section": section if section in _SECTION_LABEL_MAP else None})
    return out


def _lugha_claims(store: KnowledgeStore) -> list[dict]:
    rows = store.conn.execute(
        "SELECT claim_id, claim_text, meta FROM claims").fetchall()
    out = []
    for r in rows:
        meta = json.loads(r["meta"]) if r["meta"] else {}
        if meta.get("layer") == "lugha" and meta.get("originator"):
            out.append({"claim_id": r["claim_id"], "text": r["claim_text"],
                        "originator": meta["originator"],
                        "school": meta.get("school")})
    return out


def build(store: KnowledgeStore, probes_path: Path, seed: int = SEED) -> dict:
    excl = _probe_exclusions(probes_path)
    rng = random.Random(seed)

    rows: list[dict] = []
    excluded_books = 0
    excluded_claims = 0
    excluded_title_collisions = 0

    books = _lugha_books(store)
    for b in sorted(books, key=lambda x: x["book_id"]):
        if b["book_id"] in excl["book_ids"]:
            excluded_books += 1
            continue
        if _norm(b["title"]) in excl["title_norms"]:
            excluded_title_collisions += 1
            continue
        base = {"schema": SCHEMA, "provenance": {"book_id": b["book_id"],
                                                 "section": b["section"]}}
        if b["author"]:
            rows.append({**base,
                         "id": f"ling-sft-author-{b['book_id']}",
                         "family": "نسبة-المؤلف",
                         "prompt": ("من مؤلف «" + b["title"]
                                    + "» من كتب طبقة اللغة المثبتة؟\nالجواب:"),
                         "completion": " " + b["author"]})
        if b["section"]:
            rows.append({**base,
                         "id": f"ling-sft-section-{b['book_id']}",
                         "family": "باب-اللغة",
                         "prompt": ("إلى أي باب من أبواب طبقة اللغة يعود «"
                                    + b["title"] + "»؟\nالجواب:"),
                         "completion": " " + _SECTION_LABEL_MAP[b["section"]]})

    claims = _lugha_claims(store)
    for c in sorted(claims, key=lambda x: x["claim_id"]):
        if c["claim_id"] in excl["claim_ids"]:
            excluded_claims += 1
            continue
        rows.append({
            "schema": SCHEMA,
            "id": f"ling-sft-origin-{c['claim_id']}",
            "family": "نسبة-القاعدة-اللغوية",
            "provenance": {"claim_id": c["claim_id"], "school": c["school"]},
            "prompt": ("من صاحب هذه القاعدة/التعريف في علم العربية؟\n«"
                       + c["text"] + "»\nالجواب:"),
            "completion": " " + c["originator"],
        })

    # Fatal leakage self-check.  Long probe quotes (claim texts, passage
    # excerpts, >=25 normalized chars) must not appear as substrings; short
    # quotes are book titles — only an EXACT quoted-title match counts
    # (a substring rule would false-positive on titles that merely contain
    # another title, e.g. «الكتاب» inside «الصناعتين الكتابة والشعر»).
    leaks: list[str] = []
    for row in rows:
        quoted = [_norm(q) for q in re.findall(r"«(.+?)»", row["prompt"], re.S)]
        np_ = _norm(row["prompt"])
        for q in excl["quotes"]:
            if not q:
                continue
            if len(q) >= 25:
                if q in np_:
                    leaks.append(f"quote-leak: {row['id']}")
                    break
            elif any(q == nq for nq in quoted):
                leaks.append(f"title-leak: {row['id']}")
                break
        prov = json.dumps(row.get("provenance", {}), ensure_ascii=False)
        for cid in excl["claim_ids"]:
            if cid and cid in prov:
                leaks.append(f"claim-id-leak: {row['id']}")
        for bid in excl["book_ids"]:
            if bid and bid in prov:
                leaks.append(f"book-id-leak: {row['id']}")
    if leaks:
        raise SystemExit("[error] leakage detected:\n  " + "\n  ".join(leaks[:10]))

    # Deterministic stratified split: shuffle per family, ~10% (min 2) to val.
    by_family: dict[str, list[dict]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    train: list[dict] = []
    val: list[dict] = []
    for fam in sorted(by_family):
        pool = sorted(by_family[fam], key=lambda x: x["id"])
        rng.shuffle(pool)
        n_val = max(2, round(len(pool) * 0.10))
        val.extend(pool[:n_val])
        train.extend(pool[n_val:])

    val_ids = {r["id"] for r in val}
    for row in rows:
        row["split"] = "val" if row["id"] in val_ids else "train"

    return {"train": train, "val": val, "rows": rows,
            "excluded_books": excluded_books,
            "excluded_claims": excluded_claims,
            "excluded_title_collisions": excluded_title_collisions,
            "books_total": len(books), "claims_total": len(claims)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path,
                    default=Path("data/knowledge/ryko.sqlite"))
    ap.add_argument("--probes", type=Path,
                    default=Path("docs/benchmarks/ling_probes.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/training"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    store = KnowledgeStore(str(args.store))
    built = build(store, args.probes, args.seed)
    store.close()

    train_path = args.out_dir / "ling_sft_train.jsonl"
    val_path = args.out_dir / "ling_sft_val.jsonl"
    _write_jsonl(train_path, built["train"])
    _write_jsonl(val_path, built["val"])

    by_family = Counter(r["family"] for r in built["rows"])
    by_split = Counter(r["split"] for r in built["rows"])
    manifest = {
        "schema": "ryko-ling-sft-manifest/1",
        "seed": args.seed,
        "sources": {
            "store": str(args.store), "store_sha256": _sha256(args.store),
            "probes": str(args.probes),
            "probes_sha256": _sha256(args.probes),
        },
        "rows": {"total": len(built["rows"]),
                 "train": len(built["train"]), "val": len(built["val"])},
        "by_family": dict(sorted(by_family.items())),
        "by_split": dict(sorted(by_split.items())),
        "probe_exclusions": {
            "books_total": built["books_total"],
            "books_excluded": built["excluded_books"],
            "title_collisions_excluded": built["excluded_title_collisions"],
            "claims_total": built["claims_total"],
            "claims_excluded": built["excluded_claims"],
            "leakage_check": "fatal; 0 leaks (probe quotes + ids)",
        },
        "held_out_honest_note": (
            "نسبة-الكتاب-اللغوي (rb-ling-book) تحتاج مقاطع لغوية غير ملتزمة "
            "تحت سقف 100MiB — عائلة محتجزة بالكامل، تُدرَّب وسيطًا فقط عبر "
            "المؤلف/الباب/إعادة التشغيل."),
        "outputs": {"train": str(train_path), "train_sha256": _sha256(train_path),
                    "val": str(val_path), "val_sha256": _sha256(val_path)},
    }
    (args.out_dir / "ling_sft_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    print(f"[ling-sft] train={len(built['train'])} val={len(built['val'])} "
          f"-> {train_path}")
    for fam, n in sorted(by_family.items()):
        print(f"[ling-sft]   {fam}: {n}")
    print(f"[ling-sft] probe exclusions: books {built['excluded_books']}"
          f"/{built['books_total']}, title-collisions "
          f"{built['excluded_title_collisions']}, claims "
          f"{built['excluded_claims']}/{built['claims_total']} — leakage 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
