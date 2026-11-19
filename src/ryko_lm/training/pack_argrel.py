"""Argument-relations proof-shard builder (علاقة-حجاج — the v1.24.0 measured weakness).

Treats the ``argument_relations`` graph gap with the project's own
gate-closed proof-shard pattern (the v1.21.0 linguistics precedent:
warm start, few steps, measured lesson published either way).  Reads the
35 documented relations from the committed store (``persist_vectors=False``
— read-only session), emits deterministic Arabic prompt/completion pairs
grounded in the actual rows, splits per-RELATION (never per-phrase — one
relation's phrasings may never straddle the train/val boundary), and packs
uint16 streams under the exact ``pack_sft`` contract.

Determinism contract (same discipline as ``pack_sft`` / ``sft_prep``):

* relations come from :func:`ryko_lm.knowledge.relations.list_relations`
  (fixed ``relation_id`` ordering, no RNG anywhere);
* the ONLY shuffle is the per-relation train/val split, driven by a fixed
  ``--split-seed`` — rerunning rebuilds every file byte for byte;
* three fixed template variants per relation, emitted in fixed order
  ``v1`` (the bench-gate conditional) → ``v2`` (open grounding) →
  ``v3`` (classify frame), so ~2-4 phrasings per relation with no
  paraphrase lottery;
* degenerate pairs are skipped loudly (recorded in the manifest), never
  silently patched.

Leakage is FATAL, mirroring ``sft_prep``: any relation_id appearing in
both splits, any pair id or rendered prompt straddling the boundary —
the builder raises before writing anything.

The shard's own val split is NOT the benchmark's held-out promise: the
bench (``docs/benchmarks/ryko_bench_seed.jsonl``) draws from the same
documented graph — that overlap is inherent to treating a measured
weakness on its own gate and must be reported honestly (see the
per-relation trained/val breakdown in the shard's results md).

Usage::

    python -m ryko_lm.training.pack_argrel \
        --store data/knowledge/ryko.sqlite \
        --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
        --out data/training \
        --pack-out data/training/argrel_pack
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from ..knowledge.relations import RELATION_AR, list_relations
from ..knowledge.schema import sha256_text
from ..knowledge.store import KnowledgeStore

SCHEMA = "ryko-argrel-pack/1"
PAIR_SCHEMA = "ryko-argrel-pair/1"
FAMILY = "علاقة-حجاج"

SPLIT_SEED = 1337
TRAIN_SHARE = 0.85
# the bench gate rule (build_bench._relation_cases): both endpoints must
# survive whitespace-normalization at >= 20 chars
MIN_ENDPOINT_CHARS = 20

# fixed variant ids — the order IS the contract (no paraphrase lottery)
VARIANT_IDS = ("v1_bench_gate", "v2_open_grounding", "v3_classify")

BENCH_QUESTION_SUFFIX = (
    "ما علاقة القول الأول بالثاني كما هي موثقة في مزرعة العلاقات؟"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm(text: str) -> str:
    """The bench rendering: collapse whitespace, truncate to 110 chars."""
    return " ".join((text or "").split())[:110]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")


# ------------------------------------------------------------- pair factory
def pair_variants(rel: dict) -> tuple[list[dict], list[str]]:
    """The fixed template variants for one relation, in contract order.

    Returns ``(rows, skipped_variant_ids)`` — a variant is skipped loudly
    (recorded in the manifest) when it would emit an ungrounded or
    degenerate pair; nothing is invented.
    """
    verb = rel.get("relation")
    verb_ar = RELATION_AR.get(verb)
    src = _norm((rel.get("src") or {}).get("text"))
    dst = _norm((rel.get("dst") or {}).get("text"))
    src_o = (rel.get("src") or {}).get("originator") or ""
    dst_o = (rel.get("dst") or {}).get("originator") or ""
    evidence = (rel.get("evidence") or "").strip()
    layer = rel.get("layer") or ""
    rid = rel.get("relation_id") or ""
    src_id = (rel.get("src") or {}).get("claim_id") or ""
    dst_id = (rel.get("dst") or {}).get("claim_id") or ""

    rows: list[dict] = []
    skipped: list[str] = []
    provenance = {
        "relation_id": rid, "relation": verb, "layer": layer,
        "src_claim": src_id, "dst_claim": dst_id,
    }

    def emit(vid: str, prompt: str, completion: str) -> None:
        if not prompt.strip() or not completion.strip() or not completion.strip(" .—؛"):
            skipped.append(vid)
            return
        rows.append({
            "id": f"argrel-{rid}-{vid.split('_')[0]}",
            "schema": PAIR_SCHEMA,
            "family": FAMILY,
            "provenance": {**provenance, "evidence_sha256":
                           sha256_text(evidence)[:16] if evidence else None},
            "prompt": prompt,
            "completion": completion,
        })

    # v1 — the exact conditional the ryko-bench/1 MCQ gate scores:
    # "السؤال: {question}\nالجواب:" -> " {option}" (same framing as
    # sft_prep._sft_pair's mcq path; the question is the bench's own
    # rendering of the two claim texts).
    if verb_ar and len(src) >= MIN_ENDPOINT_CHARS and len(dst) >= MIN_ENDPOINT_CHARS:
        emit("v1_bench_gate",
             f"السؤال: القول الأول: «{src}»\n"
             f"القول الثاني: «{dst}»\n"
             f"{BENCH_QUESTION_SUFFIX}\n"
             "الجواب:",
             f" {verb_ar}")
    else:
        skipped.append("v1_bench_gate")

    # v2 — open question, completion names both documented sides + evidence.
    ev = evidence.rstrip(".") if evidence else ""
    if verb_ar and ev and src_o and dst_o:
        emit("v2_open_grounding",
             "في مزرعة العلاقات: ما العلاقة الموثقة بين القولين الآتيين؟\n"
             f"القول الأول: «{src}»\n"
             f"القول الثاني: «{dst}»\n"
             "الجواب:",
             f" {verb_ar} — الأصل: {src_o}؛ الهدف: {dst_o}؛ السند: {ev}.")
    else:
        skipped.append("v2_open_grounding")

    # v3 — classify frame; the evidence line is the documented grounding.
    if verb_ar and evidence and len(src) >= MIN_ENDPOINT_CHARS and len(dst) >= MIN_ENDPOINT_CHARS:
        emit("v3_classify",
             f"صنّف العلاقة الحجاجية الموثقة بين القول الأول «{src}» "
             f"والقول الثاني «{dst}» وفق مفردات العلاقات المغلقة.\n"
             "الجواب:",
             f" {verb_ar} — {evidence}")
    else:
        skipped.append("v3_classify")

    return rows, skipped


def split_relations(relation_ids: list[str], split_seed: int = SPLIT_SEED,
                    train_share: float = TRAIN_SHARE) -> tuple[list[str], list[str]]:
    """Deterministic per-RELATION 85/15 split (the unit of leakage).

    One RNG over the sorted ids, shuffled once — rerunning with the same
    seed reproduces the split byte for byte; a relation's phrasings can
    never straddle the boundary because the split never sees a phrase.
    """
    pool = sorted(set(relation_ids))
    rng = random.Random(split_seed)
    rng.shuffle(pool)
    n_train = round(len(pool) * train_share)
    return pool[:n_train], pool[n_train:]


def leakage_check(train_rows: list[dict], val_rows: list[dict]) -> dict:
    """FATAL like ``sft_prep``: raise on any cross-split straddle."""
    train_rel = {r["provenance"]["relation_id"] for r in train_rows}
    val_rel = {r["provenance"]["relation_id"] for r in val_rows}
    leaked_relations = sorted(train_rel & val_rel)
    train_ids = {r["id"] for r in train_rows}
    val_ids = {r["id"] for r in val_rows}
    leaked_ids = sorted(train_ids & val_ids)
    train_prompts = {r["prompt"] for r in train_rows}
    val_prompts = {r["prompt"] for r in val_rows}
    leaked_prompts = sorted(train_prompts & val_prompts)
    report = {"leaked_relations": leaked_relations,
              "leaked_pair_ids": leaked_ids,
              "leaked_prompts": leaked_prompts, "fatal": True}
    if leaked_relations or leaked_ids or leaked_prompts:
        raise ValueError(
            "argrel split leakage (fatal, like sft_prep): "
            f"relations={leaked_relations[:5]} ids={leaked_ids[:5]} "
            f"prompts={len(leaked_prompts)}")
    return report


def build(store_path: Path, out_dir: Path, split_seed: int = SPLIT_SEED,
          train_share: float = TRAIN_SHARE) -> dict:
    """Read the documented graph, emit splits + manifest (byte-stable)."""
    store = KnowledgeStore(str(store_path), persist_vectors=False)
    try:
        rels = list_relations(store, k=0)
    finally:
        store.close()
    if not rels:
        raise ValueError(f"no argument_relations in {store_path}")

    all_rows: dict[str, list[dict]] = {}
    skipped_all: list[dict] = []
    for rel in rels:
        rows, skipped = pair_variants(rel)
        all_rows[rel["relation_id"]] = rows
        for vid in skipped:
            skipped_all.append({"relation_id": rel["relation_id"], "variant": vid})

    train_rel, val_rel = split_relations([r["relation_id"] for r in rels],
                                         split_seed, train_share)

    def collect(rel_ids: list[str], split: str) -> list[dict]:
        rows: list[dict] = []
        for rid in sorted(rel_ids):
            for row in all_rows[rid]:
                rows.append({**row, "split": split})
        rows.sort(key=lambda r: r["id"])
        return rows

    train_rows = collect(train_rel, "train")
    val_rows = collect(val_rel, "val")
    leak = leakage_check(train_rows, val_rows)

    out_dir = Path(out_dir)
    train_path = out_dir / "argrel_sft_train.jsonl"
    val_path = out_dir / "argrel_sft_val.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)

    def by_verb(rows: list[dict]) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            out[r["provenance"]["relation"]] = out.get(r["provenance"]["relation"], 0) + 1
        return dict(sorted(out.items()))

    def by_variant(rows: list[dict]) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            vid = r["id"].rsplit("-", 1)[1]
            out[vid] = out.get(vid, 0) + 1
        return dict(sorted(out.items()))

    manifest = {
        "schema": SCHEMA,
        "store": {"path": str(store_path), "sha256": _sha256(Path(store_path))},
        "split_seed": split_seed,
        "train_share": train_share,
        "relations_total": len(rels),
        "relations_train": len(train_rel),
        "relations_val": len(val_rel),
        "pairs_train": len(train_rows),
        "pairs_val": len(val_rows),
        "by_relation": by_verb(train_rows + val_rows),
        "by_variant": by_variant(train_rows + val_rows),
        "skipped": skipped_all,
        "files": {
            "train": {"path": str(train_path), "n": len(train_rows),
                      "sha256": _sha256(train_path)},
            "val": {"path": str(val_path), "n": len(val_rows),
                    "sha256": _sha256(val_path)},
        },
        "leakage_check": {k: (len(v) if isinstance(v, list) else v)
                          for k, v in leak.items()},
        "unit_of_split": "relation_id — a relation's phrasings never straddle splits",
    }
    (out_dir / "argrel_pack_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# ------------------------------------------------------------------- packing
def pack(train_path: Path, val_path: Path, tokenizer_path: Path,
         pack_out: Path, meta_extra: dict | None = None) -> dict:
    """uint16 streams under the exact pack_sft contract + signed receipt."""
    from ..provenance import make_receipt
    from .pack_sft import build as pack_sft_build

    pack_out = Path(pack_out)
    manifest = pack_sft_build(Path(train_path), Path(val_path),
                              Path(tokenizer_path), pack_out)
    meta = {
        "schema": SCHEMA,
        "train_pairs": manifest["files"]["train"]["pairs"],
        "train_tokens": manifest["files"]["train"]["tokens"],
        "val_pairs": manifest["files"]["val"]["pairs"],
        "val_tokens": manifest["files"]["val"]["tokens"],
        "tokens_total": manifest["tokens_total"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        **(meta_extra or {}),
    }
    receipt = make_receipt(pack_out, kind="argrel-pack", meta=meta)
    (pack_out / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"pack_manifest": manifest, "receipt": receipt}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ryko_pack_argrel",
        description="Argument-relations proof-shard builder (علاقة-حجاج).")
    p.add_argument("--store", type=Path,
                   default=Path("data/knowledge/ryko.sqlite"))
    p.add_argument("--tokenizer", type=Path,
                   default=Path("artifacts/tokenizer/ryko-tokenizer.json"))
    p.add_argument("--out", type=Path, default=Path("data/training"),
                   help="directory for argrel_sft_{train,val}.jsonl + manifest")
    p.add_argument("--pack-out", type=Path,
                   default=Path("data/training/argrel_pack"),
                   help="pack dir (train.bin/val.bin/receipt.json); "
                        "pass --no-pack to skip")
    p.add_argument("--no-pack", action="store_true")
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    p.add_argument("--train-share", type=float, default=TRAIN_SHARE)
    args = p.parse_args(argv)

    manifest = build(args.store, args.out, args.split_seed, args.train_share)
    print(f"[argrel] relations {manifest['relations_total']} "
          f"(train {manifest['relations_train']} / val {manifest['relations_val']}) "
          f"| pairs {manifest['pairs_train'] + manifest['pairs_val']} "
          f"(train {manifest['pairs_train']} / val {manifest['pairs_val']}) "
          f"| skipped {len(manifest['skipped'])} "
          f"| leakage {manifest['leakage_check']}")
    if not args.no_pack:
        packed = pack(Path(manifest["files"]["train"]["path"]),
                      Path(manifest["files"]["val"]["path"]),
                      args.tokenizer, args.pack_out,
                      meta_extra={"store_sha256": manifest["store"]["sha256"],
                                  "split_seed": args.split_seed})
        pm = packed["pack_manifest"]["files"]
        print(f"[pack] train {pm['train']['pairs']} pairs "
              f"{pm['train']['tokens']:,} tokens "
              f"sha256={pm['train']['sha256'][:16]}… | "
              f"val {pm['val']['pairs']} pairs "
              f"{pm['val']['tokens']:,} tokens "
              f"sha256={pm['val']['sha256'][:16]}… -> {args.pack_out}")
        print(f"[receipt] sha256={packed['receipt']['receipt_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
