"""Bundle pack builder (الحزمة المجمّعة v1.26.0) — حجاج + لغة + أضعف العائلات.

The v1.26.0 combined treatment pack documented in ``docs/EVALUATION.md`` §5:
a deterministic, leakage-gated mix of four committed/deterministic components,
trained as one SFT pass to break the narrow-pack interference documented in
v1.25.0 (rb-ling −13.46 on the argrel-only shard):

* **argrel** — the committed v1.25.0 argument-relations package
  (``argrel_sft_{train,val}.jsonl``, 90/15 pairs), byte-identical reuse;
* **ling-sft** — the new attribution pairs built from the Knowledge Core by
  ``scripts/build_ling_sft.py`` (``ling_sft_{train,val}.jsonl``), probe-excluded;
* **ling-replay** — the committed v1.23.0 DPO round-2 linguistics preferences
  (families irab/sarf/balagha) replayed as SFT pairs (prompt → chosen), minus
  any row quoting a probe case text (measured and recorded, not hidden);
* **weakest-families** — the committed phase-5 SFT renderings of the three
  weakest measured RYKO-1000 families (istidlal / mustalahat / tarjih).

Determinism contract (same discipline as ``pack_argrel`` / ``pack_sft``):
fixed shuffle seed, stable id order, same inputs ⇒ byte-identical outputs and
byte-identical pack receipt.  The probe-leakage gate is fatal: no rb-ling
probe quote, title, claim/book/passage id may appear in any training row.

Usage::

    PYTHONPATH=src python -m ryko_lm.training.pack_bundle \\
        --tokenizer artifacts/tokenizer/ryko-tokenizer.json \\
        --out data/training \\
        --pack-out data/training/bundle_pack
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

SEED = 1337
SCHEMA = "ryko-bundle-pack/1"
DEFAULT_FAMILIES = "istidlal,mustalahat,tarjih"
DEFAULT_REPLAY_FAMILIES = "irab,sarf,balagha"

# Long probe quotes (claim texts, passage excerpts) must not appear as
# substrings; short quotes are book titles — exact quoted-title match only.
_SHORT_TITLE = 25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm(text: str) -> str:
    return re.sub(r"[\s\u064B-\u0652\u0670«»(){}.,،؛:!?\-—]+", "", text)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _probe_exclusions(probes_path: Path) -> dict:
    claim_ids, book_ids, passage_ids, quotes = set(), set(), set(), set()
    for case in _load_jsonl(probes_path):
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
            "title_norms": {q for q in quotes if len(q) < _SHORT_TITLE}}


def _quote_hits(prompt: str, excl: dict) -> str | None:
    """Return a leak tag if the prompt reproduces any probe case, else None."""
    np_ = _norm(prompt)
    quoted = [_norm(q) for q in re.findall(r"«(.+?)»", prompt, re.S)]
    for q in excl["quotes"]:
        if not q:
            continue
        if len(q) >= _SHORT_TITLE:
            if q in np_:
                return "long-quote"
        elif any(q == nq for nq in quoted):
            return "title"
    return None


def _prov_hits(row: dict, excl: dict) -> str | None:
    prov = json.dumps(row.get("provenance", {}), ensure_ascii=False)
    for cid in excl["claim_ids"]:
        if cid and cid in prov:
            return "claim-id"
    for bid in excl["book_ids"]:
        if bid and bid in prov:
            return "book-id"
    for pid in excl["passage_ids"]:
        if pid and pid in prov:
            return "passage-id"
    return None


def build(out: Path, argrel_train_path: Path, argrel_val_path: Path,
          ling_train_path: Path, ling_val_path: Path, sft_train_path: Path,
          dpo_r2_path: Path, probes_path: Path, seed: int = SEED,
          families: tuple[str, ...] = DEFAULT_FAMILIES.split(","),
          replay_families: tuple[str, ...] = DEFAULT_REPLAY_FAMILIES.split(",")) -> dict:
    excl = _probe_exclusions(probes_path)

    argrel_train = _load_jsonl(argrel_train_path)
    argrel_val = _load_jsonl(argrel_val_path)
    ling_train = _load_jsonl(ling_train_path)
    ling_val = _load_jsonl(ling_val_path)

    sft_rows = [r for r in _load_jsonl(sft_train_path)
                if r.get("family") in families]

    replay_rows: list[dict] = []
    replay_excluded: list[str] = []
    for row in _load_jsonl(dpo_r2_path):
        if row.get("family") not in replay_families:
            continue
        if _quote_hits(row["prompt"], excl):
            replay_excluded.append(row["id"])
            continue
        replay_rows.append({
            "schema": "ryko1000-dpo-replay/1",
            "id": row["id"],
            "family": row["family"],
            "probe_kind": row.get("probe_kind"),
            "provenance": row.get("provenance", {}),
            "prompt": row["prompt"],
            "completion": row["chosen"],
            "split": "train",
        })

    train_rows = argrel_train + ling_train + sft_rows + replay_rows
    val_rows = argrel_val + ling_val

    # Fatal leakage gate over every row of the bundle.
    leaks: list[str] = []
    for row in train_rows + val_rows:
        tag = _quote_hits(row.get("prompt", ""), excl) or _prov_hits(row, excl)
        if tag:
            leaks.append(f"{tag}: {row.get('id')}")
    if leaks:
        raise SystemExit("[error] probe leakage in bundle:\n  "
                         + "\n  ".join(leaks[:10]))

    # Deterministic mix: stable id order, then seeded shuffle.
    train_rows = sorted(train_rows, key=lambda r: r["id"])
    val_rows = sorted(val_rows, key=lambda r: r["id"])
    random.Random(seed).shuffle(train_rows)
    random.Random(seed).shuffle(val_rows)

    train_path = out / "bundle_sft_train.jsonl"
    val_path = out / "bundle_sft_val.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)

    def component(row: dict) -> str:
        schema = row.get("schema", "")
        if schema == "ryko-argrel-pair/1":
            return "argrel"
        if schema == "ryko-ling-sft/1":
            return "ling-sft"
        if schema == "ryko1000-sft/1":
            return "weakest-families"
        if schema == "ryko1000-dpo-replay/1":
            return "ling-replay"
        return "other"

    manifest = {
        "schema": SCHEMA,
        "seed": seed,
        "sources": {
            "argrel_train": {"path": str(argrel_train_path),
                             "sha256": _sha256(argrel_train_path)},
            "argrel_val": {"path": str(argrel_val_path),
                           "sha256": _sha256(argrel_val_path)},
            "ling_train": {"path": str(ling_train_path),
                           "sha256": _sha256(ling_train_path)},
            "ling_val": {"path": str(ling_val_path),
                         "sha256": _sha256(ling_val_path)},
            "sft_train": {"path": str(sft_train_path),
                          "sha256": _sha256(sft_train_path)},
            "dpo_r2": {"path": str(dpo_r2_path),
                       "sha256": _sha256(dpo_r2_path)},
            "probes": {"path": str(probes_path),
                       "sha256": _sha256(probes_path)},
        },
        "components": {
            "argrel": {"train": len(argrel_train), "val": len(argrel_val)},
            "ling-sft": {"train": len(ling_train), "val": len(ling_val)},
            "weakest-families": {"train": len(sft_rows), "val": 0,
                                 "families": sorted(families)},
            "ling-replay": {"train": len(replay_rows), "val": 0,
                            "families": sorted(replay_families),
                            "excluded_probe_overlap": len(replay_excluded),
                            "excluded_ids": replay_excluded},
        },
        "rows_train": len(train_rows),
        "rows_val": len(val_rows),
        "by_family_train": dict(sorted(
            Counter(r.get("family", "?") for r in train_rows).items())),
        "files": {
            "train": {"path": str(train_path), "n": len(train_rows),
                      "sha256": _sha256(train_path)},
            "val": {"path": str(val_path), "n": len(val_rows),
                    "sha256": _sha256(val_path)},
        },
        "leakage_check": {"probe_leaks": 0, "policy": "fatal gate; 0 tolerated"},
        "unit_of_mix": ("deterministic component union, id-sorted then "
                        "seed-1337 shuffled — no family straddles a leak"),
        "provenance_policy": (
            "argrel = committed v1.25.0 package; ling-sft = new store-derived "
            "attribution pairs, probe-excluded; ling-replay = committed v1.23.0 "
            "DPO-r2 linguistics preferences replayed as SFT (prompt→chosen); "
            "weakest-families = committed phase-5 SFT renderings of the three "
            "weakest measured RYKO-1000 families (documented phase-5 design)"),
    }
    (out / "bundle_pack_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


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
    receipt = make_receipt(pack_out, kind="bundle-pack", meta=meta)
    (pack_out / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"pack_manifest": manifest, "receipt": receipt}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ryko_pack_bundle",
        description="Combined-treatment bundle pack builder (v1.26.0).")
    p.add_argument("--tokenizer", type=Path,
                   default=Path("artifacts/tokenizer/ryko-tokenizer.json"))
    p.add_argument("--out", type=Path, default=Path("data/training"))
    p.add_argument("--pack-out", type=Path,
                   default=Path("data/training/bundle_pack"))
    p.add_argument("--argrel-train", type=Path,
                   default=Path("data/training/argrel_sft_train.jsonl"))
    p.add_argument("--argrel-val", type=Path,
                   default=Path("data/training/argrel_sft_val.jsonl"))
    p.add_argument("--ling-train", type=Path,
                   default=Path("data/training/ling_sft_train.jsonl"))
    p.add_argument("--ling-val", type=Path,
                   default=Path("data/training/ling_sft_val.jsonl"))
    p.add_argument("--sft-train", type=Path,
                   default=Path("data/training/ryko1000_sft_train.jsonl"))
    p.add_argument("--dpo-r2", type=Path,
                   default=Path("data/training/ryko1000_dpo_train_r2.jsonl"))
    p.add_argument("--probes", type=Path,
                   default=Path("docs/benchmarks/ling_probes.jsonl"))
    p.add_argument("--families", type=str, default=DEFAULT_FAMILIES)
    p.add_argument("--replay-families", type=str, default=DEFAULT_REPLAY_FAMILIES)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-pack", action="store_true")
    args = p.parse_args(argv)

    manifest = build(args.out, args.argrel_train, args.argrel_val,
                     args.ling_train, args.ling_val, args.sft_train,
                     args.dpo_r2, args.probes, args.seed,
                     tuple(args.families.split(",")),
                     tuple(args.replay_families.split(",")))
    comps = manifest["components"]
    print(f"[bundle] rows train={manifest['rows_train']} "
          f"val={manifest['rows_val']} | "
          f"argrel {comps['argrel']['train']} | ling-sft "
          f"{comps['ling-sft']['train']} | weakest "
          f"{comps['weakest-families']['train']} | replay "
          f"{comps['ling-replay']['train']} "
          f"(probe-overlap excluded {comps['ling-replay']['excluded_probe_overlap']}) "
          f"| leakage 0")

    if not args.no_pack:
        packed = pack(Path(manifest["files"]["train"]["path"]),
                      Path(manifest["files"]["val"]["path"]),
                      args.tokenizer, args.pack_out,
                      meta_extra={"seed": args.seed,
                                  "components": {k: v.get("train", 0)
                                                 for k, v in comps.items()}})
        pm = packed["pack_manifest"]["files"]
        print(f"[pack] train {pm['train']['pairs']} pairs "
              f"{pm['train']['tokens']:,} tokens "
              f"sha256={pm['train']['sha256'][:16]}… | "
              f"val {pm['val']['pairs']} pairs "
              f"{pm['val']['tokens']:,} tokens "
              f"sha256={pm['val']['sha256'][:16]}… -> {args.pack_out}")
        steps = -(-pm["train"]["tokens"] * 3 // 2048)  # ~3 epochs, ceil
        print(f"[budget] tokens/step=2048 → ~3 epochs ≈ {min(steps, 120)} steps "
              f"(raw {steps}, capped 120)")
        print(f"[receipt] sha256={packed['receipt']['receipt_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
