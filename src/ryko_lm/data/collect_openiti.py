"""Collect the OpenITI classical-Arabic corpus (release 2023.1.8) from HuggingFace.

OpenITI (Open Arabic / Islamic Texts Initiative) distributes heritage texts in
mARkdown.  The ``mittagessen/openiti-2023.1.8`` mirror exposes the release as
auto-converted parquet shards, which lets us stream whole books without cloning
the ~25 period repositories.

This collector:

1.  lists the available shards through the datasets-server parquet endpoint,
2.  downloads one shard at a time into a scratch directory,
3.  performs a *light* mARkdown-aware pre-clean (metadata headers, page-distance
    markers, margin tags, latin lines) — full Arabic normalisation stays in
    :mod:`ryko_lm.data.clean_corpus`,
4.  appends ``{"id", "text"}`` records to ``data/raw/openiti.jsonl`` and removes
    the scratch parquet file (disk-friendly).

The run is resumable: completed shard indices are recorded in
``data/raw/openiti.progress.json`` and skipped on re-run.

Usage:
    python -m ryko_lm.data.collect_openiti --shards 0-2 \
        --out data/raw/openiti.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

PARQUET_ENDPOINT = (
    "https://datasets-server.huggingface.co/parquet"
    "?dataset=mittagessen%2Fopeniti-2023.1.8"
)

# ---------------------------------------------------------------------------
# mARkdown pre-clean
# ---------------------------------------------------------------------------
META_RE = re.compile(r"^# ?META#")
EDITION_BLOCK_RE = re.compile(r"^<[^<>]*>$")          # "<edition>" lines
DISTANCE_RE = re.compile(r"^%~\d*#?$")                # "%~12#" page distances
MARGIN_TAG_RE = re.compile(r"\$(?:right|left)(?:_m\d*)?\$")
HASH_PREFIX_RE = re.compile(r"^#{1,6}\s*")
STAR_RE = re.compile(r"\*+")
PIPE_RE = re.compile(r"\|")
MULTISPACE_RE = re.compile(r"[ \t\u00a0]+")
LATIN_HEAVY_RE = re.compile(r"[A-Za-z]")


def preclean_markdown(text: str) -> str:
    """Strip mARkdown scaffolding while keeping the Arabic body text."""
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if not line.strip():
            lines.append("")
            continue
        if META_RE.match(line) or EDITION_BLOCK_RE.match(line):
            continue
        if line.startswith("...") or line.startswith("---"):
            continue
        if DISTANCE_RE.match(line.strip()):
            continue
        line = MARGIN_TAG_RE.sub(" ", line)
        line = HASH_PREFIX_RE.sub("", line)
        line = STAR_RE.sub(" ", line)
        line = PIPE_RE.sub(" ", line)
        latin = len(LATIN_HEAVY_RE.findall(line))
        if latin > len(line) * 0.4:  # mostly latin: OCR noise / footnotes
            continue
        line = MULTISPACE_RE.sub(" ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fetch_json(url: str, retries: int = 3, timeout: int = 30) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - network backoff
            last = exc
            wait = 3.0 * (attempt + 1)
            print(f"[retry] {url[:90]}... ({exc}); sleeping {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def _download(url: str, dest: Path, timeout: int = 120) -> None:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, dest.open("wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            return
        except Exception as exc:  # noqa: BLE001
            wait = 5.0 * (attempt + 1)
            print(f"[retry] download ({exc}); sleeping {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url}")


def _parse_shard_spec(spec: str, total: int) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), min(int(hi), total - 1) + 1))
    return [int(i) for i in spec.split(",") if int(i) < total]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect OpenITI parquet shards.")
    parser.add_argument("--out", type=Path, default=Path("data/raw/openiti.jsonl"))
    parser.add_argument("--progress", type=Path, default=Path("data/raw/openiti.progress.json"))
    parser.add_argument("--shards", type=str, default="0-0",
                        help="shard index or inclusive range, e.g. '2' or '0-3'")
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/ryko_openiti"))
    parser.add_argument("--min-chars", type=int, default=2000)
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq  # local import: only needed here

    listing = _fetch_json(PARQUET_ENDPOINT)
    files = [f for f in listing.get("parquet_files", [])
             if f.get("split") == "train" and f["url"].endswith(".parquet")]
    files.sort(key=lambda f: f["url"])
    if not files:
        print("[error] no parquet shards found at the endpoint", flush=True)
        return 1
    print(f"[openiti] {len(files)} shards available", flush=True)

    wanted = _parse_shard_spec(args.shards, len(files))
    done: set[int] = set()
    if args.progress.exists():
        try:
            done = set(json.loads(args.progress.read_text())["shards_done"])
        except Exception:  # noqa: BLE001
            done = set()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    total_chars, total_docs = 0, 0
    t0 = time.time()

    for idx in wanted:
        if idx in done:
            print(f"[skip] shard {idx:04d} already collected", flush=True)
            continue
        url = files[idx]["url"]
        shard_path = args.scratch / f"openiti_{idx:04d}.parquet"
        print(f"[shard {idx:04d}] downloading {files[idx]['size'] / 1e6:.0f} MB ...", flush=True)
        _download(url, shard_path)

        shard_docs, shard_chars = 0, 0
        t_shard = time.time()
        with pq.ParquetFile(shard_path) as pf, args.out.open("a", encoding="utf-8") as sink:
            for batch in pf.iter_batches(batch_size=32, columns=["text"]):
                for value in batch.column("text"):
                    text = value.as_py() or ""
                    cleaned = preclean_markdown(text)
                    if len(cleaned) < args.min_chars:
                        continue
                    record = {"id": f"openiti-{idx:04d}-{shard_docs:06d}", "text": cleaned}
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    shard_docs += 1
                    shard_chars += len(cleaned)
        shard_path.unlink(missing_ok=True)
        done.add(idx)
        args.progress.write_text(json.dumps({"shards_done": sorted(done)}), encoding="utf-8")
        total_docs += shard_docs
        total_chars += shard_chars
        print(
            f"[shard {idx:04d}] {shard_docs} books, {shard_chars / 1e6:.1f}M chars "
            f"in {time.time() - t_shard:.0f}s",
            flush=True,
        )

    print(
        f"[done] +{total_docs} docs, +{total_chars / 1e6:.1f}M chars "
        f"in {time.time() - t0:.0f}s -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
