"""Collect an Arabic text corpus from free, public sources.

This module downloads raw Arabic text without any authentication or heavy
dependencies (stdlib only).  It is the first stage of the Ryko pipeline:

    collect (this module) -> clean -> binarize -> train tokenizer -> train model

Sources
-------
wikipedia
    ``wikimedia/wikipedia`` (config ``20231101.ar``) served through the public
    HuggingFace ``datasets-server`` ``/rows`` API.  Contemporary Modern
    Standard Arabic across science, culture, biography, geography, ...
tashkeela
    ``arbml/tashkeela`` served the same way.  Classical / heritage Arabic
    texts (with a diacritized variant available), which diversifies the corpus
    with literary and religious register.

Every output record is a JSON line:  ``{"id", "title", "text", "source"}``.

Usage
-----
    python -m ryko_lm.data.collect_corpus \
        --source wikipedia --target-mb 18 --out data/raw/wiki.jsonl
    python -m ryko_lm.data.collect_corpus \
        --source tashkeela --target-mb 10 --out data/raw/tashkeela.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100  # max rows per /rows request

SOURCES: dict[str, dict] = {
    "wikipedia": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.ar",
        "split": "train",
        "title_field": "title",
        "text_field": "text",
    },
    "tashkeela": {
        "dataset": "arbml/tashkeela",
        "config": "default",
        "split": "train",
        "title_field": None,  # heritage snippets have no title
        "text_field": "text",
    },
}

USER_AGENT = "ryko-corpus-collector/1.0 (language-model research)"


def fetch_page(
    source_cfg: dict, offset: int, limit: int, retries: int = 5
) -> list[dict]:
    """Fetch one page of rows, retrying on transient errors (429/5xx)."""
    params = {
        "dataset": source_cfg["dataset"],
        "config": source_cfg["config"],
        "split": source_cfg["split"],
        "offset": str(offset),
        "length": str(limit),
    }
    url = f"{DATASETS_SERVER}?{urllib.parse.urlencode(params)}"
    backoff = 5.0
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            return [item["row"] for item in payload.get("rows", [])]
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt < retries:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            raise RuntimeError(f"network failure after {attempt} attempts: {exc}") from exc
    return []


def collect(
    source_name: str,
    out_path: Path,
    target_bytes: int,
    start_offset: int = 0,
    page_delay: float = 0.0,
) -> dict:
    """Stream pages of ``source_name`` into ``out_path`` until ``target_bytes`` of text."""
    source_cfg = SOURCES[source_name]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    collected = 0
    text_bytes = 0
    offset = start_offset
    mode = "a" if start_offset > 0 and out_path.exists() else "w"

    t0 = time.time()
    with out_path.open(mode, encoding="utf-8") as out:
        while text_bytes < target_bytes:
            try:
                rows = fetch_page(source_cfg, offset, PAGE_SIZE)
            except RuntimeError as exc:  # give up: keep what we have
                print(f"[warn] stopping '{source_name}' at offset={offset}: {exc}", flush=True)
                break
            if not rows:
                break  # exhausted the split

            for row in rows:
                text = (row.get(source_cfg["text_field"]) or "").strip()
                if len(text) < 200:  # skip stubs / empty snippets
                    continue
                title = ""
                if source_cfg["title_field"]:
                    title = (row.get(source_cfg["title_field"]) or "").strip()
                record = {
                    "id": f"{source_name}-{offset}",
                    "title": title,
                    "text": text,
                    "source": source_name,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                collected += 1
                text_bytes += len(text.encode("utf-8"))
            offset += len(rows)
            if page_delay > 0:
                time.sleep(page_delay)

            if collected % 1000 == 0:
                rate = collected / max(time.time() - t0, 1e-9)
                print(
                    f"[{source_name}] rows={collected} "
                    f"mb={text_bytes / 1e6:.1f} rate={rate:.0f} rec/s",
                    flush=True,
                )

    stats = {
        "source": source_name,
        "dataset": source_cfg["dataset"],
        "config": source_cfg["config"],
        "records": collected,
        "text_mb": round(text_bytes / 1e6, 2),
        "last_offset": offset,
        "seconds": round(time.time() - t0, 1),
    }
    print(f"[done] {json.dumps(stats, ensure_ascii=False)}", flush=True)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect Arabic corpus from free sources.")
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--out", type=Path, required=True, help="output JSONL path")
    parser.add_argument("--target-mb", type=float, default=18.0, help="target text volume in MB")
    parser.add_argument("--start-offset", type=int, default=0, help="resume offset (rows)")
    parser.add_argument(
        "--page-delay", type=float, default=0.0, help="sleep between pages (seconds), be polite"
    )
    args = parser.parse_args(argv)

    stats = collect(
        source_name=args.source,
        out_path=args.out,
        target_bytes=int(args.target_mb * 1e6),
        start_offset=args.start_offset,
        page_delay=args.page_delay,
    )
    sidecar = args.out.with_suffix(".stats.json")
    sidecar.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
