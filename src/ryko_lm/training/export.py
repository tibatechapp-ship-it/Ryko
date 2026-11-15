"""Export a training checkpoint (``last.pt``) into the serving format.

Serving loads ``final.pt`` files written by :meth:`RykoLM.save`
(``{"config": ..., "state_dict": ...}``, fp16, tied weights de-duplicated).
This small module converts a resumable training checkpoint into exactly that
format, so the API can serve a mid-schedule checkpoint without waiting for
the cosine schedule to finish.

Usage:
    python -m ryko_lm.training.export \
        --checkpoint artifacts/training_v3/last.pt \
        --config configs/models/ryko-tiny.yaml \
        --out artifacts/training_v3/final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from ..model.ryko_model import RykoConfig, RykoLM


def export(checkpoint: Path, config_path: Path, out: Path) -> dict:
    cfg = RykoConfig.from_dict(yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"])
    model = RykoLM.load(checkpoint, device="cpu", config=cfg)
    model.save(out, half=True)
    info = {"checkpoint": str(checkpoint), "out": str(out), "params": model.num_params()}
    sidecar = out.with_name(out.stem + "_export.json")
    sidecar.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export last.pt -> serving final.pt")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    info = export(args.checkpoint, args.config, args.out)
    print(f"[export] {json.dumps(info)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
