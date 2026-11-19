#!/usr/bin/env python3
"""WiSE-FT soup between two checkpoints (تجربة تقاطع الوزن — v1.26.0).

w_soup = alpha * w_A + (1 - alpha) * w_B, same config/layout (both points
share the 52,179,456-param contract).  Deterministic; writes a final.pt in
the same {config, state_dict} shape so the standard bench runner reads it.

Usage:
    python scripts/soup_checkpoints.py --a artifacts/dpo_round2_small/final.pt \
        --b artifacts/proof_argrel_small/final.pt --alpha 0.5 \
        --out artifacts/soup_small/final.pt
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--alpha", type=float, required=True,
                    help="weight of checkpoint A (0..1)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    da = torch.load(args.a, map_location="cpu", weights_only=False)
    db = torch.load(args.b, map_location="cpu", weights_only=False)
    sa, sb = da["state_dict"], db["state_dict"]
    assert set(sa.keys()) == set(sb.keys()), "state_dict layout mismatch"
    soup = copy.deepcopy(da)
    with torch.no_grad():
        for k in sa:
            fa, fb_ = sa[k].float(), sb[k].float()
            soup["state_dict"][k] = (args.alpha * fa
                                     + (1.0 - args.alpha) * fb_).to(sa[k].dtype)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup, args.out)
    print(f"[soup] alpha={args.alpha} a={args.a} b={args.b} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
