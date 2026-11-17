"""Phase-9 DPO training on the RYKO-1000 preference package.

Direct Preference Optimization over the structurally derived preference
pairs of ``ryko_lm.training.sft_prep`` (chosen = grounded/cited behaviour,
rejected = the exact embedded failure mode). The reference model is a
frozen copy of the policy's *initial* weights (the SFT checkpoint), so the
learned drift is measured against the phase-5 point — the same two-point
discipline the corpus training follows.

Loss (per pair)::

    loss = -log sigmoid(beta * ((s_chosen - s_chosen_ref)
                              - (s_rejected - s_rejected_ref)))

where ``s_*`` is the summed token log-probability of the response under
the policy / reference model given the shared prompt.  Responses are
encoded *separately* from the prompt (documented approximation: BPE
boundary effects at the prompt/response seam are identical for both
models, so the DPO margin is not biased by them).

The trainer follows ``train.py`` discipline exactly: seeded RNG, AdamW,
grad clipping, cosine LR, exact resume from ``last.pt`` (model + optimizer
+ step), graceful ``--time-budget`` stop with checkpoint, JSONL metrics
log, and a final summary with the policy's DPO accuracy (chosen preferred
over rejected) over the whole package.

Memory layout: policy (params + grads + AdamW) + frozen reference
(no grad) ≈ 1.1 GB for ryko-small in fp32 — sized for a 3 GB CPU host.

Usage (resumable; re-run the same command until it reports completion)::

    python -m ryko_lm.training.dpo_train \\
        --config configs/models/ryko-small.yaml \\
        --init-from artifacts/sft_ryko_small/final.pt \\
        --pairs data/training/ryko1000_dpo_train.jsonl \\
        --tokenizer artifacts/tokenizer/ryko-tokenizer.json \\
        --out-dir artifacts/dpo_ryko_small
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ..model.ryko_model import RykoConfig, RykoLM
from ..tokenizer.loader import RykoTokenizer
from .train import lr_at, make_optimizer

SCHEMA = "ryko1000-dpo-train/1"


def load_pairs(path: Path, tok: RykoTokenizer, block_size: int) -> list[dict]:
    """Tokenize every preference pair; count (never silently drop) rejects."""
    pairs: list[dict] = []
    skipped: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        p_ids = tok.encode(rec["prompt"])
        c_ids = tok.encode(rec["chosen"])
        r_ids = tok.encode(rec["rejected"])
        if not c_ids or not r_ids or not p_ids:
            skipped.append(rec["id"])
            continue
        if len(p_ids) + max(len(c_ids), len(r_ids)) > block_size:
            skipped.append(rec["id"])
            continue
        pairs.append({"id": rec["id"], "p": p_ids, "c": c_ids, "r": r_ids})
    return pairs


def _response_logprob(model: RykoLM, ids: torch.Tensor,
                      n_prompt: int) -> torch.Tensor:
    """Sum log-prob of response tokens; ``ids`` = prompt + response row."""
    logits, _ = model(ids)
    logprobs = torch.log_softmax(logits[0].float(), dim=-1)
    picked = logprobs.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)
    return picked[n_prompt - 1:].sum()


def _batch_logprobs(model: RykoLM, seqs: list[dict], resp_key: str,
                    pad_id: int) -> torch.Tensor:
    """Padded forward pass; returns summed response log-probs per row.

    Trailing PAD tokens sit after the response, so causal attention keeps
    the scored positions untouched (label masking is unnecessary by
    construction); PAD predictions are simply never gathered.
    """
    max_len = max(len(s["p"]) + len(s[resp_key]) for s in seqs)
    ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    n_prompt = torch.zeros(len(seqs), dtype=torch.long)
    row_len = torch.zeros(len(seqs), dtype=torch.long)
    for i, s in enumerate(seqs):
        row = s["p"] + s[resp_key]
        ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        n_prompt[i] = len(s["p"])
        row_len[i] = len(row)
    logits, _ = model(ids)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    picked = logprobs.gather(2, ids[:, 1:, None]).squeeze(2)  # (B, L-1)
    # picked[j] = logprob of token ids[j+1]; the response tokens span
    # positions n_prompt .. row_len-1, i.e. picked indices
    # n_prompt-1 .. row_len-2.  The mask is PER ROW — using the batch
    # max length would wrongly count trailing PAD log-probs.
    pos = torch.arange(max_len - 1).unsqueeze(0)
    mask = (pos >= (n_prompt - 1).unsqueeze(1)) \
        & (pos <= (row_len - 2).unsqueeze(1))
    return (picked * mask).sum(dim=1)


@torch.no_grad()
def dpo_accuracy(policy: RykoLM, pairs: list[dict], pad_id: int,
                 batch: int = 16) -> dict:
    """Forward-only policy preference accuracy over ``pairs``."""
    policy.eval()
    wins = 0
    margins: list[float] = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        c = _batch_logprobs(policy, chunk, "c", pad_id)
        r = _batch_logprobs(policy, chunk, "r", pad_id)
        wins += int((c > r).sum())
        margins.extend((c - r).tolist())
    policy.train()
    return {"accuracy": round(wins / max(1, len(pairs)), 4),
            "mean_margin": round(float(np.mean(margins)), 4) if margins else 0.0}


def dpo_train(args: argparse.Namespace) -> int:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = RykoConfig.from_dict(cfg["model"])
    pad_id = 0

    tok = RykoTokenizer.from_file(args.tokenizer)
    pairs = load_pairs(Path(args.pairs), tok, model_cfg.block_size)
    if not pairs:
        print("[error] no usable pairs", flush=True)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)

    policy = RykoLM(model_cfg)
    blob = torch.load(args.init_from, map_location="cpu", weights_only=True)
    state = blob["state_dict"] if "state_dict" in blob else blob["model"]
    state = {k: v.float() for k, v in state.items()}
    missing, unexpected = policy.load_state_dict(state, strict=False)
    missing = [k for k in missing
               if not (model_cfg.tie_embeddings and k == "lm_head.weight")]
    if missing or unexpected:
        raise RuntimeError(f"[init] incompatible checkpoint: "
                           f"missing={missing} unexpected={list(unexpected)}")
    if model_cfg.tie_embeddings:
        policy.lm_head.weight = policy.tok_emb.weight
    policy.train()
    ref = RykoLM(model_cfg)
    ref_missing, ref_unexpected = ref.load_state_dict(state, strict=False)
    allowed = {"lm_head.weight"} if model_cfg.tie_embeddings else set()
    if set(ref_missing) - allowed or ref_unexpected:
        raise RuntimeError(f"[ref] incompatible checkpoint: "
                           f"missing={ref_missing} "
                           f"unexpected={list(ref_unexpected)}")
    if model_cfg.tie_embeddings:
        ref.lm_head.weight = ref.tok_emb.weight  # restore the tie
    ref.eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)

    optimizer = make_optimizer(policy, args.lr, args.weight_decay)
    step, best_acc = 0, -1.0
    last_path = out_dir / "last.pt"
    if last_path.exists() and not args.reset:
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        step = ck["step"]
        best_acc = ck.get("best_acc", -1.0)
        rng = np.random.default_rng(args.seed + step)
        print(f"[resume] step={step} best_acc={best_acc}", flush=True)

    total_steps = min(args.epochs * math.ceil(len(pairs) / args.batch_size),
                      args.max_steps or 10**9)
    order = np.arange(len(pairs))
    deadline = time.time() + args.time_budget if args.time_budget else None
    t0 = time.time()
    start_step = step

    def save_last() -> None:
        ck = {"model": policy.state_dict(),
              "optimizer": optimizer.state_dict(), "step": step,
              "best_acc": best_acc, "total_steps": total_steps,
              "schema": SCHEMA}
        tmp = last_path.with_suffix(".pt.tmp")
        torch.save(ck, tmp)
        os.replace(tmp, last_path)

    print(f"[dpo] pairs={len(pairs)} batch={args.batch_size} "
          f"total_steps={total_steps} beta={args.beta}", flush=True)

    while step < total_steps:
        if deadline and time.time() > deadline:
            save_last()
            print(f"[budget] time budget reached at step {step}", flush=True)
            break
        idx = rng.choice(order, size=args.batch_size, replace=False) \
            if len(pairs) >= args.batch_size else order[:args.batch_size]
        chunk = [pairs[int(i)] for i in idx]
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            c_ref = _batch_logprobs(ref, chunk, "c", pad_id)
            r_ref = _batch_logprobs(ref, chunk, "r", pad_id)
        c_pol = _batch_logprobs(policy, chunk, "c", pad_id)
        r_pol = _batch_logprobs(policy, chunk, "r", pad_id)
        logits = args.beta * ((c_pol - c_ref) - (r_pol - r_ref))
        loss = -F.logsigmoid(logits).mean()
        loss.backward()
        lr_now = lr_at(step, total_steps, args.warmup, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()
        step += 1

        if step == 1 or step % 25 == 0 or step == total_steps:
            acc = float((logits > 0).float().mean())
            record = {"step": step, "loss": round(loss.item(), 4),
                      "batch_acc": round(acc, 3), "lr": round(lr_now, 8),
                      "elapsed_s": round(time.time() - t0, 1)}
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")
            print(f"step {step}/{total_steps} loss {loss.item():.4f} "
                  f"batch_acc {acc:.2f} lr {lr_now:.2e} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if step % args.save_every == 0 or step == total_steps:
            save_last()

    if step >= total_steps:
        policy.save(out_dir / "final.pt", half=True)
        summary = dpo_accuracy(policy, pairs, pad_id)
        summary.update({"status": "completed", "steps": step,
                        "pairs": len(pairs), "beta": args.beta,
                        "params": policy.num_params(),
                        "elapsed_s": round(time.time() - t0, 1)})
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[done] {json.dumps(summary)}", flush=True)
    else:
        print(f"[paused] step {step}/{total_steps} — rerun to resume",
              flush=True)
    return 0 if step >= total_steps else 3


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ryko_dpo_train",
        description="Phase-9 DPO training (resumable, CPU/GPU).")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--init-from", type=Path, required=True,
                   help="SFT checkpoint (policy + frozen reference origin)")
    p.add_argument("--pairs", type=Path,
                   default=Path("data/training/ryko1000_dpo_train.jsonl"))
    p.add_argument("--tokenizer", type=Path,
                   default=Path("artifacts/tokenizer/ryko-tokenizer.json"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/dpo_ryko_small"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--time-budget", type=int, default=None)
    p.add_argument("--reset", action="store_true")
    args = p.parse_args(argv)
    return dpo_train(args)


if __name__ == "__main__":
    sys.exit(main())
