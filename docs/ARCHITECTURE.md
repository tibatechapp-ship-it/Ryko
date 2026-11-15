# Ryko architecture

## Model family

Ryko is a decoder-only autoregressive transformer in the Llama/Jais design
line, adapted for Arabic at small scale:

```
tokens (B, T)
   │
   ├─ token embedding  32,768 × 256            (tied with the LM head)
   │
   ├─ 4 × block:
   │     x += GQA-attention( RMSNorm(x) )       8 query heads, 4 KV heads
   │     x += SwiGLU( RMSNorm(x) )              hidden 768, no biases
   │
   ├─ RMSNorm
   └─ LM head (tied) → logits (B, T, 32,768)
```

### Component rationale

| component | choice | why |
|---|---|---|
| Normalization | **RMSNorm** (pre-norm) | stable without bias terms; cheaper than LayerNorm |
| Position | **RoPE** θ=10⁴ | relative encoding, extrapolates past the trained window; cache built on demand |
| Attention | **GQA** 8Q/4KV | KV cache and parameters shrink ~2× with negligible quality loss at this scale |
| FFN | **SwiGLU** (gate × up, down) | the modern standard (Llama/Jais); hidden sized 3× d rounded to 64 |
| Masking | `scaled_dot_product_attention(is_causal=True)` | no materialized T×T mask; uses PyTorch's fused kernel |
| Embedding | **tied** input/output | saves 8.4M parameters — decisive at 11.5M total |
| Init | N(0, 0.02); residual out-projections scaled by 1/√(2L) | GPT-2 recipe; verified: initial loss = ln(32,768) ≈ 10.4 |
| Decoding | KV-cache decode, `generate_iter` streaming sampler | temperature, top-k, top-p (nucleus), repetition penalty, seeded RNG |

### Configurations

| preset | params | layers | d_model | heads / KV | ffn | context | trained? |
|---|---|---|---|---|---|---|---|
| `ryko-tiny` | 11,536,640 | 4 | 256 | 8 / 4 | 768 | 256 | yes — CPU, 2,400 steps |
| `ryko-small` | 52,179,456 | 12 | 512 | 8 / 4 | 1,408 | 512 | config only (GPU) |
| `ryko-base` | 100,682,496 | 12 | 768 | 12 / 4 | 2,048 | 1,024 | config only (8× GPU) |

Configs live in `configs/models/*.yaml` (`model:` = architecture, `train:` =
optimization).  `RykoLM.save()` serializes config + state dict in one file;
`RykoLM.load()` reconstructs without any external registry.

## Data path

* corpus → `train.bin` / `val.bin`: flat `uint16` token arrays (32K vocab fits
  in 16 bits), read via `numpy.memmap`;
* the trainer samples random windows of `block_size` — **no padding, no
  attention masks**; documents start with `<bos>` and every paragraph ends
  with `<eos>`;
* this is the nanoGPT-style path: zero-copy, CPU-friendly, trivially
  shardable on GPU clusters.

## Training loop

* AdamW (β = 0.9/0.95), weight decay 0.1 on ≥2-D tensors only;
* cosine schedule with linear warmup (100 steps for tiny);
* gradient clipping 1.0, gradient accumulation supported;
* every `eval_every` steps: validation loss → `log.jsonl`, best/fp16 +
  last/resumable checkpoints (`last.pt` carries optimizer + RNG state);
* `--time-budget` enables graceful stop/resume — the full reference run was
  produced across resumable chunks on a 2-core CPU.

## Serving stack

FastAPI app (`src/ryko_lm/serving/api.py`):

* `POST /generate` — JSON in / JSON out (batch = 1);
* `POST /generate/stream` — **SSE**, one event per token with the cumulative
  decoded text (byte-safe full-decode-per-step), `done` event carries latency;
* `GET /model` — architecture + training metadata for the UI badges;
* `GET /` — RTL Arabic playground (single self-contained HTML file, no CDN);
* env knobs: `RYKO_MODEL_PATH`, `RYKO_TOKENIZER_DIR`, `RYKO_DEVICE`,
  `RYKO_THREADS`.

Measured on 2 CPU cores: ~5 ms/token decode with KV cache; the 12-token
round-trip `/generate` call completes in ~136 ms.

## Testing strategy

30 tests in `tests/`:

* **tokenizer** — special ids stable, roundtrip on hard inputs, no `unk` ever,
  diacritized input ≡ clean input, compression sanity band;
* **normalizer** — Python reference ≡ data-pipeline semantics, NFKC folding,
  tatweel/harakat removal, opt-in unifications;
* **model** — shapes, causality (future tokens cannot change past logits),
  KV-cache ≡ full forward (max Δ < 1e-4), tied weights, save/load determinism,
  preset sizing monotonicity;
* **serving** — health/model/generate/determinism-with-seed, SSE event count,
  playground HTML served;
* **data** — dedup, fingerprint stability, normalization semantics, binarize
  roundtrip.
