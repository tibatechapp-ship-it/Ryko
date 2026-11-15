"""Ryko — Arabic decoder-only language model.

Modern GPT-style transformer, tuned for Arabic morphology:

* Byte-Level BPE embeddings (32,768) with optional weight tying — Arabic
  vocabularies are large, so tying saves ~8-25M parameters at small scale.
* Rotary Position Embeddings (RoPE) — better length extrapolation than
  learned absolute positions.
* RMSNorm (pre-norm) — stable training without bias terms.
* SwiGLU feed-forward — the Llama/Jais family standard.
* Grouped-Query Attention (GQA) — fewer KV heads shrink both parameters and
  inference cache without hurting quality at this scale.
* Causal attention via ``scaled_dot_product_attention`` (no materialized
  masks), optional KV cache for generation.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RykoConfig:
    vocab_size: int = 32768
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 8
    n_kv_head: int = 4
    n_embd: int = 256
    ffn_hidden: int = 768
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RykoConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, theta: float = 10000.0):
    """Return cos/sin tables of shape (seq_len, head_dim//2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate half-dimension pairs.  x: (B, H, T, D); cos/sin: (T, D//2)."""
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: RykoConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.attn_dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:  # incremental decoding
            if kv_cache[0] is not None:
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)
            kv_cache[0], kv_cache[1] = k, v

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=(seq_len == k.size(2)),
            enable_gqa=True,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: RykoConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: RykoConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class RykoLM(nn.Module):
    def __init__(self, cfg: RykoConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.register_buffer("rope_cos", torch.zeros(0), persistent=False)
        self.register_buffer("rope_sin", torch.zeros(0), persistent=False)
        self._maybe_extend_rope(cfg.block_size)
        self.apply(self._init_weights)
        # scaled init for residual-out projections (GPT-2 recipe)
        for name, param in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _maybe_extend_rope(self, needed: int) -> None:
        current = int(self.rope_cos.numel())
        if needed > current:
            target = max(needed, self.cfg.block_size)
            cos, sin = build_rope_cache(target, self.head_dim, self.cfg.rope_theta)
            self.rope_cos, self.rope_sin = cos, sin

    @property
    def head_dim(self) -> int:
        return self.cfg.n_embd // self.cfg.n_head

    def num_params(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.tok_emb.weight.numel()
        return total

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: list[list[torch.Tensor]] | None = None,
    ):
        bsz, seq_len = idx.shape
        past = 0
        if kv_cache is not None and kv_cache[0][0] is not None:
            past = kv_cache[0][0].size(2)
        if past + seq_len > self.rope_cos.size(0):
            self._maybe_extend_rope(past + seq_len)
        cos = self.rope_cos[past: past + seq_len]
        sin = self.rope_sin[past: past + seq_len]

        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            cache = kv_cache[i] if kv_cache is not None else None
            x = block(x, cos, sin, cache)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    # ------------------------------------------------------------ generation
    @torch.no_grad()
    def generate_iter(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        repetition_penalty: float = 1.1,
        eos_id: int | None = 2,
        generator: torch.Generator | None = None,
    ):
        """Yield sampled tokens one at a time (enables SSE streaming).

        The prompt prefill populates the KV cache; every following step is a
        single-token decode.  A prompt is trimmed so that prompt + generated
        tokens fit inside the trained context window when possible.
        """
        self.eval()
        idx = idx.clone()
        room = self.cfg.block_size - max_new_tokens
        if room >= 1 and idx.size(1) > room:
            idx = idx[:, -room:]

        kv_cache: list[list[torch.Tensor]] = [[None, None] for _ in range(self.cfg.n_layer)]
        logits, _ = self(idx, kv_cache=kv_cache)
        logits = logits[:, -1, :]

        generated: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            if repetition_penalty != 1.0:
                history = torch.cat([idx] + ([torch.cat(generated, dim=1)] if generated else []), dim=1)
                for row in range(history.size(0)):
                    seen = torch.unique(history[row])
                    score = logits[row, seen]
                    logits[row, seen] = torch.where(
                        score > 0, score / repetition_penalty, score * repetition_penalty
                    )

            logits = logits / max(temperature, 1e-5)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs_sorted = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs_sorted, dim=-1)
                cut = cumulative - probs_sorted > top_p
                sorted_logits = sorted_logits.masked_fill(cut, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1, generator=generator)
            yield next_tok
            generated.append(next_tok)
            if eos_id is not None and bool((next_tok == eos_id).all()):
                return
            idx = torch.cat([idx, next_tok], dim=1)
            logits, _ = self(next_tok, kv_cache=kv_cache)  # single-token decode step
            logits = logits[:, -1, :]

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        repetition_penalty: float = 1.1,
        eos_id: int | None = 2,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Non-streaming generation on top of :meth:`generate_iter`."""
        generated = list(
            self.generate_iter(
                idx, max_new_tokens=max_new_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                eos_id=eos_id, generator=generator,
            )
        )
        return torch.cat(generated, dim=1) if generated else idx

    # ------------------------------------------------------------- save/load
    def save(self, path: str | Path, half: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            k: v.detach().cpu().half() if half else v.detach().cpu()
            for k, v in self.state_dict().items()
        }
        if self.cfg.tie_embeddings:
            state.pop("lm_head.weight")  # tied to tok_emb.weight — do not duplicate
        # atomic write: a kill mid-save must never truncate the destination
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({"config": self.cfg.to_dict(), "state_dict": state}, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu",
             config: "RykoConfig | None" = None) -> "RykoLM":
        """Load a model saved via :meth:`save`, or a training checkpoint.

        Training checkpoints (``last.pt``) store ``{"model": state_dict, ...}``
        without an embedded config — pass ``config=`` explicitly for those.
        """
        blob = torch.load(Path(path), map_location=device, weights_only=True)
        if config is None:
            config = RykoConfig.from_dict(blob["config"])
        state = blob["state_dict"] if "state_dict" in blob else blob["model"]
        model = cls(config)
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_missing = {"lm_head.weight"} if model.cfg.tie_embeddings else set()
        if set(missing) - allowed_missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
        if model.cfg.tie_embeddings:
            model.lm_head.weight = model.tok_emb.weight  # restore the tie
        model.to(device)
        model.eval()
        return model


def get_preset(name: str) -> RykoConfig:
    """Named architecture presets used across configs and tests."""
    presets = {
        "tiny": RykoConfig(),
        "small": RykoConfig(n_layer=12, n_head=8, n_kv_head=4, n_embd=512,
                            ffn_hidden=1408, block_size=512),
        "base": RykoConfig(n_layer=12, n_head=12, n_kv_head=4, n_embd=768,
                           ffn_hidden=2048, block_size=1024),
    }
    if name not in presets:
        raise KeyError(f"unknown preset '{name}', choose from {sorted(presets)}")
    return presets[name]
