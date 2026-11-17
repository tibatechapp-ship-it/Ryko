"""CPU training throughput benchmark for ryko-small — decision evidence.

Measures real forward+backward tok/s for the ryko-small architecture at a
few (batch, block) points on this machine so the phase-5 SFT decision
(train now on CPU vs wait for GPU) is made with measured numbers.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from ryko_lm.model.ryko_model import RykoConfig, RykoLM

CFG = dict(vocab_size=32768, block_size=512, n_layer=12, n_head=8, n_kv_head=4,
           n_embd=512, ffn_hidden=1408, dropout=0.0, rope_theta=10000.0,
           tie_embeddings=True)

torch.set_num_threads(2)


def bench(batch: int, block: int, steps: int = 3) -> None:
    cfg = RykoConfig.from_dict({**CFG, "block_size": block})
    model = RykoLM(cfg)
    model.train()
    x = torch.randint(0, 32768, (batch, block))
    y = torch.randint(0, 32768, (batch, block))
    # warmup
    _, loss = model(x, y)
    loss.backward()
    model.zero_grad(set_to_none=True)
    t0 = time.time()
    for _ in range(steps):
        _, loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    dt = time.time() - t0
    tok_s = batch * block * steps / dt
    print(f"batch={batch} block={block}: {dt/steps:.2f}s/step  "
          f"{tok_s:,.0f} tok/s", flush=True)
    del model


if __name__ == "__main__":
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}")
    for batch, block in ((2, 256), (4, 256), (4, 512), (8, 256)):
        bench(batch, block)
