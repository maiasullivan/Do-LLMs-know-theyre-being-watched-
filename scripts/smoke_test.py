"""Phase 0 smoke test: load a model, run one chat-templated prompt, check hidden states.

Usage: uv run python scripts/smoke_test.py --model qwen3-4b
"""

from __future__ import annotations

import argparse
import time

import torch

from eap.models import chat_prompt, load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    t0 = time.time()
    tok, model, cfg = load_model(args.model, args.device)
    device = next(model.parameters()).device
    print(f"Loaded {cfg.hf_id}@{cfg.revision[:10]} on {device} ({cfg.dtype}) in {time.time() - t0:.1f}s")

    text = chat_prompt(tok, cfg, "What is the capital of Australia?")
    print("---- templated prompt ----")
    print(text)
    print("--------------------------")

    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    seq = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    hs = torch.stack(out.hidden_states)  # (layers + 1, batch, seq, d_model)
    hs = hs[:, 0]
    text_cfg = getattr(model.config, "text_config", model.config)
    n_layers, d_model = text_cfg.num_hidden_layers, text_cfg.hidden_size

    print(f"hidden_states shape: {tuple(hs.shape)}  (expected ({n_layers + 1}, {seq}, {d_model}))")
    assert tuple(hs.shape) == (n_layers + 1, seq, d_model), "unexpected hidden-state shape"
    assert torch.isfinite(hs).all(), "NaN/inf in hidden states"
    norms = hs[:, -1].float().norm(dim=-1)
    print("last-token residual norm by layer:", [round(x, 1) for x in norms.tolist()])

    gen = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print("greedy completion:", repr(tok.decode(gen[0, seq:], skip_special_tokens=True)))
    print("OK")


if __name__ == "__main__":
    main()
