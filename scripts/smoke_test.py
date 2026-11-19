"""Phase 0 smoke test: load a model, run one chat-templated prompt, check hidden states.

Prints the templated prompt with the user-token mask shown as [[...]], so you can confirm
system and template tokens are excluded.

Usage: uv run python scripts/smoke_test.py --model qwen3-4b
"""

from __future__ import annotations

import argparse
import time

import torch

from eap.models import decoder_layers, encode_with_user_mask, load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    t0 = time.time()
    tok, model, cfg = load_model(args.model, args.device)
    device = next(model.parameters()).device
    print(f"Loaded {cfg.hf_id}@{cfg.revision[:10]} as {type(model).__name__} on {device} ({cfg.dtype}) in {time.time() - t0:.1f}s")

    msg = "What is the capital of Australia?"
    input_ids, mask = encode_with_user_mask(tok, cfg, msg)
    pieces = [tok.decode([t]) for t in input_ids[0]]
    shown = "".join(f"[[{p}]]" if m else p for p, m in zip(pieces, mask.tolist()))
    print("---- templated prompt (user-token mask in [[ ]]) ----")
    print(shown)
    print("-----------------------------------------------------")
    masked_text = "".join(p for p, m in zip(pieces, mask.tolist()) if m)
    assert masked_text.strip() == msg, f"user mask covers {masked_text!r}, expected {msg!r}"

    input_ids = input_ids.to(device)
    seq = input_ids.shape[1]
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)

    hs = torch.stack(out.hidden_states)[:, 0]  # (layers + 1, seq, d_model)
    text_cfg = getattr(model.config, "text_config", model.config)
    n_layers, d_model = text_cfg.num_hidden_layers, text_cfg.hidden_size
    print(f"hidden_states shape: {tuple(hs.shape)}  (expected ({n_layers + 1}, {seq}, {d_model}))")
    assert tuple(hs.shape) == (n_layers + 1, seq, d_model), "unexpected hidden-state shape"
    assert torch.isfinite(hs).all(), "NaN/inf in hidden states"
    assert len(decoder_layers(model, cfg)) == n_layers, f"layers_path {cfg.layers_path!r} has wrong length"
    print(f"layers_path {cfg.layers_path!r} -> {n_layers} decoder layers")
    norms = hs[:, -1].float().norm(dim=-1)
    print("last-token residual norm by layer:", [round(x, 1) for x in norms.tolist()])

    gen = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=20, do_sample=False)
    print("greedy completion:", repr(tok.decode(gen[0, seq:], skip_special_tokens=True)))
    print("OK")


if __name__ == "__main__":
    main()
