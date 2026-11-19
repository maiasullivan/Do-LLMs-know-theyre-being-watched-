# eval-awareness-probe

**Does it know it's being watched?** Probing small open-weight language models for an internal representation of *being evaluated* vs *being deployed*. The project tests whether that representation goes beyond keywords like "test" to structural cues, real benchmark items and register.

This is a small-model, multi-family replication and extension of Nguyen et al. (2025), *Probing and Steering Evaluation Awareness of Language Models* (arXiv:2507.01786).

## Models

| Key | HF repo | Revision |
|---|---|---|
| `qwen3-4b` | `Qwen/Qwen3-4B` | `1cfa9a7` |
| `gemma3-4b` | `google/gemma-3-4b-it` | `093f9f3` |
| `llama3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4` |

Configs, including dtype and chat-template settings, are in [src/eap/models.py](src/eap/models.py).

## Setup

```bash
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY
hf auth login          # Gemma and Llama are gated: accept their licences on HF first
uv run python scripts/smoke_test.py --model qwen3-4b
```

All prompts are synthetic or come from public benchmarks.

## Status

- **Phase 0 (setup):** done for all three models on Apple M1 Pro (MPS, bf16). Hidden-state shapes are correct with no NaNs, and the user-token mask covers exactly the user message (Llama's date block and Qwen's empty think block are excluded). Decoder-layer paths are verified.
  - Qwen3-4B: 36 layers, d_model 2560
  - Gemma 3 4B: 34 layers, d_model 2560 (loads as `Gemma3ForConditionalGeneration`)
  - Llama 3.2 3B: 28 layers, d_model 3072
- **Phase 1 (dataset):** pipeline rebuilt for the factorial design (label × format × register, 6 template families per cell, split 4/2, tasks split 70/30) plus the explicit and benchmark sets. Dry-run passes on placeholder data, including the keyword grep. Waiting on an API key to generate the real tasks, contexts and benchmark rewrites.
