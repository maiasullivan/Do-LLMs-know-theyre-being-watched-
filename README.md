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

- **Phase 0 (setup):** in progress.
