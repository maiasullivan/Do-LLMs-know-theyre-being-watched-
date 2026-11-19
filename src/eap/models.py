"""Per-model configuration.

Every script takes a model *key* (e.g. ``qwen3-4b``) and looks it up here, so
adding a model is a config change. Caches and results are keyed by the same key.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class ModelConfig:
    key: str
    hf_id: str
    revision: str  # pinned commit SHA, for reproducibility
    dtype: torch.dtype = torch.bfloat16
    # Extra kwargs passed to tokenizer.apply_chat_template.
    chat_template_kwargs: dict = field(default_factory=dict)
    gated: bool = False
    notes: str = ""


MODELS: dict[str, ModelConfig] = {
    "qwen3-4b": ModelConfig(
        key="qwen3-4b",
        hf_id="Qwen/Qwen3-4B",
        revision="1cfa9a7208912126459214e8b04321603b3df60c",
        # Without this, the template inserts <think> tags and moves the last-token position.
        chat_template_kwargs={"enable_thinking": False},
        notes="Text-only, Apache 2.0, not gated.",
    ),
    "gemma3-4b": ModelConfig(
        key="gemma3-4b",
        hf_id="google/gemma-3-4b-it",
        revision="093f9f388b31de276ce2de164bdc2081324b9767",
        # bf16 only: Gemma can overflow to NaN in fp16.
        gated=True,
        notes="Multimodal checkpoint (Gemma3ForConditionalGeneration); we use text only. "
        "Gemma 4 E4B is ~8B total params (~16 GB bf16), too big for a 16 GB Mac.",
    ),
    "llama3.2-3b": ModelConfig(
        key="llama3.2-3b",
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
        revision="0cb88a4f764b7a12671c53f0838cd831a0843b95",
        # The template adds a "Today Date" system block; pin it so it's constant.
        chat_template_kwargs={"date_string": "01 Jan 2026"},
        gated=True,
        notes="Newest small text Llama on the hub as of Nov 2026.",
    ),
}


def get_config(key: str) -> ModelConfig:
    try:
        return MODELS[key]
    except KeyError:
        raise SystemExit(f"Unknown model key {key!r}. Choose from: {', '.join(MODELS)}")


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(key: str, device: str | None = None):
    """Load (tokenizer, model, config) for a model key, in eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = get_config(key)
    device = device or default_device()
    tok = AutoTokenizer.from_pretrained(cfg.hf_id, revision=cfg.revision)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_id, revision=cfg.revision, dtype=cfg.dtype
    ).to(device)
    model.eval()
    return tok, model, cfg


def chat_prompt(tok, cfg: ModelConfig, user_message: str) -> str:
    """Render a single user turn with the model's chat template (generation prompt added)."""
    return tok.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
        **cfg.chat_template_kwargs,
    )
