"""Thin wrapper around the Anthropic SDK for dataset generation (structured outputs)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, TypeVar

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

GEN_MODEL = "claude-sonnet-5-5"
T = TypeVar("T", bound=BaseModel)

_client: anthropic.Anthropic | None = None
usage = {"input_tokens": 0, "output_tokens": 0}


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        load_dotenv()
        _client = anthropic.Anthropic(max_retries=4)
    return _client


def parse(prompt: str, schema: type[T], effort: str = "medium", model: str = GEN_MODEL) -> T:
    """One structured-output call. Raises on refusal."""
    common = dict(model=model, max_tokens=16000, output_config={"effort": effort},
                  messages=[{"role": "user", "content": prompt}], output_format=schema)
    if model.startswith("claude-haiku"):
        # Haiku has no server-side refusal fallback.
        resp = client().messages.parse(**common)
    else:
        # Server-side refusal fallback: if Sonnet 5.5 declines, the API retries on another model.
        resp = client().beta.messages.parse(**common, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refusal: {resp.stop_details}")
    usage["input_tokens"] += resp.usage.input_tokens
    usage["output_tokens"] += resp.usage.output_tokens
    return resp.parsed_output


def run_parallel(fn: Callable, items: list, workers: int = 8, desc: str = "") -> list:
    """Map fn over items with a thread pool. Failures are reported and returned as None."""
    out = [None] * len(items)
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for k, f in enumerate(as_completed(futs), 1):
            i = futs[f]
            try:
                out[i] = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"  [{desc}] item {i} failed: {e}")
            if k % 20 == 0 or k == len(items):
                print(f"  [{desc}] {k}/{len(items)}")
    return out


def cost_estimate(in_price: float = 2.0, out_price: float = 10.0) -> str:
    """Sonnet 5.5 list prices per MTok."""
    c = usage["input_tokens"] / 1e6 * in_price + usage["output_tokens"] / 1e6 * out_price
    return f"tokens in={usage['input_tokens']:,} out={usage['output_tokens']:,} (~${c:.2f} at Sonnet 5.5 prices)"
