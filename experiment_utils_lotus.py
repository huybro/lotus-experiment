"""
Shared setup for FEVER experiment pipelines.

Provides:
- Configuration constants and instruction templates
- LOTUS and Palimpzest initialization
- litellm interceptor for prompt capture/rewriting
- Helper functions (write_csv, find_match, pz_map_with_fallback)
- Data loading from pre-retrieved CSV

Usage:
    from experiment_utils import (
        state, joined_df, pz_config, pz,
        FILTER_RELEVANCE, FILTER_SUPPORT, MAP_VERDICT,
        write_csv, find_match, pz_map_with_fallback,
    )
"""
import os
import csv
import lotus
import pandas as pd
import litellm as _litellm
from lotus.models import LM

from universal_prompts import override_lotus_prompt

from transformers import AutoTokenizer


# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_TOKENS = 512
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
VLLM_API_BASE = "http://localhost:8003/v1"

# ============================================================
# Instruction templates
# ============================================================

# Filter: is the evidence relevant to the claim?
FILTER_RELEVANCE = (
    "{claim} {content} The evidence can determine whether the claim is true or false\n"
)

# Filter: does the evidence support the claim?
FILTER_SUPPORT = (
    "{content}\n"
    "Based on the above evidence, the following claim is supported: {claim}"
)

# Map: based on both claim and evidence, generate a verdict (Claim first, Evidence second)
MAP_VERDICT = (
    "Based on the following claim and evidence, determine if the claim is true or false.\n"
    "Claim: {claim}\nEvidence: {content}\n"
    "Answer with exactly TRUE or FALSE, nothing else."
)

# Filter: verify verdict against evidence and claim (used after sem_map)
FILTER_VERDICT = (
    "Based on the evidence and verdict, the claim is correctly assessed.\n"
    "Evidence: {content}\nClaim: {claim}\nVerdict: {verdict}"
)


# ============================================================
# Interceptor state (mutable, shared across pipeline steps)
# ============================================================
logger = []

# ============================================================
# Setup — LOTUS
# ============================================================
override_lotus_prompt()
_lotus_lm = LM(
    model=f"hosted_vllm/{MODEL_NAME}",
    api_base=VLLM_API_BASE,
    max_tokens=MAX_TOKENS,
    temperature=0,
)
lotus.settings.configure(lm=_lotus_lm)



_original_completion = _litellm.completion
 


def _interceptor(*args, **kwargs):
    kwargs.setdefault("max_tokens", MAX_TOKENS)
    kwargs.setdefault("temperature", 0)
    kwargs.setdefault("seed", 42)

    messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    result = _original_completion(*args, **kwargs)
    output_text = result.choices[0].message.content if result.choices else ""

    logger.append({"input": prompt_text, "output": output_text})
    return result


_litellm.completion = _interceptor


