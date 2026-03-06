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
import re
import csv
import time

import lotus
import pandas as pd
import palimpzest as pz
import litellm as _litellm
from lotus.models import LM
from palimpzest.constants import Model
from palimpzest.query.processor.config import QueryProcessorConfig

from universal_prompts import (
    get_prompt,
    lotus_df2text_row,
    install_prompt_overrides,
    install_pz_prompt_overrides,
)


def nle2str(nle, cols):
    """Replicate LOTUS's nle2str: replace {col} with col.capitalize()."""
    d = {col: col.capitalize() for col in cols}
    return nle.format(**d)


# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MAX_TOKENS = 512
VLLM_API_BASE = "http://localhost:8000/v1"

# ============================================================
# Instruction templates
# ============================================================

# Filter: is the evidence relevant to the claim?
FILTER_RELEVANCE = (
    "The following evidence is relevant to the claim.\n"
    "Evidence: {content}\nClaim: {claim}"
)

# Filter: does the evidence support the claim?
FILTER_SUPPORT = (
    "{content}\n"
    "Based on the above evidence, the following claim is supported: {claim}"
)

# Map: based on both evidence and claim, generate a verdict
MAP_VERDICT = (
    "Based on the following evidence, determine if the claim is true or false.\n"
    "Evidence: {content}\nClaim: {claim}\n"
    "Answer with exactly TRUE or FALSE, nothing else."
)


# ============================================================
# Interceptor state (mutable, shared across pipeline steps)
# ============================================================
class _State:
    """Mutable state for the litellm interceptor."""
    def __init__(self):
        self.captured = []
        self.rewrite_mode = False
        self.current_filter_instruction = None
        self.current_filter_cols = None
        self.debug = False  # Set to False to silence prompt logging
        self.call_count = 0

state = _State()


# ============================================================
# Setup — LOTUS
# ============================================================
install_prompt_overrides()
_lotus_lm = LM(
    model=f"hosted_vllm/{MODEL_NAME}",
    api_base=VLLM_API_BASE,
    max_tokens=MAX_TOKENS,
    temperature=0,
)
lotus.settings.configure(lm=_lotus_lm)


# ============================================================
# litellm interceptor — captures LOTUS + rewrites PZ
# ============================================================
_original_completion = _litellm.completion


def _interceptor(*args, **kwargs):
    kwargs.setdefault("max_tokens", MAX_TOKENS)
    kwargs.setdefault("temperature", 0)
    messages = kwargs.get("messages", args[1] if len(args) > 1 else [])

    claim_val = content_val = None

    # Try PZ JSON format
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        text = msg.get("content", "")
        cm = re.search(r'"claim":\s*"(.*?)"', text, re.DOTALL)
        co = re.search(r'"content":\s*"(.*?)"', text, re.DOTALL)
        if cm and co:
            claim_val, content_val = cm.group(1), co.group(1)
            break
        elif cm:
            claim_val = cm.group(1)
            break

    # Try LOTUS format
    if not claim_val:
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            text = msg.get("content", "")
            m = re.search(r'\[Claim\]:\s*«(.*?)»', text, re.DOTALL)
            if m:
                claim_val = m.group(1)
            m2 = re.search(r'\[Content\]:\s*«(.*?)»', text, re.DOTALL)
            if m2:
                content_val = m2.group(1)
            if claim_val:
                break

    # PZ mode: rewrite messages
    if state.rewrite_mode:
        rebuilt = False

        if claim_val and content_val and state.current_filter_instruction:
            cols = state.current_filter_cols or ["content", "claim"]
            data = lotus_df2text_row({"content": content_val, "claim": claim_val}, cols)
            formatted_instr = nle2str(state.current_filter_instruction, cols)
            kwargs["messages"] = get_prompt(formatted_instr, data, op='sem_filter')
            rebuilt = True

        elif claim_val and content_val and not state.current_filter_instruction:
            cols = ["content", "claim"]
            data = lotus_df2text_row({"content": content_val, "claim": claim_val}, cols)
            formatted_instr = nle2str(MAP_VERDICT, cols)
            kwargs["messages"] = get_prompt(formatted_instr, data, op='sem_map')
            rebuilt = True

        elif claim_val and not content_val and state.current_filter_instruction:
            cols = state.current_filter_cols or ["claim"]
            data = lotus_df2text_row({"claim": claim_val}, cols)
            formatted_instr = nle2str(state.current_filter_instruction, cols)
            kwargs["messages"] = get_prompt(formatted_instr, data, op='sem_filter')
            rebuilt = True

        elif claim_val and not content_val:
            data = lotus_df2text_row({"claim": claim_val}, ["claim"])
            formatted_instr = nle2str(MAP_VERDICT, ["claim"])
            kwargs["messages"] = get_prompt(formatted_instr, data, op='sem_map')
            rebuilt = True

        if rebuilt and len(args) > 1:
            args = (args[0],) + (kwargs["messages"],) + args[2:]

    # Capture
    final_msgs = kwargs.get("messages", messages)
    prompt_text = "\n".join(
        m.get("content", "") for m in final_msgs if isinstance(m, dict)
    )

    # Debug: print the prompt before sending
    if state.debug:
        state.call_count += 1
        source = "PZ" if state.rewrite_mode else "LOTUS"
        print(f"\n{'─'*70}")
        print(f"  LLM CALL #{state.call_count} [{source}]")
        print(f"{'─'*70}")
        for msg in final_msgs:
            if isinstance(msg, dict):
                role = msg.get('role', '?')
                content = msg.get('content', '')
                print(f"  [{role}] {content[:500]}{'...' if len(content) > 500 else ''}")
        print(f"{'─'*70}")

    result = _original_completion(*args, **kwargs)
    output_text = result.choices[0].message.content if result.choices else ""

    # Debug: print the response
    if state.debug:
        print(f"  → RESPONSE: {output_text[:200]}{'...' if len(output_text) > 200 else ''}")
        print(f"{'─'*70}\n")

    state.captured.append({
        "input": prompt_text,
        "output": output_text,
        "claim": claim_val or "",
        "content": content_val or "",
    })
    return result


_litellm.completion = _interceptor


# ============================================================
# Setup — PZ
# ============================================================
install_pz_prompt_overrides()

PZ_MODEL = Model(f"hosted_vllm/{MODEL_NAME}")
PZ_MODEL.api_base = VLLM_API_BASE
pz_config = QueryProcessorConfig(
    api_base=VLLM_API_BASE,
    available_models=[PZ_MODEL],
    allow_model_selection=False,
    allow_bonded_query=False,
    allow_mixtures=False,
    allow_critic=False,
    allow_split_merge=False,
    verbose=False,
)


# ============================================================
# Load Data
# ============================================================
DATA_PATH = "data/fever_claims_with_evidence.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Run 'python prepare_data.py' first to generate it."
    )
joined_df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(joined_df)} (claim, evidence) pairs from {DATA_PATH}")

os.makedirs("logs", exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def write_csv(filepath, rows):
    if not rows:
        return
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def find_match(cap_list, claim, content=None):
    claim_short = claim[:40]
    for entry in cap_list:
        if claim_short in entry.get("claim", ""):
            if content is None or content[:40] in entry.get("content", ""):
                return entry
    for entry in cap_list:
        if claim_short in entry.get("input", ""):
            if content is None or content[:40] in entry.get("input", ""):
                return entry
    return {"input": "", "output": "", "claim": "", "content": ""}


def pz_map_with_fallback(instruction, data_df, col_name, pz_desc, cols_used):
    """Run PZ sem_map, falling back to direct LLM calls if optimizer crashes."""
    try:
        ds = pz.MemoryDataset(
            id=f"cmp-{col_name}",
            vals=data_df.to_dict("records"),
        )
        ds = ds.sem_map(
            cols=[{"name": col_name, "type": str, "desc": pz_desc}],
            depends_on=cols_used,
        )
        result = ds.run(config=pz_config, max_quality=True)
        return result.to_df()
    except Exception:
        outputs = []
        for _, row in data_df.iterrows():
            data_dict = {c: row[c] for c in cols_used}
            data = lotus_df2text_row(data_dict, cols_used)
            instr = nle2str(instruction, cols_used)
            msgs = get_prompt(instr, data, op='sem_map')
            res = _original_completion(
                model=f"hosted_vllm/{MODEL_NAME}",
                messages=msgs,
                max_tokens=MAX_TOKENS,
                temperature=0,
                api_base=VLLM_API_BASE,
            )
            output = res.choices[0].message.content if res.choices else ""
            outputs.append(output.strip())
            state.captured.append({
                "input": "\n".join(m.get("content", "") for m in msgs if isinstance(m, dict)),
                "output": output,
                "claim": row.get("claim", ""),
                "content": row.get("content", ""),
            })
        df = data_df.copy()
        df[col_name] = outputs
        return df
