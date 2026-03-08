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
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_TOKENS = 512
VLLM_API_BASE = "http://localhost:8003/v1"

# ============================================================
# Instruction templates
# ============================================================

# Filter: is the evidence relevant to the claim?
FILTER_RELEVANCE = (
    "The following evidence is relevant to the claim.\n"
    "Claim: {claim} Evidence: {content}\n"
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
class _State:
    """Mutable state for the litellm interceptor."""
    def __init__(self):
        self.captured = []
        self.rewrite_mode = False
        self.current_filter_instruction = None
        self.current_filter_cols = None
        self.current_map_instruction = None
        self.current_map_cols = None  # input fields for map (e.g. ["claim", "content"])
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
# litellm interceptor — captures PZ prompts (no rewrite needed)
# ============================================================
_original_completion = _litellm.completion


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def _interceptor(*args, **kwargs):
    kwargs.setdefault("max_tokens", MAX_TOKENS)
    kwargs.setdefault("temperature", 0)
    kwargs.setdefault("seed", 42)

    messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    result = _original_completion(*args, **kwargs)
    output_text = result.choices[0].message.content if result.choices else ""

    state.captured.append({"input": prompt_text, "output": output_text})
    return result


_litellm.completion = _interceptor


# ============================================================
# Setup — PZ
# ============================================================
def _get_map_instruction(input_fields):
    """Return Lotus-style map instruction when set (for FEVER alignment with Lotus)."""
    if state.current_map_instruction and state.current_map_cols:
        return nle2str(state.current_map_instruction, state.current_map_cols)
    return None


install_pz_prompt_overrides(get_map_instruction=_get_map_instruction)

PZ_MODEL = Model(f"hosted_vllm/{MODEL_NAME}")
PZ_MODEL.api_base = VLLM_API_BASE
pz_config = QueryProcessorConfig(
    api_base=VLLM_API_BASE,
    available_models=[PZ_MODEL],
    allow_model_selection=False,
    allow_bonded_query=True,  # Use direct LLM (LLMConvertBonded), not RAG
    allow_rag_reduction=False,  # Disable RAG (needs OpenAI embeddings)
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
# FEVER format: add [Claim]/[Evidence] prefixes for Lotus/PZ alignment
joined_df["claim"] = "[Claim] " + joined_df["claim"]
joined_df["content"] = "[Evidence] " + joined_df["content"]
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



def pz_map_with_fallback(instruction, data_df, col_name, pz_desc, cols_used):
    """Run PZ sem_map, falling back to direct LLM calls if optimizer crashes."""
    state.current_map_instruction = instruction
    state.current_map_cols = cols_used
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
            data = "".join(str(data_dict[c]) for c in cols_used)  # Lotus df2text format
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
