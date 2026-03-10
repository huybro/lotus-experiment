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

import palimpzest as pz
import litellm as _litellm
from palimpzest.constants import Model
from palimpzest.query.processor.config import QueryProcessorConfig

from palimpzest.prompts import prompt_factory
from data_utils import write_csv, load_fever


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
    "The sentence can determine whether the claim is true or false.\n"
    "Answer TRUE if the context is sufficient to judge the claim, and FALSE otherwise.\n"
    "Output TRUE or FALSE only."
)

# Filter: does the evidence support the claim?
FILTER_SUPPORT = (
    "{content}\n"
    "Based on the above evidence, the following claim is supported: {claim}"
)

# Map: based on both claim and evidence, generate a verdict (Claim first, Evidence second)

MAP_VERDICT = (
    "Explain how the claim can be supported by the evidence.\n"
    "Provide a short explanation in natural language."
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

prompt_factory.CUSTOM_MAP_INSTRUCTION_FUNC = _get_map_instruction

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

# FEVER joined_df reused across PZ pipelines
FEVER_PATH = os.path.join("data", "fever_claims_with_evidence.csv")
joined_df = load_fever(FEVER_PATH)


def find_match(row, cap_list):
    """
    Find the corresponding captured prompt for a source row.
    Palimpzest processes asynchronously, so cap_list is out of order.
    """
    for cap in cap_list:
        if row["claim"] in cap["input"] and str(row["content"]) in cap["input"]:
            return cap
    return {"input": "", "output": "", "claim": "", "content": ""}


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
            from palimpzest.prompts.prompt_utils import get_prompt
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
