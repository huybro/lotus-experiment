"""
Side-by-side comparison of LOTUS vs Palimpzest FEVER experiments.

Runs all 4 pipelines on both systems with identical data, prompts, and model.
Outputs one CSV per pipeline to logs/.

Pipelines:
  1. filter          — single filter on pre-retrieved evidence
  2. filter_filter   — relevance filter → support filter
  3. map             — direct LLM verdict (no retrieval)
  4. map_filter      — generate search query → retrieve → filter

Each CSV has columns showing the exact prompt sent and output received
by each system, for every data tuple.
"""
import os
import re
import csv
import time

import lotus
import pandas as pd
import palimpzest as pz
from lotus.models import LM
from palimpzest.constants import Model
from palimpzest.query.processor.config import QueryProcessorConfig

from data_loader import load_fever_claims, load_oracle_wiki_kb
from retrieval import retrieve_for_claims
from universal_prompts import (
    get_prompt,
    install_prompt_overrides,
    install_pz_prompt_overrides,
)

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "qwen/Qwen1.5-0.5B-Chat"
DATASET_NAME = "fever"
N_CLAIMS = 20
K_RETRIEVAL = 3
MAX_TOKENS = 512
VLLM_API_BASE = "http://localhost:8000/v1"

# ============================================================
# Setup — LOTUS
# ============================================================
install_prompt_overrides()
lm = LM(model=f"hosted_vllm/{MODEL_NAME}", api_base=VLLM_API_BASE, max_tokens=MAX_TOKENS)
lotus.settings.configure(lm=lm)

# ============================================================
# Setup — PZ (patch is_vllm_model + litellm interceptor)
# ============================================================
install_pz_prompt_overrides()

import litellm as _litellm
_original_completion = _litellm.completion

# Global list that each PZ experiment clears and fills
pz_captured = []

def _rewrite_and_capture(*args, **kwargs):
    """Intercept PZ's litellm calls: rewrite messages to match LOTUS, capture I/O."""
    kwargs.setdefault("max_tokens", MAX_TOKENS)
    messages = kwargs.get("messages", args[1] if len(args) > 1 else [])

    # --- Try to extract claim + content (filter ops) ---
    claim_val = content_val = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        text = msg.get("content", "")
        cm = re.search(r'"claim":\s*"(.*?)"', text, re.DOTALL)
        co = re.search(r'"content":\s*"(.*?)"', text, re.DOTALL)
        if cm and co:
            claim_val = cm.group(1)
            content_val = co.group(1)
            break

    # --- Detect operation type and rebuild prompt ---
    rebuilt = False

    if claim_val and content_val:
        # Filter operation (has both claim + content)
        # Detect which filter instruction to use by scanning for keywords
        is_relevance_filter = False
        for msg in messages:
            if isinstance(msg, dict) and "relevant" in msg.get("content", "").lower():
                is_relevance_filter = True
                break

        if is_relevance_filter:
            instruction = (
                f"The following evidence is relevant to the claim.\n"
                f"Evidence: {content_val}\nClaim: {claim_val}"
            )
        else:
            instruction = (
                f"{content_val}\n"
                f"Based on the above evidence, the following claim is supported: {claim_val}"
            )
        new_messages = get_prompt(instruction, instruction, op='sem_filter')
        kwargs["messages"] = new_messages
        rebuilt = True

    elif claim_val and not content_val:
        # Map operation (has claim only)
        # Detect map type: search query generation vs verdict
        is_verdict = False
        for msg in messages:
            if isinstance(msg, dict):
                t = msg.get("content", "").lower()
                if "true" in t and "false" in t:
                    is_verdict = True
                    break

        if is_verdict:
            instruction = (
                f"Given the claim: {claim_val}\n"
                "Is this claim true or false based on your knowledge? "
                "Answer with exactly TRUE or FALSE, nothing else."
            )
        else:
            instruction = (
                f"Given the claim: {claim_val}\n"
                "Write a short factual search query to find evidence about this claim. "
                "Output only the search query, nothing else."
            )
        new_messages = get_prompt(instruction, claim_val, op='sem_map')
        kwargs["messages"] = new_messages
        rebuilt = True

    if rebuilt and len(args) > 1:
        args = (args[0],) + (kwargs["messages"],) + args[2:]

    # Capture the final prompt
    final_msgs = kwargs.get("messages", messages)
    prompt_text = "\n".join(
        m.get("content", "") for m in final_msgs if isinstance(m, dict)
    )

    result = _original_completion(*args, **kwargs)
    output_text = result.choices[0].message.content if result.choices else ""
    pz_captured.append({"input": prompt_text, "output": output_text})
    return result

_litellm.completion = _rewrite_and_capture

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
# Load Data (shared)
# ============================================================
claims_df = load_fever_claims(n=N_CLAIMS)
wiki_df = load_oracle_wiki_kb(claims_split="labelled_dev", n_claims=N_CLAIMS)
claims_df["true_label"] = claims_df["label"].apply(lambda l: l == "SUPPORTS")

print("\n[Shared] Retrieving evidence...")
joined_df = retrieve_for_claims(claims_df, wiki_df, query_col="claim", K=K_RETRIEVAL)
print(f"  Total tuples: {len(joined_df)}")

os.makedirs("logs", exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def write_csv(filepath, rows):
    """Write a list of dicts to CSV."""
    if not rows:
        print(f"  ⚠️  No rows to write to {filepath}")
        return
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✅ Wrote {len(rows)} rows → {filepath}")


def capture_lotus_prompts(df, instruction, op='sem_filter'):
    """Generate LOTUS prompts for each row (for logging, not execution)."""
    prompts = []
    for _, row in df.iterrows():
        filled = instruction.format(**row)
        msgs = get_prompt(filled, filled, op=op)
        prompts.append("\n".join(m["content"] for m in msgs))
    return prompts


# ============================================================
# Pipeline 1: filter only
# ============================================================
print("\n\n" + "=" * 60)
print("  PIPELINE 1: filter")
print("=" * 60)

FILTER_INSTRUCTION = (
    "{content}\n"
    "Based on the above evidence, the following claim is supported: {claim}"
)

# -- LOTUS --
lotus_filter_prompts = capture_lotus_prompts(joined_df, FILTER_INSTRUCTION, 'sem_filter')
t0 = time.time()
df_lotus_f = joined_df.copy().sem_filter(FILTER_INSTRUCTION)
lotus_filter_time = time.time() - t0
lotus_filter_passed = set(df_lotus_f.index.tolist())
print(f"  LOTUS filter: {len(df_lotus_f)} passed ({lotus_filter_time:.1f}s)")

# -- PZ --
pz_captured.clear()
ds = pz.MemoryDataset(id="cmp-filter", vals=joined_df.to_dict("records"))
ds = ds.sem_filter(
    "Based on the evidence, the following claim is supported: the claim states that {claim}",
    depends_on=["content", "claim"],
)
t0 = time.time()
pz_out = ds.run(config=pz_config)
pz_filter_time = time.time() - t0
pz_df = pz_out.to_df()
print(f"  PZ filter: {len(pz_df)} passed ({pz_filter_time:.1f}s)")

# -- CSV --
rows = []
for i in range(len(joined_df)):
    rows.append({
        "tuple": i,
        "claim": joined_df.iloc[i]["claim"][:80],
        "evidence": joined_df.iloc[i]["content"][:80],
        "lotus_input": lotus_filter_prompts[i] if i < len(lotus_filter_prompts) else "",
        "lotus_output": "PASS" if i in lotus_filter_passed else "FAIL",
        "pz_input": pz_captured[i]["input"] if i < len(pz_captured) else "",
        "pz_output": pz_captured[i]["output"] if i < len(pz_captured) else "",
    })
write_csv("logs/comparison_filter.csv", rows)


# ============================================================
# Pipeline 2: filter → filter
# ============================================================
print("\n\n" + "=" * 60)
print("  PIPELINE 2: filter → filter")
print("=" * 60)

FILTER1_INSTRUCTION = (
    "The following evidence is relevant to the claim.\n"
    "Evidence: {content}\nClaim: {claim}"
)
FILTER2_INSTRUCTION = FILTER_INSTRUCTION  # same support-check filter

# -- LOTUS --
lotus_f1_prompts = capture_lotus_prompts(joined_df, FILTER1_INSTRUCTION, 'sem_filter')
t0 = time.time()
df_f1 = joined_df.copy().sem_filter(FILTER1_INSTRUCTION)
lotus_f1_passed = set(df_f1.index.tolist())
lotus_f2_prompts = capture_lotus_prompts(df_f1, FILTER2_INSTRUCTION, 'sem_filter')
df_f2 = df_f1.sem_filter(FILTER2_INSTRUCTION)
lotus_ff_time = time.time() - t0
lotus_f2_passed = set(df_f2.index.tolist())
print(f"  LOTUS filter→filter: {len(df_f1)}→{len(df_f2)} passed ({lotus_ff_time:.1f}s)")

# -- PZ --
pz_captured.clear()
ds2 = pz.MemoryDataset(id="cmp-ff", vals=joined_df.to_dict("records"))
ds2 = ds2.sem_filter(
    "The following evidence is relevant to the claim",
    depends_on=["content", "claim"],
)
t0 = time.time()
pz_f1_out = ds2.run(config=pz_config)
pz_f1_df = pz_f1_out.to_df()
pz_f1_captured = list(pz_captured)

pz_captured.clear()
if len(pz_f1_df) > 0:
    ds2b = pz.MemoryDataset(id="cmp-ff2", vals=pz_f1_df.to_dict("records"))
    ds2b = ds2b.sem_filter(
        "Based on the evidence, the following claim is supported: the claim states that {claim}",
        depends_on=["content", "claim"],
    )
    pz_f2_out = ds2b.run(config=pz_config)
    pz_f2_df = pz_f2_out.to_df()
else:
    pz_f2_df = pd.DataFrame()
pz_ff_time = time.time() - t0
pz_f2_captured = list(pz_captured)
print(f"  PZ filter→filter: {len(pz_f1_df)}→{len(pz_f2_df)} passed ({pz_ff_time:.1f}s)")

# -- CSV --
rows = []
for i in range(len(joined_df)):
    rows.append({
        "tuple": i,
        "claim": joined_df.iloc[i]["claim"][:80],
        "evidence": joined_df.iloc[i]["content"][:80],
        "lotus_f1_input": lotus_f1_prompts[i] if i < len(lotus_f1_prompts) else "",
        "lotus_f1_output": "PASS" if i in lotus_f1_passed else "FAIL",
        "pz_f1_input": pz_f1_captured[i]["input"] if i < len(pz_f1_captured) else "",
        "pz_f1_output": pz_f1_captured[i]["output"] if i < len(pz_f1_captured) else "",
        "lotus_f2_input": lotus_f2_prompts[i - min(lotus_f1_passed)] if i in lotus_f1_passed and (i - min(lotus_f1_passed)) < len(lotus_f2_prompts) else "SKIPPED",
        "lotus_f2_output": "PASS" if i in lotus_f2_passed else ("FAIL" if i in lotus_f1_passed else "SKIPPED"),
        "pz_f2_input": "",  # F2 only runs on F1-passed tuples; hard to align
        "pz_f2_output": "",
    })
write_csv("logs/comparison_filter_filter.csv", rows)


# ============================================================
# Pipeline 3: map only (direct verdict, no retrieval)
# ============================================================
print("\n\n" + "=" * 60)
print("  PIPELINE 3: map only")
print("=" * 60)

MAP_INSTRUCTION = (
    "Given the claim: {claim}\n"
    "Is this claim true or false based on your knowledge? "
    "Answer with exactly TRUE or FALSE, nothing else."
)

# -- LOTUS --
lotus_map_prompts = capture_lotus_prompts(claims_df, MAP_INSTRUCTION, 'sem_map')
t0 = time.time()
df_map = claims_df.copy().sem_map(MAP_INSTRUCTION, suffix="verdict")
lotus_map_time = time.time() - t0
print(f"  LOTUS map: {len(df_map)} rows ({lotus_map_time:.1f}s)")

# -- PZ --
pz_captured.clear()
ds3 = pz.MemoryDataset(
    id="cmp-map",
    vals=claims_df[["id", "claim", "label", "true_label"]].to_dict("records"),
)
ds3 = ds3.sem_map(
    cols=[{"name": "verdict", "type": str,
           "desc": "TRUE if the claim is factually correct, FALSE otherwise. Answer with exactly TRUE or FALSE."}],
    depends_on=["claim"],
)
t0 = time.time()
pz_map_out = ds3.run(config=pz_config)
pz_map_time = time.time() - t0
pz_map_df = pz_map_out.to_df()
print(f"  PZ map: {len(pz_map_df)} rows ({pz_map_time:.1f}s)")

# -- CSV --
rows = []
for i in range(len(claims_df)):
    rows.append({
        "tuple": i,
        "claim": claims_df.iloc[i]["claim"][:80],
        "lotus_input": lotus_map_prompts[i] if i < len(lotus_map_prompts) else "",
        "lotus_output": df_map.iloc[i]["verdict"] if i < len(df_map) else "",
        "pz_input": pz_captured[i]["input"] if i < len(pz_captured) else "",
        "pz_output": pz_captured[i]["output"] if i < len(pz_captured) else "",
    })
write_csv("logs/comparison_map.csv", rows)


# ============================================================
# Pipeline 4: map → filter
# ============================================================
print("\n\n" + "=" * 60)
print("  PIPELINE 4: map → filter")
print("=" * 60)

MAP_QUERY_INSTRUCTION = (
    "Given the claim: {claim}\n"
    "Write a short factual search query to find evidence about this claim. "
    "Output only the search query, nothing else."
)

# -- LOTUS --
lotus_mq_prompts = capture_lotus_prompts(claims_df, MAP_QUERY_INSTRUCTION, 'sem_map')
t0 = time.time()
df_mf = claims_df.copy().sem_map(MAP_QUERY_INSTRUCTION, suffix="search_query")
mf_retrieved = retrieve_for_claims(df_mf, wiki_df, query_col="search_query", K=K_RETRIEVAL)
lotus_mf_filter_prompts = capture_lotus_prompts(mf_retrieved, FILTER_INSTRUCTION, 'sem_filter')
df_mf_verified = mf_retrieved.sem_filter(FILTER_INSTRUCTION)
lotus_mf_time = time.time() - t0
lotus_mf_passed = set(df_mf_verified["id"].tolist())
print(f"  LOTUS map→filter: map={len(df_mf)}, filter={len(df_mf_verified)} passed ({lotus_mf_time:.1f}s)")

# -- PZ --
pz_captured.clear()
ds4 = pz.MemoryDataset(
    id="cmp-mf-map",
    vals=claims_df[["id", "claim", "label", "true_label"]].to_dict("records"),
)
ds4 = ds4.sem_map(
    cols=[{"name": "search_query", "type": str,
           "desc": "A short factual search query to find evidence about the claim"}],
    depends_on=["claim"],
)
t0 = time.time()
pz_mf_map_out = ds4.run(config=pz_config)
pz_mf_map_df = pz_mf_map_out.to_df()
pz_map_captured = list(pz_captured)

# Retrieve using PZ-generated queries
pz_mf_claims = claims_df.copy()
pz_mf_claims["search_query"] = pz_mf_map_df["search_query"].tolist()[:len(pz_mf_claims)]
pz_mf_retrieved = retrieve_for_claims(pz_mf_claims, wiki_df, query_col="search_query", K=K_RETRIEVAL)

pz_captured.clear()
ds4f = pz.MemoryDataset(id="cmp-mf-filter", vals=pz_mf_retrieved.to_dict("records"))
ds4f = ds4f.sem_filter(
    "Based on the evidence, the following claim is supported: the claim states that {claim}",
    depends_on=["content", "claim"],
)
pz_mf_filter_out = ds4f.run(config=pz_config)
pz_mf_time = time.time() - t0
pz_mf_filter_df = pz_mf_filter_out.to_df()
pz_filter_captured = list(pz_captured)
print(f"  PZ map→filter: map={len(pz_mf_map_df)}, filter={len(pz_mf_filter_df)} passed ({pz_mf_time:.1f}s)")

# -- CSV (map stage) --
rows = []
for i in range(len(claims_df)):
    rows.append({
        "tuple": i,
        "claim": claims_df.iloc[i]["claim"][:80],
        "lotus_map_input": lotus_mq_prompts[i] if i < len(lotus_mq_prompts) else "",
        "lotus_map_output": df_mf.iloc[i]["search_query"] if i < len(df_mf) else "",
        "pz_map_input": pz_map_captured[i]["input"] if i < len(pz_map_captured) else "",
        "pz_map_output": pz_map_captured[i]["output"] if i < len(pz_map_captured) else "",
    })
write_csv("logs/comparison_map_filter.csv", rows)


# ============================================================
# Summary
# ============================================================
print("\n\n" + "=" * 60)
print("  ALL COMPARISONS COMPLETE")
print("=" * 60)
print(f"  logs/comparison_filter.csv")
print(f"  logs/comparison_filter_filter.csv")
print(f"  logs/comparison_map.csv")
print(f"  logs/comparison_map_filter.csv")
