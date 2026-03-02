"""
Side-by-side comparison of LOTUS vs Palimpzest FEVER experiments.

Runs all 4 pipelines on both systems with identical data, prompts, and model.
Outputs one CSV per pipeline to logs/.

Pipelines:
  1. filter          — single filter on pre-retrieved evidence
  2. filter_filter   — relevance filter → support filter
  3. map             — direct LLM verdict (no retrieval)
  4. map_filter      — generate search query → retrieve → filter

Each CSV has columns showing the exact prompt sent and raw output received
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
# Litellm interceptor — captures I/O for both LOTUS and PZ
# ============================================================
import litellm as _litellm
_original_completion = _litellm.completion

captured = []          # list of {"input": str, "output": str}
rewrite_mode = False   # True when running PZ (rewrite prompts), False for LOTUS (capture only)

# Current filter instruction for PZ rewriting (set before each PZ run)
current_filter_instruction = None


def _interceptor(*args, **kwargs):
    """
    Universal interceptor for litellm.completion.
    - Always injects max_tokens and captures prompt/output.
    - When rewrite_mode=True (PZ), rewrites PZ messages to match LOTUS format.
    """
    kwargs.setdefault("max_tokens", MAX_TOKENS)
    messages = kwargs.get("messages", args[1] if len(args) > 1 else [])

    if rewrite_mode:
        # --- PZ MODE: extract fields and rebuild prompt ---
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
            elif cm and not co:
                claim_val = cm.group(1)
                break

        if claim_val and content_val and current_filter_instruction:
            # Filter operation — use the exact instruction LOTUS uses
            instruction = current_filter_instruction.format(
                content=content_val, claim=claim_val
            )
            new_messages = get_prompt(instruction, instruction, op='sem_filter')
            kwargs["messages"] = new_messages
            if len(args) > 1:
                args = (args[0],) + (new_messages,) + args[2:]
        elif claim_val and not content_val:
            # Map operation — detect type from PZ's message
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
            if len(args) > 1:
                args = (args[0],) + (new_messages,) + args[2:]

    # Capture the final prompt
    final_msgs = kwargs.get("messages", messages)
    prompt_text = "\n".join(
        m.get("content", "") for m in final_msgs if isinstance(m, dict)
    )

    result = _original_completion(*args, **kwargs)
    output_text = result.choices[0].message.content if result.choices else ""
    captured.append({"input": prompt_text, "output": output_text})
    return result

_litellm.completion = _interceptor

# ============================================================
# Setup — LOTUS
# ============================================================
install_prompt_overrides()
lm = LM(model=f"hosted_vllm/{MODEL_NAME}", api_base=VLLM_API_BASE, max_tokens=MAX_TOKENS)
lotus.settings.configure(lm=lm)

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
    if not rows:
        print(f"  ⚠️  No rows to write to {filepath}")
        return
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✅ Wrote {len(rows)} rows → {filepath}")


# Filter instructions (used by both LOTUS and PZ interceptor)
FILTER_SUPPORT = (
    "{content}\n"
    "Based on the above evidence, the following claim is supported: {claim}"
)
FILTER_RELEVANCE = (
    "The following evidence is relevant to the claim.\n"
    "Evidence: {content}\nClaim: {claim}"
)
MAP_VERDICT = (
    "Given the claim: {claim}\n"
    "Is this claim true or false based on your knowledge? "
    "Answer with exactly TRUE or FALSE, nothing else."
)
MAP_QUERY = (
    "Given the claim: {claim}\n"
    "Write a short factual search query to find evidence about this claim. "
    "Output only the search query, nothing else."
)


# ============================================================
# Pipeline 1: filter only
# ============================================================
print("\n\n" + "=" * 60)
print("  PIPELINE 1: filter")
print("=" * 60)

# -- LOTUS --
rewrite_mode = False
captured.clear()
t0 = time.time()
df_lotus_f = joined_df.copy().sem_filter(FILTER_SUPPORT)
lotus_filter_time = time.time() - t0
lotus_captured = list(captured)
print(f"  LOTUS filter: {len(df_lotus_f)} passed / {len(joined_df)} total ({lotus_filter_time:.1f}s)")

# -- PZ --
rewrite_mode = True
current_filter_instruction = FILTER_SUPPORT
captured.clear()
ds = pz.MemoryDataset(id="cmp-filter", vals=joined_df.to_dict("records"))
ds = ds.sem_filter(
    "Based on the evidence, the following claim is supported: the claim states that {claim}",
    depends_on=["content", "claim"],
)
t0 = time.time()
pz_out = ds.run(config=pz_config)
pz_filter_time = time.time() - t0
pz_df = pz_out.to_df()
pz_captured = list(captured)
rewrite_mode = False
print(f"  PZ filter: {len(pz_df)} passed / {len(joined_df)} total ({pz_filter_time:.1f}s)")

# -- CSV --
rows = []
for i in range(len(joined_df)):
    rows.append({
        "tuple": i,
        "claim": joined_df.iloc[i]["claim"][:80],
        "evidence": joined_df.iloc[i]["content"][:80],
        "lotus_input": lotus_captured[i]["input"] if i < len(lotus_captured) else "",
        "lotus_output": lotus_captured[i]["output"] if i < len(lotus_captured) else "",
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

# -- LOTUS --
rewrite_mode = False
captured.clear()
t0 = time.time()
df_f1 = joined_df.copy().sem_filter(FILTER_RELEVANCE)
lotus_f1_captured = list(captured)

captured.clear()
df_f2 = df_f1.sem_filter(FILTER_SUPPORT)
lotus_f2_captured = list(captured)
lotus_ff_time = time.time() - t0
print(f"  LOTUS filter→filter: {len(df_f1)}→{len(df_f2)} passed ({lotus_ff_time:.1f}s)")

# -- PZ filter 1 --
rewrite_mode = True
current_filter_instruction = FILTER_RELEVANCE
captured.clear()
ds2 = pz.MemoryDataset(id="cmp-ff1", vals=joined_df.to_dict("records"))
ds2 = ds2.sem_filter(
    "The following evidence is relevant to the claim",
    depends_on=["content", "claim"],
)
t0 = time.time()
pz_f1_out = ds2.run(config=pz_config)
pz_f1_df = pz_f1_out.to_df()
pz_f1_captured = list(captured)

# -- PZ filter 2 --
current_filter_instruction = FILTER_SUPPORT
captured.clear()
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
pz_f2_captured = list(captured)
rewrite_mode = False
print(f"  PZ filter→filter: {len(pz_f1_df)}→{len(pz_f2_df)} passed ({pz_ff_time:.1f}s)")

# -- CSV --
rows = []
for i in range(len(joined_df)):
    rows.append({
        "tuple": i,
        "claim": joined_df.iloc[i]["claim"][:80],
        "evidence": joined_df.iloc[i]["content"][:80],
        "lotus_f1_input": lotus_f1_captured[i]["input"] if i < len(lotus_f1_captured) else "",
        "lotus_f1_output": lotus_f1_captured[i]["output"] if i < len(lotus_f1_captured) else "",
        "pz_f1_input": pz_f1_captured[i]["input"] if i < len(pz_f1_captured) else "",
        "pz_f1_output": pz_f1_captured[i]["output"] if i < len(pz_f1_captured) else "",
    })
write_csv("logs/comparison_filter_filter.csv", rows)


# ============================================================
# Pipeline 3: map only (direct verdict, no retrieval)
# ============================================================
print("\n\n" + "=" * 60)
print("  PIPELINE 3: map only")
print("=" * 60)

# -- LOTUS --
rewrite_mode = False
captured.clear()
t0 = time.time()
df_map = claims_df.copy().sem_map(MAP_VERDICT, suffix="verdict")
lotus_map_time = time.time() - t0
lotus_map_captured = list(captured)
print(f"  LOTUS map: {len(df_map)} rows ({lotus_map_time:.1f}s)")

# -- PZ --
rewrite_mode = True
captured.clear()
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
pz_map_out = ds3.run(config=pz_config, max_quality=True)
pz_map_time = time.time() - t0
pz_map_df = pz_map_out.to_df()
pz_map_captured = list(captured)
rewrite_mode = False
print(f"  PZ map: {len(pz_map_df)} rows ({pz_map_time:.1f}s)")

# -- CSV --
rows = []
for i in range(len(claims_df)):
    rows.append({
        "tuple": i,
        "claim": claims_df.iloc[i]["claim"][:80],
        "lotus_input": lotus_map_captured[i]["input"] if i < len(lotus_map_captured) else "",
        "lotus_output": lotus_map_captured[i]["output"] if i < len(lotus_map_captured) else "",
        "pz_input": pz_map_captured[i]["input"] if i < len(pz_map_captured) else "",
        "pz_output": pz_map_captured[i]["output"] if i < len(pz_map_captured) else "",
    })
write_csv("logs/comparison_map.csv", rows)


# ============================================================
# Pipeline 4: map → filter
# ============================================================
print("\n\n" + "=" * 60)
print("  PIPELINE 4: map → filter")
print("=" * 60)

# -- LOTUS map --
rewrite_mode = False
captured.clear()
t0 = time.time()
df_mf = claims_df.copy().sem_map(MAP_QUERY, suffix="search_query")
lotus_mq_captured = list(captured)
mf_retrieved = retrieve_for_claims(df_mf, wiki_df, query_col="search_query", K=K_RETRIEVAL)

# -- LOTUS filter --
captured.clear()
df_mf_verified = mf_retrieved.sem_filter(FILTER_SUPPORT)
lotus_mf_filter_captured = list(captured)
lotus_mf_time = time.time() - t0
print(f"  LOTUS map→filter: map={len(df_mf)}, filter={len(df_mf_verified)} passed ({lotus_mf_time:.1f}s)")

# -- PZ map --
rewrite_mode = True
captured.clear()
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
pz_mf_map_out = ds4.run(config=pz_config, max_quality=True)
pz_mf_map_df = pz_mf_map_out.to_df()
pz_mq_captured = list(captured)

# Retrieve using PZ-generated queries
pz_mf_claims = claims_df.copy()
pz_mf_claims["search_query"] = pz_mf_map_df["search_query"].tolist()[:len(pz_mf_claims)]
pz_mf_retrieved = retrieve_for_claims(pz_mf_claims, wiki_df, query_col="search_query", K=K_RETRIEVAL)

# -- PZ filter --
current_filter_instruction = FILTER_SUPPORT
captured.clear()
ds4f = pz.MemoryDataset(id="cmp-mf-filter", vals=pz_mf_retrieved.to_dict("records"))
ds4f = ds4f.sem_filter(
    "Based on the evidence, the following claim is supported: the claim states that {claim}",
    depends_on=["content", "claim"],
)
pz_mf_filter_out = ds4f.run(config=pz_config)
pz_mf_filter_df = pz_mf_filter_out.to_df()
pz_mf_filter_captured = list(captured)
pz_mf_time = time.time() - t0
rewrite_mode = False
print(f"  PZ map→filter: map={len(pz_mf_map_df)}, filter={len(pz_mf_filter_df)} passed ({pz_mf_time:.1f}s)")

# -- CSV (map + filter stages) --
rows = []
for i in range(len(claims_df)):
    rows.append({
        "tuple": i,
        "claim": claims_df.iloc[i]["claim"][:80],
        "lotus_map_input": lotus_mq_captured[i]["input"] if i < len(lotus_mq_captured) else "",
        "lotus_map_output": lotus_mq_captured[i]["output"] if i < len(lotus_mq_captured) else "",
        "pz_map_input": pz_mq_captured[i]["input"] if i < len(pz_mq_captured) else "",
        "pz_map_output": pz_mq_captured[i]["output"] if i < len(pz_mq_captured) else "",
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
