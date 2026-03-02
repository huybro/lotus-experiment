"""
Side-by-side comparison of LOTUS vs Palimpzest FEVER experiments.

Both systems use the professor's get_prompt() format:
  - LOTUS is monkey-patched via install_prompt_overrides()
  - PZ is intercepted at litellm.completion and rewritten

Outputs one CSV per pipeline to logs/.
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
MODEL_NAME = "qwen/Qwen1.5-0.5B-Chat"
DATASET_NAME = "fever"
N_CLAIMS = 20
K_RETRIEVAL = 3
MAX_TOKENS = 512
VLLM_API_BASE = "http://localhost:8000/v1"

# ============================================================
# Setup — LOTUS (monkey-patch to use get_prompt)
# ============================================================
install_prompt_overrides()
lm = LM(model=f"hosted_vllm/{MODEL_NAME}", api_base=VLLM_API_BASE, max_tokens=MAX_TOKENS, temperature=0)
lotus.settings.configure(lm=lm)

# ============================================================
# litellm interceptor — captures LOTUS + rewrites PZ
# ============================================================
import litellm as _litellm
_original_completion = _litellm.completion

captured = []
rewrite_mode = False   # False=LOTUS (capture only), True=PZ (rewrite using get_prompt)

# Set before each PZ filter run: the raw instruction template LOTUS uses
current_filter_instruction = None
# Set before each PZ filter run: column names for data formatting
current_filter_cols = None

def _interceptor(*args, **kwargs):
    kwargs.setdefault("max_tokens", MAX_TOKENS)
    kwargs.setdefault("temperature", 0)
    messages = kwargs.get("messages", args[1] if len(args) > 1 else [])

    # --- Extract claim + content from PZ's messages ---
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
        elif cm:
            claim_val = cm.group(1)
            break

    # --- Also try to extract from LOTUS format for capture ---
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

    # --- PZ mode: rewrite to get_prompt() format ---
    if rewrite_mode:
        rebuilt = False

        if claim_val and content_val and current_filter_instruction:
            # Build data in LOTUS's [Column]: «value» format
            cols = current_filter_cols or ["content", "claim"]
            data = lotus_df2text_row(
                {"content": content_val, "claim": claim_val},
                cols
            )
            # Apply nle2str: {content} → Content, {claim} → Claim
            formatted_instr = nle2str(current_filter_instruction, cols)
            new_messages = get_prompt(formatted_instr, data, op='sem_filter')
            kwargs["messages"] = new_messages
            rebuilt = True

        elif claim_val and not content_val:
            # Map operation
            is_verdict = any(
                isinstance(m, dict) and "true" in m.get("content", "").lower()
                and "false" in m.get("content", "").lower()
                for m in messages
            )
            data = lotus_df2text_row({"claim": claim_val}, ["claim"])
            if is_verdict:
                instruction = (
                    "Given the claim: {claim}\n"
                    "Is this claim true or false based on your knowledge? "
                    "Answer with exactly TRUE or FALSE, nothing else."
                )
            else:
                instruction = (
                    "Given the claim: {claim}\n"
                    "Write a short factual search query to find evidence about this claim. "
                    "Output only the search query, nothing else."
                )
            formatted_instr = nle2str(instruction, ["claim"])
            new_messages = get_prompt(formatted_instr, data, op='sem_map')
            kwargs["messages"] = new_messages
            rebuilt = True

        if rebuilt and len(args) > 1:
            args = (args[0],) + (kwargs["messages"],) + args[2:]

    # --- Capture ---
    final_msgs = kwargs.get("messages", messages)
    prompt_text = "\n".join(
        m.get("content", "") for m in final_msgs if isinstance(m, dict)
    )
    result = _original_completion(*args, **kwargs)
    output_text = result.choices[0].message.content if result.choices else ""
    captured.append({
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


def find_match(cap_list, claim, content=None):
    claim_short = claim[:40]
    for entry in cap_list:
        if claim_short in entry.get("claim", ""):
            if content is None:
                return entry
            if content[:40] in entry.get("content", ""):
                return entry
    for entry in cap_list:
        if claim_short in entry.get("input", ""):
            if content is None or content[:40] in entry.get("input", ""):
                return entry
    return {"input": "", "output": "", "claim": "", "content": ""}


# Filter instructions (raw templates — {column} stays as column name reference)
FILTER_SUPPORT = (
    "{content}\n"
    "Based on the above evidence, the following claim is supported: {claim}"
)
FILTER_RELEVANCE = (
    "The following evidence is relevant to the claim.\n"
    "Evidence: {content}\nClaim: {claim}"
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
print(f"  LOTUS filter: {len(df_lotus_f)} passed ({lotus_filter_time:.1f}s)")

# -- PZ --
rewrite_mode = True
current_filter_instruction = FILTER_SUPPORT
current_filter_cols = ["content", "claim"]
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
pz_captured_f = list(captured)
rewrite_mode = False
print(f"  PZ filter: {len(pz_df)} passed ({pz_filter_time:.1f}s)")

# -- CSV --
rows = []
for i in range(len(joined_df)):
    row = joined_df.iloc[i]
    lotus_match = find_match(lotus_captured, row["claim"], row["content"])
    pz_match = find_match(pz_captured_f, row["claim"], row["content"])
    rows.append({
        "tuple": i,
        "claim": row["claim"][:80],
        "evidence": row["content"][:80],
        "lotus_input": lotus_match["input"],
        "lotus_output": lotus_match["output"],
        "pz_input": pz_match["input"],
        "pz_output": pz_match["output"],
    })
write_csv("logs/comparison_filter.csv", rows)


# ============================================================
# Pipeline 2: filter → filter
# ============================================================
print("\n\n" + "=" * 60)
print("  PIPELINE 2: filter → filter")
print("=" * 60)

# -- LOTUS F1 --
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

# -- PZ F1 --
rewrite_mode = True
current_filter_instruction = FILTER_RELEVANCE
current_filter_cols = ["content", "claim"]
captured.clear()
ds2 = pz.MemoryDataset(id="cmp-ff", vals=joined_df.to_dict("records"))
ds2 = ds2.sem_filter(
    "The following evidence is relevant to the claim",
    depends_on=["content", "claim"],
)
t0 = time.time()
pz_f1_out = ds2.run(config=pz_config)
pz_f1_df = pz_f1_out.to_df()
pz_f1_captured = list(captured)

# -- PZ F2 --
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
    row = joined_df.iloc[i]
    lotus_f1_match = find_match(lotus_f1_captured, row["claim"], row["content"])
    pz_f1_match = find_match(pz_f1_captured, row["claim"], row["content"])
    lotus_f2_match = find_match(lotus_f2_captured, row["claim"], row["content"])
    pz_f2_match = find_match(pz_f2_captured, row["claim"], row["content"])
    rows.append({
        "tuple": i,
        "claim": row["claim"][:80],
        "evidence": row["content"][:80],
        "lotus_f1_input": lotus_f1_match["input"],
        "lotus_f1_output": lotus_f1_match["output"],
        "pz_f1_input": pz_f1_match["input"],
        "pz_f1_output": pz_f1_match["output"],
        "lotus_f2_input": lotus_f2_match["input"],
        "lotus_f2_output": lotus_f2_match["output"],
        "pz_f2_input": pz_f2_match["input"],
        "pz_f2_output": pz_f2_match["output"],
    })
write_csv("logs/comparison_filter_filter.csv", rows)


# ============================================================
# Pipeline 3: map only
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
rewrite_mode = False
captured.clear()
t0 = time.time()
df_map = claims_df.copy().sem_map(MAP_INSTRUCTION, suffix="verdict")
lotus_map_time = time.time() - t0
lotus_map_captured = list(captured)
print(f"  LOTUS map: {len(df_map)} rows ({lotus_map_time:.1f}s)")

# -- PZ --
rewrite_mode = True
captured.clear()
try:
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
except Exception as e:
    print(f"  ⚠️  PZ sem_map optimizer failed: {e}")
    print("  Falling back to direct LLM calls...")
    t0 = time.time()
    verdicts = []
    for _, row in claims_df.iterrows():
        data = lotus_df2text_row({"claim": row["claim"]}, ["claim"])
        instr = nle2str(MAP_INSTRUCTION, ["claim"])
        msgs = get_prompt(instr, data, op='sem_map')
        result = _original_completion(
            model=f"hosted_vllm/{MODEL_NAME}",
            messages=msgs,
            max_tokens=MAX_TOKENS,
            temperature=0,
            api_base=VLLM_API_BASE,
        )
        output = result.choices[0].message.content if result.choices else ""
        verdicts.append(output.strip())
        captured.append({
            "input": "\n".join(m.get("content", "") for m in msgs if isinstance(m, dict)),
            "output": output,
            "claim": row["claim"],
            "content": "",
        })
    pz_map_time = time.time() - t0
    pz_map_df = claims_df.copy()
    pz_map_df["verdict"] = verdicts
pz_map_captured = list(captured)
rewrite_mode = False
print(f"  PZ map: {len(pz_map_df)} rows ({pz_map_time:.1f}s)")

# -- CSV --
rows = []
for i in range(len(claims_df)):
    row = claims_df.iloc[i]
    lotus_match = find_match(lotus_map_captured, row["claim"])
    pz_match = find_match(pz_map_captured, row["claim"])
    rows.append({
        "tuple": i,
        "claim": row["claim"][:80],
        "lotus_input": lotus_match["input"],
        "lotus_output": lotus_match["output"],
        "pz_input": pz_match["input"],
        "pz_output": pz_match["output"],
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

# -- LOTUS map --
rewrite_mode = False
captured.clear()
t0 = time.time()
df_mf = claims_df.copy().sem_map(MAP_QUERY_INSTRUCTION, suffix="search_query")
lotus_mq_captured = list(captured)
mf_retrieved = retrieve_for_claims(df_mf, wiki_df, query_col="search_query", K=K_RETRIEVAL)

captured.clear()
df_mf_verified = mf_retrieved.sem_filter(FILTER_SUPPORT)
lotus_mf_filter_captured = list(captured)
lotus_mf_time = time.time() - t0
print(f"  LOTUS map→filter: map={len(df_mf)}, filter={len(df_mf_verified)} passed ({lotus_mf_time:.1f}s)")

# -- PZ map --
rewrite_mode = True
captured.clear()
try:
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
except Exception as e:
    print(f"  ⚠️  PZ sem_map optimizer failed: {e}")
    print("  Falling back to direct LLM calls...")
    t0 = time.time()
    queries = []
    for _, row in claims_df.iterrows():
        data = lotus_df2text_row({"claim": row["claim"]}, ["claim"])
        instr = nle2str(MAP_QUERY_INSTRUCTION, ["claim"])
        msgs = get_prompt(instr, data, op='sem_map')
        result = _original_completion(
            model=f"hosted_vllm/{MODEL_NAME}",
            messages=msgs,
            max_tokens=MAX_TOKENS,
            temperature=0,
            api_base=VLLM_API_BASE,
        )
        output = result.choices[0].message.content if result.choices else ""
        queries.append(output.strip())
        captured.append({
            "input": "\n".join(m.get("content", "") for m in msgs if isinstance(m, dict)),
            "output": output,
            "claim": row["claim"],
            "content": "",
        })
    pz_mf_map_df = claims_df.copy()
    pz_mf_map_df["search_query"] = queries
pz_mq_captured = list(captured)

pz_mf_claims = claims_df.copy()
pz_mf_claims["search_query"] = pz_mf_map_df["search_query"].tolist()[:len(pz_mf_claims)]
pz_mf_retrieved = retrieve_for_claims(pz_mf_claims, wiki_df, query_col="search_query", K=K_RETRIEVAL)

current_filter_instruction = FILTER_SUPPORT
current_filter_cols = ["content", "claim"]
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

rows = []
for i in range(len(claims_df)):
    row = claims_df.iloc[i]
    lotus_match = find_match(lotus_mq_captured, row["claim"])
    pz_match = find_match(pz_mq_captured, row["claim"])
    rows.append({
        "tuple": i,
        "claim": row["claim"][:80],
        "lotus_map_input": lotus_match["input"],
        "lotus_map_output": lotus_match["output"],
        "pz_map_input": pz_match["input"],
        "pz_map_output": pz_match["output"],
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
