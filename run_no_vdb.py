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
N_CLAIMS = 20
MAX_TOKENS = 512
VLLM_API_BASE = "http://localhost:8000/v1"

# ============================================================
# Instruction templates
# ============================================================

# Map: generate a verdict (TRUE/FALSE) based on world knowledge
MAP_VERDICT = (
    "Given the claim: {claim}\n"
    "Is this claim true or false based on your knowledge? "
    "Answer with exactly TRUE or FALSE, nothing else."
)

# Filter: does the claim mention a person?
FILTER_PERSON = (
    "Does the following claim mention a specific person by name?\n"
    "Claim: {claim}\n"
    "Answer TRUE if it mentions a person, FALSE otherwise."
)

# Filter: is the evidence relevant to the claim?
FILTER_SUPPORT = (
    "{content}\n"
    "Based on the above evidence, the following claim is supported: {claim}"
)

# ============================================================
# Setup — LOTUS
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
rewrite_mode = False
current_filter_instruction = None
current_filter_cols = None


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
    if rewrite_mode:
        rebuilt = False

        if claim_val and content_val and current_filter_instruction:
            # Filter with evidence
            cols = current_filter_cols or ["content", "claim"]
            data = lotus_df2text_row({"content": content_val, "claim": claim_val}, cols)
            formatted_instr = nle2str(current_filter_instruction, cols)
            kwargs["messages"] = get_prompt(formatted_instr, data, op='sem_filter')
            rebuilt = True

        elif claim_val and not content_val and current_filter_instruction:
            # Filter on claim only
            cols = current_filter_cols or ["claim"]
            data = lotus_df2text_row({"claim": claim_val}, cols)
            formatted_instr = nle2str(current_filter_instruction, cols)
            kwargs["messages"] = get_prompt(formatted_instr, data, op='sem_filter')
            rebuilt = True

        elif claim_val and not content_val:
            # Map operation
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
# Load Data (evidence is pre-retrieved and stored in DataFrame)
# ============================================================
claims_df = load_fever_claims(n=N_CLAIMS)
claims_df["true_label"] = claims_df["label"].apply(lambda l: l == "SUPPORTS")

# Pre-retrieve evidence and join with claims
# Evidence is baked into the data — no vector DB calls during pipeline execution
K_RETRIEVAL = 3
wiki_df = load_oracle_wiki_kb(claims_split="labelled_dev", n_claims=N_CLAIMS)
joined_df = retrieve_for_claims(claims_df, wiki_df, query_col="claim", K=K_RETRIEVAL)
joined_df.to_csv("data/fever_claims_with_evidence.csv", index=False)

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


def find_match(cap_list, claim, extra=None):
    claim_short = claim[:40]
    for entry in cap_list:
        if claim_short in entry.get("claim", ""):
            if extra is None or extra[:40] in entry.get("content", ""):
                return entry
    for entry in cap_list:
        if claim_short in entry.get("input", ""):
            if extra is None or extra[:40] in entry.get("input", ""):
                return entry
    return {"input": "", "output": "", "claim": "", "content": ""}


def pz_map_with_fallback(instruction, claims, col_name, pz_desc):
    """Run PZ sem_map, falling back to direct LLM calls if optimizer crashes."""
    try:
        ds = pz.MemoryDataset(
            id=f"cmp-{col_name}",
            vals=claims[["id", "claim", "label", "true_label"]].to_dict("records"),
        )
        ds = ds.sem_map(
            cols=[{"name": col_name, "type": str, "desc": pz_desc}],
            depends_on=["claim"],
        )
        result = ds.run(config=pz_config, max_quality=True)
        return result.to_df()
    except Exception as e:
        outputs = []
        for _, row in claims.iterrows():
            data = lotus_df2text_row({"claim": row["claim"]}, ["claim"])
            instr = nle2str(instruction, ["claim"])
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
            captured.append({
                "input": "\n".join(m.get("content", "") for m in msgs if isinstance(m, dict)),
                "output": output,
                "claim": row["claim"],
                "content": "",
            })
        df = claims.copy()
        df[col_name] = outputs
        return df


# ============================================================
# Pipeline 1: filter only (does claim mention a person?)
# ============================================================
print("\n" + "=" * 60)
print("  PIPELINE 1: filter only")
print("=" * 60)

# LOTUS
rewrite_mode = False
captured.clear()
t0 = time.time()
df_f = claims_df.copy().sem_filter(FILTER_PERSON)
lotus_time = time.time() - t0
lotus_cap = list(captured)
print(f"  LOTUS: {len(df_f)} passed ({lotus_time:.1f}s)")

# PZ
rewrite_mode = True
current_filter_instruction = FILTER_PERSON
current_filter_cols = ["claim"]
captured.clear()
t0 = time.time()
ds1 = pz.MemoryDataset(id="cmp-f1", vals=claims_df.to_dict("records"))
ds1 = ds1.sem_filter(
    "Does this claim mention a specific person? {claim}",
    depends_on=["claim"],
)
pz_df = ds1.run(config=pz_config).to_df()
pz_time = time.time() - t0
pz_cap = list(captured)
rewrite_mode = False
print(f"  PZ:    {len(pz_df)} passed ({pz_time:.1f}s)")

rows = []
for i in range(len(claims_df)):
    row = claims_df.iloc[i]
    lm = find_match(lotus_cap, row["claim"])
    pm = find_match(pz_cap, row["claim"])
    rows.append({
        "tuple": i, "claim": row["claim"][:80],
        "lotus_input": lm["input"], "lotus_output": lm["output"],
        "pz_input": pm["input"], "pz_output": pm["output"],
    })
write_csv("logs/no_vdb_filter.csv", rows)


# ============================================================
# Pipeline 2: map only (verdict: TRUE/FALSE)
# ============================================================
print("\n" + "=" * 60)
print("  PIPELINE 2: map only (verdict)")
print("=" * 60)

# LOTUS
rewrite_mode = False
captured.clear()
t0 = time.time()
df_m = claims_df.copy().sem_map(MAP_VERDICT, suffix="verdict")
lotus_time = time.time() - t0
lotus_cap = list(captured)
print(f"  LOTUS: {len(df_m)} rows ({lotus_time:.1f}s)")

# PZ
rewrite_mode = True
captured.clear()
t0 = time.time()
pz_m_df = pz_map_with_fallback(
    MAP_VERDICT, claims_df, "verdict",
    "TRUE if the claim is factually correct, FALSE otherwise.",
)
pz_time = time.time() - t0
pz_cap = list(captured)
rewrite_mode = False
print(f"  PZ:    {len(pz_m_df)} rows ({pz_time:.1f}s)")

rows = []
for i in range(len(claims_df)):
    row = claims_df.iloc[i]
    lm = find_match(lotus_cap, row["claim"])
    pm = find_match(pz_cap, row["claim"])
    rows.append({
        "tuple": i, "claim": row["claim"][:80],
        "lotus_input": lm["input"], "lotus_output": lm["output"],
        "pz_input": pm["input"], "pz_output": pm["output"],
    })
write_csv("logs/no_vdb_map.csv", rows)


# ============================================================
# Pipeline 3: filter → map (filter persons, then verdict)
# ============================================================
print("\n" + "=" * 60)
print("  PIPELINE 3: filter → map")
print("=" * 60)

# LOTUS: filter claims about persons, then get verdict
rewrite_mode = False
captured.clear()
t0 = time.time()
df_fm_f = claims_df.copy().sem_filter(FILTER_PERSON)
lotus_f_cap = list(captured)
captured.clear()
df_fm_m = df_fm_f.sem_map(MAP_VERDICT, suffix="verdict")
lotus_m_cap = list(captured)
lotus_time = time.time() - t0
print(f"  LOTUS: filter={len(df_fm_f)} passed, map={len(df_fm_m)} rows ({lotus_time:.1f}s)")

# PZ: filter then map
rewrite_mode = True
current_filter_instruction = FILTER_PERSON
current_filter_cols = ["claim"]
captured.clear()
t0 = time.time()
ds3 = pz.MemoryDataset(id="cmp-fm", vals=claims_df.to_dict("records"))
ds3 = ds3.sem_filter(
    "Does this claim mention a specific person? {claim}",
    depends_on=["claim"],
)
pz_fm_f_df = ds3.run(config=pz_config).to_df()
pz_f_cap = list(captured)

current_filter_instruction = None  # next op is map, not filter
captured.clear()
if len(pz_fm_f_df) > 0:
    pz_fm_m_df = pz_map_with_fallback(
        MAP_VERDICT, pz_fm_f_df, "verdict",
        "TRUE if the claim is factually correct, FALSE otherwise.",
    )
else:
    pz_fm_m_df = pd.DataFrame()
pz_m_cap = list(captured)
pz_time = time.time() - t0
rewrite_mode = False
print(f"  PZ:    filter={len(pz_fm_f_df)} passed, map={len(pz_fm_m_df)} rows ({pz_time:.1f}s)")

rows = []
for i in range(len(claims_df)):
    row = claims_df.iloc[i]
    lf = find_match(lotus_f_cap, row["claim"])
    pf = find_match(pz_f_cap, row["claim"])
    lm = find_match(lotus_m_cap, row["claim"])
    pm = find_match(pz_m_cap, row["claim"])
    rows.append({
        "tuple": i, "claim": row["claim"][:80],
        "lotus_filter_input": lf["input"], "lotus_filter_output": lf["output"],
        "pz_filter_input": pf["input"], "pz_filter_output": pf["output"],
        "lotus_map_input": lm["input"], "lotus_map_output": lm["output"],
        "pz_map_input": pm["input"], "pz_map_output": pm["output"],
    })
write_csv("logs/no_vdb_filter_map.csv", rows)


