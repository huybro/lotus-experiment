"""
Side-by-side comparison of LOTUS vs Palimpzest FEVER experiments.

Runs the same filter experiment on both systems with identical data,
prompts, and model. Outputs a CSV with columns:
  tuple | filter_lotus_input | filter_lotus_output | filter_pz_input | filter_pz_output

This lets you verify that prompts are identical and compare outputs.
"""
import os
import csv
import time

import lotus
import pandas as pd
from lotus.models import LM
from data_loader import load_fever_claims, load_oracle_wiki_kb
from retrieval import retrieve_for_claims

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "fever"
N_CLAIMS = 20
K_RETRIEVAL = 3
MAX_TOKENS = 512
VLLM_API_BASE = "http://localhost:8000/v1"
OUTPUT_CSV = "logs/comparison_filter.csv"

# ============================================================
# Load Data
# ============================================================
claims_df = load_fever_claims(n=N_CLAIMS)
wiki_df = load_oracle_wiki_kb(claims_split="labelled_dev", n_claims=N_CLAIMS)
claims_df["true_label"] = claims_df["label"].apply(lambda l: l == "SUPPORTS")

# Shared retrieval
print("\n[Shared] Retrieving evidence...")
joined_df = retrieve_for_claims(claims_df, wiki_df, query_col="claim", K=K_RETRIEVAL)
print(f"  Total tuples: {len(joined_df)}")

# ============================================================
# Part 1: Run LOTUS filter — capture prompts and outputs
# ============================================================
print("\n" + "=" * 60)
print("  Running LOTUS filter...")
print("=" * 60)

from universal_prompts import install_prompt_overrides, get_prompt
install_prompt_overrides()

lm = LM(model=f"hosted_vllm/{MODEL_NAME}", api_base=VLLM_API_BASE, max_tokens=MAX_TOKENS)
lotus.settings.configure(lm=lm)

# Capture LOTUS prompts by generating them manually
lotus_prompts = []
lotus_outputs = []

filter_instruction = (
    "{content}\n"
    "Based on the above evidence, the following claim is supported: {claim}"
)

# Generate prompts for each tuple (same logic as sem_filter uses internally)
for _, row in joined_df.iterrows():
    filled = filter_instruction.format(content=row["content"], claim=row["claim"])
    msgs = get_prompt(filled, filled, op='sem_filter')
    prompt_text = "\n".join(m["content"] for m in msgs)
    lotus_prompts.append(prompt_text)

# Run the actual LOTUS filter
df_lotus = joined_df.copy()
t0 = time.time()
df_lotus = df_lotus.sem_filter(filter_instruction)
lotus_time = time.time() - t0

# The filter removes rows that don't pass — we need to track which passed
# Re-run to capture individual outputs via the logger
from lotus_logger import LotusLogger
logger = LotusLogger(model_name=MODEL_NAME, dataset_name=DATASET_NAME,
                     experiment_name="comparison_lotus_filter", debug=False)
logger.install()

# Re-run filter to capture outputs
df_lotus2 = joined_df.copy()
df_lotus2 = df_lotus2.sem_filter(filter_instruction)
logger.uninstall()

# Read captured outputs from the logger CSV
lotus_log_path = logger.output_path
lotus_log_df = pd.read_csv(lotus_log_path) if os.path.exists(lotus_log_path) else pd.DataFrame()

print(f"  LOTUS filter: {len(df_lotus2)} passed out of {len(joined_df)} ({lotus_time:.1f}s)")

# ============================================================
# Part 2: Run PZ filter — capture prompts and outputs
# ============================================================
print("\n" + "=" * 60)
print("  Running PZ filter...")
print("=" * 60)

import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.query.processor.config import QueryProcessorConfig
from universal_prompts import install_pz_prompt_overrides

install_pz_prompt_overrides()

# Monkey-patch litellm to capture prompts and outputs
import litellm as _litellm
_original_completion = _litellm.completion

pz_captured = []  # list of (prompt_text, output_text)

def _capturing_completion(*args, **kwargs):
    kwargs.setdefault("max_tokens", MAX_TOKENS)
    messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
    prompt_text = "\n".join(
        m.get("content", "") for m in messages if isinstance(m, dict)
    )

    result = _original_completion(*args, **kwargs)

    output_text = result.choices[0].message.content if result.choices else ""
    pz_captured.append({"input": prompt_text, "output": output_text})
    return result

_litellm.completion = _capturing_completion

PZ_MODEL = Model("hosted_vllm/Qwen/Qwen2.5-1.5B-Instruct")
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

ds_pz = pz.MemoryDataset(
    id="fever-comparison-filter",
    vals=joined_df.to_dict("records"),
)
ds_pz = ds_pz.sem_filter(
    "Based on the evidence, the following claim is supported: the claim states that {claim}",
    depends_on=["content", "claim"],
)

t0 = time.time()
pz_output = ds_pz.run(config=pz_config)
pz_time = time.time() - t0
pz_df = pz_output.to_df()

print(f"  PZ filter: {len(pz_df)} passed out of {len(joined_df)} ({pz_time:.1f}s)")
print(f"  PZ captured {len(pz_captured)} LLM calls")

# ============================================================
# Part 3: Build side-by-side comparison CSV
# ============================================================
print("\n" + "=" * 60)
print("  Writing comparison CSV...")
print("=" * 60)

os.makedirs("logs", exist_ok=True)

# Build rows: one per tuple
rows = []
n_tuples = len(joined_df)

# LOTUS outputs from the logger CSV
lotus_inputs_from_log = lotus_log_df["full_input"].tolist() if "full_input" in lotus_log_df.columns else lotus_prompts
lotus_outputs_from_log = lotus_log_df["output"].tolist() if "output" in lotus_log_df.columns else [""] * n_tuples

for i in range(n_tuples):
    row = {
        "tuple": i,
        "claim": joined_df.iloc[i]["claim"][:80],
        "evidence": joined_df.iloc[i]["content"][:80],
        "filter_lotus_input": lotus_inputs_from_log[i] if i < len(lotus_inputs_from_log) else "",
        "filter_lotus_output": lotus_outputs_from_log[i] if i < len(lotus_outputs_from_log) else "",
        "filter_pz_input": pz_captured[i]["input"] if i < len(pz_captured) else "",
        "filter_pz_output": pz_captured[i]["output"] if i < len(pz_captured) else "",
    }
    rows.append(row)

# Write CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"  ✅ Wrote {len(rows)} rows to {OUTPUT_CSV}")
print(f"\n  Columns: tuple | claim | evidence | filter_lotus_input | filter_lotus_output | filter_pz_input | filter_pz_output")
print(f"\n  LOTUS time: {lotus_time:.1f}s | PZ time: {pz_time:.1f}s")
