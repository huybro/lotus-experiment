"""
Palimpzest FEVER experiment — runs the same 4 pipelines as run_lotus_fever.py
but using Palimpzest operators, with identical prompts and the same vLLM model.

Experiments:
  1. map → filter
  2. filter → filter
  3. map only
  4. filter only
"""
import os
import time

os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.query.processor.config import QueryProcessorConfig

from data_loader import load_fever_claims, load_oracle_wiki_kb
from retrieval import retrieve_for_claims
from universal_prompts import install_pz_prompt_overrides

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "fever"
N_CLAIMS = 20
K_RETRIEVAL = 3
VLLM_API_BASE = "http://localhost:8000/v1"

# Install universal prompt overrides for PZ (registers model + patches prompts)
install_pz_prompt_overrides()

# PZ execution config — force single vLLM model, no optimization
PZ_MODEL = Model("hosted_vllm/Qwen/Qwen2.5-1.5B-Instruct")
pz_config = QueryProcessorConfig(
    api_base=VLLM_API_BASE,
    available_models=[PZ_MODEL],
    allow_model_selection=False,
    allow_bonded_query=False,
    allow_mixtures=False,
    allow_critic=False,
    allow_split_merge=False,
    verbose=True,
)

# ============================================================
# Load Data
# ============================================================
claims_df = load_fever_claims(n=N_CLAIMS)
wiki_df = load_oracle_wiki_kb(claims_split="labelled_dev", n_claims=N_CLAIMS)
claims_df["true_label"] = claims_df["label"].apply(lambda l: l == "SUPPORTS")

# Pre-compute retrieval (same evidence as LOTUS experiments)
print("\n[Shared] Retrieving evidence with sentence-transformers + FAISS...")
joined_df = retrieve_for_claims(claims_df, wiki_df, query_col="claim", K=K_RETRIEVAL)


def evaluate(claims_df, passed_ids, experiment_name, elapsed):
    """Compute and print accuracy."""
    claims_df = claims_df.copy()
    claims_df["predicted_label"] = claims_df["id"].apply(lambda x: x in passed_ids)
    correct = (claims_df["predicted_label"] == claims_df["true_label"]).sum()
    accuracy = correct / len(claims_df)
    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Accuracy:   {accuracy:.1%}  ({correct}/{len(claims_df)})")
    print(f"  Total Time: {elapsed:.1f}s")
    print(f"{'='*60}")
    for _, row in claims_df.iterrows():
        match = "✓" if row["predicted_label"] == row["true_label"] else "✗"
        print(f"  {match}  [{row['label']:>15}] pred={str(row['predicted_label']):>5}  {row['claim'][:70]}")
    return accuracy


# ============================================================
# Experiment 1: map → filter
# ============================================================
print("\n\n" + "=" * 60)
print("  PZ EXPERIMENT 1: map → filter")
print("=" * 60)

t0 = time.time()

# MAP: generate search queries from claims
ds1 = pz.MemoryDataset(
    id="fever-claims-1",
    vals=claims_df[["id", "claim", "label", "true_label"]].to_dict("records"),
)
ds1 = ds1.sem_map(
    cols=[{"name": "search_query", "type": str, "desc": "A short factual search query to find evidence about the claim"}],
    depends_on=["claim"],
)
map_output = ds1.run(config=pz_config)
map_df = map_output.to_df()

# Retrieve evidence using MAP-generated search queries
map_claims = claims_df.copy()
map_claims["search_query"] = map_df["search_query"].tolist()[:len(map_claims)]
map_retrieved = retrieve_for_claims(map_claims, wiki_df, query_col="search_query", K=K_RETRIEVAL)

# FILTER: verify claim against evidence
ds1_filter = pz.MemoryDataset(
    id="fever-map-filter-1",
    vals=map_retrieved.to_dict("records"),
)
ds1_filter = ds1_filter.sem_filter(
    "Based on the evidence, the following claim is supported: the claim states that {claim}",
    depends_on=["content", "claim"],
)
filter_output = ds1_filter.run(config=pz_config)
filter_df = filter_output.to_df()

elapsed1 = time.time() - t0
passed1 = set(filter_df["id"].tolist()) if len(filter_df) > 0 else set()
evaluate(claims_df, passed1, "pz_map_filter", elapsed1)


# ============================================================
# Experiment 2: filter → filter
# ============================================================
print("\n\n" + "=" * 60)
print("  PZ EXPERIMENT 2: filter → filter")
print("=" * 60)

t0 = time.time()

# FILTER 1: is the evidence relevant to the claim?
ds2 = pz.MemoryDataset(
    id="fever-filter-filter-2",
    vals=joined_df.to_dict("records"),
)
ds2 = ds2.sem_filter(
    "The following evidence is relevant to the claim",
    depends_on=["content", "claim"],
)
f1_output = ds2.run(config=pz_config)
f1_df = f1_output.to_df()

# FILTER 2: does the evidence support the claim?
if len(f1_df) > 0:
    ds2b = pz.MemoryDataset(
        id="fever-filter-filter-2b",
        vals=f1_df.to_dict("records"),
    )
    ds2b = ds2b.sem_filter(
        "Based on the evidence, the following claim is supported: the claim states that {claim}",
        depends_on=["content", "claim"],
    )
    f2_output = ds2b.run(config=pz_config)
    f2_df = f2_output.to_df()
else:
    f2_df = pd.DataFrame()

elapsed2 = time.time() - t0
passed2 = set(f2_df["id"].tolist()) if len(f2_df) > 0 else set()
evaluate(claims_df, passed2, "pz_filter_filter", elapsed2)


# ============================================================
# Experiment 3: map only
# ============================================================
print("\n\n" + "=" * 60)
print("  PZ EXPERIMENT 3: map only")
print("=" * 60)

t0 = time.time()

ds3 = pz.MemoryDataset(
    id="fever-map-3",
    vals=claims_df[["id", "claim", "label", "true_label"]].to_dict("records"),
)
ds3 = ds3.sem_map(
    cols=[{"name": "verdict", "type": str, "desc": "TRUE if the claim is factually correct, FALSE otherwise. Answer with exactly TRUE or FALSE."}],
    depends_on=["claim"],
)
map3_output = ds3.run(config=pz_config)
map3_df = map3_output.to_df()

elapsed3 = time.time() - t0
map3_df["predicted_label"] = map3_df["verdict"].str.strip().str.upper().str.contains("TRUE")
correct3 = (map3_df["predicted_label"] == map3_df["true_label"]).sum()
accuracy3 = correct3 / len(map3_df)
print(f"\n{'='*60}")
print(f"  Experiment: pz_map")
print(f"  Accuracy:   {accuracy3:.1%}  ({correct3}/{len(map3_df)})")
print(f"  Total Time: {elapsed3:.1f}s")
print(f"{'='*60}")


# ============================================================
# Experiment 4: filter only
# ============================================================
print("\n\n" + "=" * 60)
print("  PZ EXPERIMENT 4: filter only")
print("=" * 60)

t0 = time.time()

ds4 = pz.MemoryDataset(
    id="fever-filter-4",
    vals=joined_df.to_dict("records"),
)
ds4 = ds4.sem_filter(
    "Based on the evidence, the following claim is supported: the claim states that {claim}",
    depends_on=["content", "claim"],
)
f4_output = ds4.run(config=pz_config)
f4_df = f4_output.to_df()

elapsed4 = time.time() - t0
passed4 = set(f4_df["id"].tolist()) if len(f4_df) > 0 else set()
evaluate(claims_df, passed4, "pz_filter", elapsed4)


# ============================================================
# Final Summary
# ============================================================
print("\n\n" + "=" * 60)
print("  ALL PALIMPZEST EXPERIMENTS COMPLETE")
print("=" * 60)
