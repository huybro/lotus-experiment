import os
import time

os.environ["OMP_NUM_THREADS"] = "1"  # Prevent FAISS segfault on Apple Silicon

import lotus
import pandas as pd
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store import FaissVS
from data_loader import load_fever_claims, load_oracle_wiki_kb
from retrieval import retrieve_for_claims
from lotus_logger import LotusLogger
from universal_prompts import install_prompt_overrides

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "fever"
N_CLAIMS = 20
K_RETRIEVAL = 3

# Install universal prompt overrides (filter + map)
install_prompt_overrides()

# LM setup (no RM/VS needed — retrieval is done externally)
lm = LM(model=f"hosted_vllm/{MODEL_NAME}", api_base="http://localhost:8000/v1")
lotus.settings.configure(lm=lm)

# ============================================================
# Load Data (shared across all experiments)
# ============================================================
claims_df = load_fever_claims(n=N_CLAIMS)
wiki_df = load_oracle_wiki_kb(claims_split="labelled_dev", n_claims=N_CLAIMS)
claims_df["true_label"] = claims_df["label"].apply(lambda l: l == "SUPPORTS")

# Pre-compute retrieval using raw claims as queries (shared across experiments)
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
print("  EXPERIMENT 1: map → filter")
print("=" * 60)

logger1 = LotusLogger(model_name=MODEL_NAME, dataset_name=DATASET_NAME,
                       experiment_name="lotus_map_filter", debug=True, debug_max_chars=300)
logger1.install()

t0 = time.time()

# MAP: generate search queries from claims
df1 = claims_df.copy()
df1 = df1.sem_map(
    "Given the claim: {claim}\n"
    "Write a short factual search query to find evidence about this claim. "
    "Output only the search query, nothing else.",
    suffix="search_query"
)

# Retrieve evidence using MAP-generated search queries
df1_retrieved = retrieve_for_claims(df1, wiki_df, query_col="search_query", K=K_RETRIEVAL)

# FILTER: verify claim against evidence
df1_verified = df1_retrieved.sem_filter(
    "{content}\n"
    "Based on the above evidence, the following claim is supported: {claim}"
)

elapsed1 = time.time() - t0
passed1 = set(df1_verified["id"].tolist())
evaluate(claims_df, passed1, "lotus_map_filter", elapsed1)
logger1.summary()
logger1.uninstall()


# ============================================================
# Experiment 2: filter → filter
# ============================================================
print("\n\n" + "=" * 60)
print("  EXPERIMENT 2: filter → filter")
print("=" * 60)

logger2 = LotusLogger(model_name=MODEL_NAME, dataset_name=DATASET_NAME,
                       experiment_name="lotus_filter_filter", debug=True, debug_max_chars=300)
logger2.install()

t0 = time.time()

# Use pre-computed retrieval (joined_df)
df2 = joined_df.copy()

# FILTER 1: is the evidence relevant to the claim?
df2 = df2.sem_filter(
    "The following evidence is relevant to the claim.\n"
    "Evidence: {content}\nClaim: {claim}"
)

# FILTER 2: does the evidence support the claim?
df2_verified = df2.sem_filter(
    "{content}\n"
    "Based on the above evidence, the following claim is supported: {claim}"
)

elapsed2 = time.time() - t0
passed2 = set(df2_verified["id"].tolist())
evaluate(claims_df, passed2, "lotus_filter_filter", elapsed2)
logger2.summary()
logger2.uninstall()


# ============================================================
# Experiment 3: map only
# ============================================================
print("\n\n" + "=" * 60)
print("  EXPERIMENT 3: map only")
print("=" * 60)

logger3 = LotusLogger(model_name=MODEL_NAME, dataset_name=DATASET_NAME,
                       experiment_name="lotus_map", debug=True, debug_max_chars=300)
logger3.install()

t0 = time.time()

# MAP: directly ask the LLM to classify the claim as TRUE/FALSE
df3 = claims_df.copy()
df3 = df3.sem_map(
    "Given the claim: {claim}\n"
    "Is this claim true or false based on your knowledge? "
    "Answer with exactly TRUE or FALSE, nothing else.",
    suffix="verdict"
)

elapsed3 = time.time() - t0
# Parse the map output to get predictions
df3["predicted_label"] = df3["verdict"].str.strip().str.upper().str.contains("TRUE")
correct3 = (df3["predicted_label"] == df3["true_label"]).sum()
accuracy3 = correct3 / len(df3)
print(f"\n{'='*60}")
print(f"  Experiment: lotus_map")
print(f"  Accuracy:   {accuracy3:.1%}  ({correct3}/{len(df3)})")
print(f"  Total Time: {time.time() - t0:.1f}s")
print(f"{'='*60}")
logger3.summary()
logger3.uninstall()


# ============================================================
# Experiment 4: filter only
# ============================================================
print("\n\n" + "=" * 60)
print("  EXPERIMENT 4: filter only")
print("=" * 60)

logger4 = LotusLogger(model_name=MODEL_NAME, dataset_name=DATASET_NAME,
                       experiment_name="lotus_filter", debug=True, debug_max_chars=300)
logger4.install()

t0 = time.time()

# Use pre-computed retrieval (joined_df)
df4 = joined_df.copy()

# FILTER: single-step verification
df4_verified = df4.sem_filter(
    "{content}\n"
    "Based on the above evidence, the following claim is supported: {claim}"
)

elapsed4 = time.time() - t0
passed4 = set(df4_verified["id"].tolist())
evaluate(claims_df, passed4, "lotus_filter", elapsed4)
logger4.summary()
logger4.uninstall()


# ============================================================
# Final Summary
# ============================================================
print("\n\n" + "=" * 60)
print("  ALL LOTUS EXPERIMENTS COMPLETE")
print("=" * 60)