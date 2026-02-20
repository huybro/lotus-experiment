import os
import time

os.environ["OMP_NUM_THREADS"] = "1"  # Prevent FAISS segfault on Apple Silicon

import lotus
import pandas as pd
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store import FaissVS
from data_loader import load_fever_claims, load_oracle_wiki_kb
from lotus_logger import LotusLogger

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "fever"
N_CLAIMS = 20
K_RETRIEVAL = 3

# Initialize logger 
logger = LotusLogger(
    model_name=MODEL_NAME,
    dataset_name=DATASET_NAME,
    experiment_name="map_filter",  
    debug=True,
    debug_max_chars=500,
)
logger.install()

lm = LM(model=f"hosted_vllm/{MODEL_NAME}", api_base="http://localhost:8000/v1")
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = FaissVS()
lotus.settings.configure(lm=lm, rm=rm, vs=vs)

# ============================================================
# Load Data
# ============================================================
claims_df = load_fever_claims(n=N_CLAIMS)
wiki_df = load_oracle_wiki_kb(claims_split="labelled_dev", n_claims=N_CLAIMS)

# SUPPORTS -> True; REFUTES, NOT ENOUGH INFO -> False
claims_df["true_label"] = claims_df["label"].apply(lambda l: l == "SUPPORTS")

# ============================================================
# Pipeline: Map -> Search -> Filter
# ============================================================
pipeline_start = time.time()

print("\n[1/4] Indexing Wikipedia KB...")
wiki_df = wiki_df.sem_index("content", index_dir="./fever_index")

print("\n[2/4] Generating search queries (sem_map)...")
claims_df = claims_df.sem_map(
    "Given the claim: {claim}\n"
    "Write a short factual search query to find evidence about this claim. "
    "Output only the search query, nothing else.",
    suffix="search_query"
)

print("\n[3/4] Retrieving evidence (sem_sim_join)...")
search_results = claims_df.sem_sim_join(
    wiki_df,
    left_on="search_query",
    right_on="content",
    K=K_RETRIEVAL
)

print("\n[4/4] Verifying claims (sem_filter)...")
verified_results = search_results.sem_filter(
    "{content}\n"
    "Based on the above evidence, the following claim is supported: {claim}"
)

pipeline_elapsed = time.time() - pipeline_start

# ============================================================
# Results
# ============================================================
passed_ids = set(verified_results["id"].tolist())
claims_df["predicted_label"] = claims_df["id"].apply(lambda x: x in passed_ids)

correct = (claims_df["predicted_label"] == claims_df["true_label"]).sum()
accuracy = correct / len(claims_df)

print(f"\n{'='*60}")
print(f"  Accuracy:   {accuracy:.1%}  ({correct}/{len(claims_df)})")
print(f"  Total Time: {pipeline_elapsed:.1f}s")
print(f"{'='*60}")

for _, row in claims_df.iterrows():
    match = "✓" if row["predicted_label"] == row["true_label"] else "✗"
    print(f"  {match}  [{row['label']:>15}] pred={str(row['predicted_label']):>5}  {row['claim'][:70]}")

logger.summary()