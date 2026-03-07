import sys
import os

# Add lotus source AFTER stdlib init
sys.path.append("/home/hojaeson_umass_edu/project/vllm-test/ref/lotus")

from lotus.models import LM
import lotus 
import pandas as pd
import time

lm = LM(
    model='hosted_vllm/meta-llama/Llama-3.2-3B-Instruct',
    api_base='http://localhost:8003/v1',
    max_ctx_len=8000,
    max_tokens=1000
)

lotus.settings.configure(lm=lm)


# ============================================================
# Load Data
# ============================================================
DATA_PATH = "data/fever_claims_with_evidence.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Run 'python prepare_data.py' first to generate it."
    )
joined_df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(joined_df)} (claim, evidence) pairs from {DATA_PATH}")


FILTER_RELEVANCE = (
    "{claim} The evidence can determine whether the claim is true or false\n"
    # "Claim: {claim} Evidence: {content}\n"
)

t0 = time.perf_counter()
joined_df = joined_df.sem_filter(FILTER_RELEVANCE, strategy="cot").sem_filter(FILTER_RELEVANCE, strategy="cot")
t1 = time.perf_counter()

print(f"sem_filter {len(joined_df)} tuples, took {t1 - t0:.4f} seconds")
print(f"Throughput {(t1 - t0) / len(joined_df)}")


