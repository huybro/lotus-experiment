import sys
import os

# Add lotus source AFTER stdlib init
sys.path.append("/home/hojaeson_umass_edu/project/vllm-test/ref/lotus-experiment/projects/lotus")

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
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} (claim, evidence) pairs from {DATA_PATH}")



df['claim'] = "[Claim] " + df['claim']
df['content'] = "[Evidence] " + df['content']

FILTER_RELEVANCE = (
    "{claim} {content} The evidence can determine whether the claim is true or false\n"
)

MAP_REASONING = (
    "{claim} {content} Explain how the evidence supports or refutes the claim.\n"
)
t0 = time.perf_counter()
df = df.sem_filter(FILTER_RELEVANCE, strategy="cot").sem_map(MAP_REASONING, strategy="cot")
t1 = time.perf_counter()

print(f"sem_filter {len(df)} tuples, took {t1 - t0:.4f} seconds")
print(f"Throughput {(t1 - t0) / len(df)}")




FILTER_RELEVANCE = (
    "{claim} {content} The evidence can determine whether the claim is true or false\n"
    # "Claim: {claim} Evidence: {content}\n"
)

t0 = time.perf_counter()
joined_df = joined_df.sem_filter(FILTER_RELEVANCE, strategy="cot")
t1 = time.perf_counter()
