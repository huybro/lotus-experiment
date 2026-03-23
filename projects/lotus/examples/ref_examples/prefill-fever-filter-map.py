import sys
import os

# Add lotus source AFTER stdlib init
sys.path.append("/home/hojaeson_umass_edu/project/vllm-test/ref/lotus-experiment/projects/lotus")

from lotus.models import LM
import lotus 
import pandas as pd
import time
import ttft
import requests


from transformers import AutoTokenizer
import vllm_utils

# ============================================================
# Configuration
# ============================================================


MAX_TOKENS = 1
lm = LM(
    model='hosted_vllm/meta-llama/Llama-3.2-3B-Instruct',
    api_base='http://localhost:8003/v1',
    max_ctx_len=8000,
    max_tokens=MAX_TOKENS
)

lotus.settings.configure(lm=lm)


URL = "http://localhost:8003/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# ============================================================
# Load Data
# ============================================================
DATA_PATH = "data/fever_claims_with_evidence_2000_1.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Run 'python prepare_data.py' first to generate it."
    )
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} (claim, evidence) pairs from {DATA_PATH}")


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
claim_target = max(len(tokenizer.encode(x)) for x in df["claim"])
content_target = max(len(tokenizer.encode(x)) for x in df["content"])

def pad_to_tokens(text, target):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    diff = target - len(tokens)
    if diff <= 0:
        return text
    return (text + ("pad " * diff))*2

df["claim"] = df["claim"].apply(lambda x: pad_to_tokens(x, claim_target))
df["content"] = df["content"].apply(lambda x: pad_to_tokens(x, content_target))

port = 8003
run_base_url = f"http://localhost:{port}"

# VLLM restart
vllm_proc = vllm_utils.start_vllm(8003, MODEL)
run_base_url = f"http://localhost:{port}"
if not vllm_utils.wait_for_vllm(run_base_url, max_wait=300):
    print("    [ERROR] vLLM failed to start within 300s")
    if vllm_proc:
        vllm_utils.stop_vllm(vllm_proc)
    sys.exit(1)
vllm_utils.warm_up_vllm(f"{run_base_url}/v1")

start = 0
for idx in [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250]:
    print(f" {idx}  tuples")


    # Varying tuple #
    FILTER_RELEVANCE = (
        "{claim} {content} The evidence can determine whether the claim is true or false\n"
    )

    MAP_REASONING = (
        "{claim} {content} Explain how the evidence supports or refutes the claim.\n"
    )

    end = start + idx
    _df = df.iloc[start:end].copy()
    baseline_sum, baseline_count = ttft.get_prefill_metrics()
    t0 = time.perf_counter()
    _df = _df.sem_map(MAP_REASONING, strategy="cot")
    t1 = time.perf_counter()
    print(f"sem_map took {t1 - t0:.4f} seconds")
    t2 = time.perf_counter()
    _df = _df.sem_filter(FILTER_RELEVANCE, strategy="cot")
    t3 = time.perf_counter()
    print(f"sem_filter took {t3 - t2:.4f} seconds")
    

    end_sum, end_count = ttft.get_prefill_metrics()

    delta_sum = end_sum - baseline_sum
    delta_count = end_count - baseline_count

    avg_prefill  = delta_sum / delta_count
    print(f"Throughput {(end-start) / (t3 - t0)}")
    print(f"{idx} Average Prefill time:", avg_prefill)
    vllm_utils.warm_up_vllm(f"{run_base_url}/v1")
    time.sleep(3)
    start = end