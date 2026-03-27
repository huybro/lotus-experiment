import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../.."
LOTUS_ROOT = PROJECT_ROOT + "/projects/lotus"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, LOTUS_ROOT)

import time

from pipelines import scenarios
import lotus
from lotus.models import LM
from transformers import AutoTokenizer
from pipelines import llm_intercepter
from data_utils import write_csv, load_enron


project = 'lotus'
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_TOKENS = 512
VLLM_API_BASE = "http://localhost:8003/v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


_lotus_lm = LM(
    model=f"hosted_vllm/{MODEL_NAME}",
    api_base=VLLM_API_BASE,
    max_tokens=MAX_TOKENS,
    temperature=0,
)
lotus.settings.configure(lm=_lotus_lm)


# -- LOTUS --
joined_df = load_enron(os.path.join(PROJECT_ROOT, "projects/palimpzest/testdata/enron-eval"), test=True)
log = []
params = {'log': log, 'max_tokens': MAX_TOKENS, 'tokenizer': tokenizer}
llm_intercepter.set_intercept(**params)


t0 = time.time()
df_filter1 = joined_df.sem_filter(scenarios.FILTER_ENRON_FRAUD_2)
df_filter2 = df_filter1.sem_filter(scenarios.FILTER_ENRON_NOT_NEWS_2)
df_map = df_filter2.sem_map(scenarios.MAP_ENRON_EXPLANATION_2, suffix="fraud_explanation")
lotus_time = time.time() - t0
print(f"  LOTUS: {len(joined_df)}->{len(df_filter1)}->{len(df_filter2)} ({lotus_time:.1f}s)")

# Slice logger into stages (one LM call per row per op)
f1_len = len(df_filter1)
f2_len = len(df_filter2)
m_len = len(df_map)

lotus_f1_cap = log[0:len(joined_df)]
lotus_f2_cap = log[len(joined_df):len(joined_df) + f1_len]
lotus_m_cap = log[len(joined_df) + f1_len:len(joined_df) + f1_len + m_len]

# -- Log (wide format: one row per original email, stage outputs as columns) --
rows = []

for i in range(len(lotus_f1_cap)):
    lm = lotus_f1_cap[i]
    rows.append({"op": 'filter_1', "lotus_input": lm["input"], "lotus_output": lm["output"]})

for i in range(len(lotus_f2_cap)):
    lm = lotus_f2_cap[i]
    rows.append({"op": 'filter_2', "lotus_input": lm["input"], "lotus_output": lm["output"]})


for i in range(len(df_map)):
    lm = lotus_f2_cap[i]
    rows.append({"op": 'map', "lotus_input": lm["input"], "lotus_output": lm["output"]})

write_csv(f"logs/{project}_enron_filter_filter_map.csv", rows)
print(f"  Saved logs/{project}_enron_filter_filter_map_2.csv")
