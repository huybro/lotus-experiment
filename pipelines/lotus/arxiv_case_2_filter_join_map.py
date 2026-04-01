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
from data_utils import write_csv, load_arxiv

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


# Load Fever data
df = load_arxiv("/home/hojaeson_umass_edu/.cache/kagglehub/datasets/spsayakpaul/arxiv-paper-abstracts/versions/2/arxiv_txt_500")
df_robotic = load_arxiv("/home/hojaeson_umass_edu/.cache/kagglehub/datasets/spsayakpaul/arxiv-paper-abstracts/versions/2/arxiv_txt_robotic", column="robotic_abstract")
# df = df.iloc[:100]
log = []
params = {'log': log, 'max_tokens': MAX_TOKENS, 'tokenizer': tokenizer}
llm_intercepter.set_intercept(**params)

t0 = time.time()
input_len = len(df)
df = df.sem_filter(scenarios.CASE_2_FILTER_ARXIV) 
df = df.sem_join(df_robotic, scenarios.CASE_2_JOIN_ARXIV)
df = df.sem_map(scenarios.CASE_2_MAP_ARXIV)
print(f"  LOTUS: {len(df)}/{input_len} passed ({time.time() - t0:.1f}s)")

rows = []
for i in range(len(log)):
    rows.append({
        "lotus_input": log[i]["input"], "lotus_output": log[i]["output"],
    })

write_csv(f"logs/{project}_topk_map_arxiv.csv", rows)
print(f"  Saved logs/{project}_topk_map_arxiv.csv")

