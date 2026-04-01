import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../.."
LOTUS_ROOT = PROJECT_ROOT + "/projects/palimpzest/src"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, LOTUS_ROOT)


import time
import palimpzest as pz
from palimpzest.constants import Model

from pipelines import scenarios

from transformers import AutoTokenizer
from pipelines import llm_intercepter
from data_utils import write_csv, load_arxiv
from palimpzest.query.processor.config import QueryProcessorConfig

project = 'palimpzest'
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_TOKENS = 512
VLLM_API_BASE = "http://localhost:8003/v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

PZ_MODEL = Model(f"hosted_vllm/{MODEL_NAME}")
pz_config = QueryProcessorConfig(
    api_base=VLLM_API_BASE,
    available_models=[PZ_MODEL],
    allow_model_selection=False,
    allow_bonded_query=True,  # Use direct LLM (LLMConvertBonded), not RAG
    allow_rag_reduction=False,  # Disable RAG (needs OpenAI embeddings)
    allow_mixtures=False,
    allow_critic=False,
    allow_split_merge=False,
    verbose=False,
)

# Load Fever data
df = load_arxiv("/home/hojaeson_umass_edu/.cache/kagglehub/datasets/spsayakpaul/arxiv-paper-abstracts/versions/2/arxiv_txt_500")
log = []
params = {'log': log, 'max_tokens': MAX_TOKENS, 'tokenizer': tokenizer}
llm_intercepter.set_intercept(**params)

t0 = time.time()
ds = pz.MemoryDataset(id="cmp-f1", vals=df.to_dict("records"))

_ds = ds.sem_filter(
    scenarios.CASE_3_FILTER_1.replace('{abstract}', ""),
    depends_on=["abstract"],
)
_ds = _ds.sem_filter(
    scenarios.CASE_3_FILTER_2.replace('{abstract}', ""),
    depends_on=["abstract"],
)
_ds = _ds.sem_filter(
    scenarios.CASE_3_FILTER_3.replace('{abstract}', ""),
    depends_on=["abstract"],
)
pz_df = _ds.run(config=pz_config).to_df()
pz_time = time.time() - t0
pz_cap = list(log)
print(f"  PZ:    {len(pz_df)}/{len(df)} passed ({pz_time:.1f}s)")

# ── Log ──
rows = []
for i in range(len(log)): 
    rows.append({ 
        "pz_input": log[i]["input"], "pz_output": log[i]["output"],
    })
write_csv(f"logs/{project}_arxiv_topk_map.csv", rows)
print(f"  Saved logs/{project}_arxiv_topk_map.csv")
