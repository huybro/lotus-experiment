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
from data_utils import write_csv, load_fever
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
df = load_fever(os.path.join(PROJECT_ROOT, "data", "fever_claims_with_evidence.csv"))
df = df.iloc[:5]
log = []
params = {'log': log, 'max_tokens': MAX_TOKENS, 'tokenizer': tokenizer}
llm_intercepter.set_intercept(**params)

t0 = time.time()
ds = pz.MemoryDataset(id="cmp-f1", vals=df.to_dict("records"))
ds = ds.sem_filter(
    scenarios.FEVER_FILTER,
    depends_on=["claim", "content"],
)
pz_df = ds.run(config=pz_config).to_df()
pz_time = time.time() - t0
pz_cap = list(log)
print(f"  PZ:    {len(pz_df)}/{len(df)} passed ({pz_time:.1f}s)")

# ── Log ──
rows = []
for i in range(len(log)): 
    rows.append({ 
        "pz_input": log[i]["input"], "pz_output": log[i]["output"],
    })
write_csv(f"logs/{project}_fever_filter.csv", rows)
print(f"  Saved logs/{project}_fever_filter.csv")
