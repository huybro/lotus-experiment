import os
import sys
import time

from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../.."
DOCETL_ROOT = PROJECT_ROOT + "/projects/docetl"

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DOCETL_ROOT)

from data_utils import load_fever, write_csv
from pipelines import llm_intercepter, scenarios

project = "docetl"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_TOKENS = 512
VLLM_API_BASE = "http://localhost:8003/v1"


def _strip_placeholders(text: str) -> str:
    return text.replace("{claim}{content}", "").strip()


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    os.environ.setdefault("DOCETL_HOME_DIR", os.path.join(PROJECT_ROOT, ".docetl_home"))

    df = load_fever(os.path.join(PROJECT_ROOT, "data", "fever_claims_with_evidence.csv"))
    df = df.head(5)
    records = df.to_dict("records")
    input_len = len(records)

    log = []
    llm_intercepter.set_intercept(log=log, max_tokens=MAX_TOKENS, tokenizer=tokenizer)
    from docetl.runner import DSLRunner 

    model = f"hosted_vllm/{MODEL_NAME}"
    filter_instruction = "{{ '' }}\n" + _strip_placeholders(scenarios.FEVER_FILTER)

    config = {
        "default_model": model,
        "default_lm_api_base": VLLM_API_BASE,
        "bypass_cache": True,
        "custom_prompt": True,
        "semantic_context_columns": ["claim", "content"],
        "datasets": {"fever": {"type": "memory", "path": records}},
        "operations": [
            {
                "name": "fever_filter",
                "type": "filter",
                "prompt": filter_instruction,
                "output": {"schema": {"keep": "boolean"}},
                "model": model,
            }
        ],
        "pipeline": {
            "steps": [{"name": "step1", "input": "fever", "operations": ["fever_filter"]}],
            "output": {"type": "file", "path": f"logs/{project}_fever_filter.json"},
        },
    }

    t0 = time.time()
    runner = DSLRunner(config, max_threads=1)
    runner.load()
    output, _, _ = runner.last_op_container.next()
    dt = time.time() - t0
    print(f"  Docetl: {len(output)}/{input_len} passed ({dt:.1f}s)")

    rows = [{"docetl_input": x["input"], "docetl_output": x["output"]} for x in log]
    write_csv(f"logs/{project}_fever_filter.csv", rows)
    print(f"  Saved logs/{project}_fever_filter.csv")


if __name__ == "__main__":
    main()
