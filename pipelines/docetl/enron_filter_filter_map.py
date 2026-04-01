import os
import sys
import time

from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../.."
DOCETL_ROOT = PROJECT_ROOT + "/projects/docetl"

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DOCETL_ROOT)

from data_utils import load_enron, write_csv
from pipelines import llm_intercepter, scenarios

project = "docetl"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_TOKENS = 512
VLLM_API_BASE = "http://localhost:8003/v1"

def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    os.environ.setdefault("DOCETL_HOME_DIR", os.path.join(PROJECT_ROOT, ".docetl_home"))

    df = load_enron(os.path.join(PROJECT_ROOT, "projects/palimpzest/testdata/enron-eval"), test=False)
    df = df.head(50)
    records = df.to_dict("records")
    records_len = len(records)

    log = []
    llm_intercepter.set_intercept(log=log, max_tokens=MAX_TOKENS, tokenizer=tokenizer)
    from docetl.runner import DSLRunner  # noqa: E402

    model = f"hosted_vllm/{MODEL_NAME}"
    config = {
        "default_model": model,
        "default_lm_api_base": VLLM_API_BASE,
        "bypass_cache": True,
        "custom_prompt": True,
        "semantic_context_columns": ["filename", "contents"],
        "datasets": {"enron": {"type": "memory", "path": records}},
        "operations": [
            {
                "name": "filter_1",
                "type": "filter",
                "prompt": "{{ '' }}\n" + scenarios.FILTER_ENRON_FRAUD.strip(),
                "output": {"schema": {"keep": "boolean"}},
                "model": model,
            },
            {
                "name": "filter_2",
                "type": "filter",
                "prompt": "{{ '' }}\n" + scenarios.FILTER_ENRON_NOT_NEWS.strip(),
                "output": {"schema": {"keep": "boolean"}},
                "model": model,
            },
            {
                "name": "map",
                "type": "map",
                "prompt": "{{ '' }}\n" + scenarios.MAP_ENRON_EXPLANATION.strip(),
                "output": {"schema": {"fraud_explanation": "string"}},
                "model": model,
            },
        ],
        "pipeline": {
            "steps": [
                {
                    "name": "step1",
                    "input": "enron",
                    "operations": ["filter_1", "filter_2", "map"],
                }
            ],
            "output": {
                "type": "file",
                "path": f"logs/{project}_enron_filter_filter_map.json",
            },
        },
    }

    t0 = time.time()
    runner = DSLRunner(config, max_threads=1)
    runner.load()
    output, _, _ = runner.last_op_container.next()

    log_f1 = log[:records_len]
    remaining = log[records_len:]
    # filter_2 and map both run on rows that passed filter_1
    half = len(remaining) // 2
    log_f2 = remaining[:half]
    log_m = remaining[half:]
    dt = time.time() - t0
    print(f"  Docetl: {records_len}->{len(log_f2)}->{len(output)} ({dt:.1f}s)")

    rows = []
    rows.extend(
        {"op": "filter_1", "docetl_input": x["input"], "docetl_output": x["output"]}
        for x in log_f1
    )
    rows.extend(
        {"op": "filter_2", "docetl_input": x["input"], "docetl_output": x["output"]}
        for x in log_f2
    )
    rows.extend(
        {"op": "map", "docetl_input": x["input"], "docetl_output": x["output"]}
        for x in log_m
    )
    write_csv(f"logs/{project}_enron_filter_filter_map.csv", rows)
    print(f"  Saved logs/{project}_enron_filter_filter_map.csv")


if __name__ == "__main__":
    main()
