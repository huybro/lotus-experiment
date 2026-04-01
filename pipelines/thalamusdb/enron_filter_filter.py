import json
import os
import re
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../.."
THALAMUSDB_SRC_ROOT = PROJECT_ROOT + "/projects/thalamusdb/src"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, THALAMUSDB_SRC_ROOT)

from transformers import AutoTokenizer

from data_utils import load_enron, write_csv
from pipelines import llm_intercepter, scenarios


project = "thalamusdb"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_TOKENS = 512
VLLM_API_BASE = "http://localhost:8003/v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def _strip_placeholders(text: str) -> str:
    return re.sub(r"\{[^}]+\}", "", text).strip()


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def _ensure_model_config(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    config = {
        "models": [
            {
                "modalities": ["text"],
                "priority": 100,
                "kwargs": {
                    "filter": {
                        "model": f"hosted_vllm/{MODEL_NAME}",
                        "api_base": VLLM_API_BASE,
                        "max_tokens": MAX_TOKENS,
                        "temperature": 0,
                    },
                    "join": {
                        "model": f"hosted_vllm/{MODEL_NAME}",
                        "api_base": VLLM_API_BASE,
                        "max_tokens": MAX_TOKENS,
                        "temperature": 0,
                    },
                },
            }
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f)


def main() -> None:
    joined_df = load_enron(os.path.join(PROJECT_ROOT, "projects/palimpzest/testdata/enron-eval"))

    log = []
    params = {"log": log, "max_tokens": MAX_TOKENS, "tokenizer": tokenizer}
    llm_intercepter.set_intercept(**params)

    # Import TDB modules after interceptor setup so `from litellm import completion`
    # in TDB operators resolves to the wrapped function.
    from tdb.data.relational import Database
    from tdb.execution.constraints import Constraints
    from tdb.execution.engine import ExecutionEngine
    from tdb.queries.query import Query

    db = Database(":memory:")
    db.con.register("emails_df", joined_df[["contents"]])
    db.execute2list("CREATE TABLE emails AS SELECT contents FROM emails_df")

    model_config_path = os.path.join(PROJECT_ROOT, "pipelines", "thalamusdb", "models.local.json")
    _ensure_model_config(model_config_path)

    engine = ExecutionEngine(db, dop=1, model_config_path=model_config_path)
    constraints = Constraints(max_error=0)

    filter_1 = _strip_placeholders(scenarios.FILTER_ENRON_FRAUD)
    filter_2 = _strip_placeholders(scenarios.FILTER_ENRON_NOT_NEWS)

    t0 = time.time()
    start_1 = len(log)
    query_1 = Query(
        db,
        f"SELECT contents FROM emails WHERE NLfilter(contents, '{_sql_quote(filter_1)}');",
    )
    df_filter1, _ = engine.run(query_1, constraints)
    end_1 = len(log)

    db.con.register("emails_f1_df", df_filter1[["contents"]])
    db.execute2list("CREATE OR REPLACE TABLE emails_f1 AS SELECT contents FROM emails_f1_df")

    start_2 = len(log)
    query_2 = Query(
        db,
        f"SELECT contents FROM emails_f1 WHERE NLfilter(contents, '{_sql_quote(filter_2)}');",
    )
    df_filter2, _ = engine.run(query_2, constraints)
    end_2 = len(log)
    elapsed = time.time() - t0

    print(
        f"  THALAMUSDB: {len(joined_df)}->{len(df_filter1)}->{len(df_filter2)} ({elapsed:.1f}s)"
    )

    rows = []
    for item in log[start_1:end_1]:
        rows.append(
            {
                "op": "filter_1",
                "thalamusdb_input": item["input"],
                "thalamusdb_output": item["output"],
            }
        )
    for item in log[start_2:end_2]:
        rows.append(
            {
                "op": "filter_2",
                "thalamusdb_input": item["input"],
                "thalamusdb_output": item["output"],
            }
        )

    out_path = f"logs/{project}_enron_filter_filter.csv"
    write_csv(out_path, rows)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
