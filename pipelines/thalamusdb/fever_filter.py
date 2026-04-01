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

from data_utils import load_fever, write_csv
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
    df = load_fever(os.path.join(PROJECT_ROOT, "data", "fever_claims_with_evidence.csv"))
    df = df.iloc[:5].copy()
    df["data"] = df["claim"].astype(str) + "\n" + df["content"].astype(str)

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
    db.con.register("fever_df", df[["data"]])
    db.execute2list("CREATE TABLE fever AS SELECT data FROM fever_df")

    model_config_path = os.path.join(PROJECT_ROOT, "pipelines", "thalamusdb", "models.local.json")
    _ensure_model_config(model_config_path)

    condition = _strip_placeholders(scenarios.FEVER_FILTER)
    sql = f"SELECT data FROM fever WHERE NLfilter(data, '{_sql_quote(condition)}');"

    query = Query(db, sql)
    engine = ExecutionEngine(db, dop=1, model_config_path=model_config_path)
    constraints = Constraints(max_error=0)

    t0 = time.time()
    result_df, _ = engine.run(query, constraints)
    elapsed = time.time() - t0
    print(f"  THALAMUSDB: {len(result_df)}/{len(df)} passed ({elapsed:.1f}s)")

    rows = []
    for item in log:
        rows.append(
            {
                "thalamusdb_input": item["input"],
                "thalamusdb_output": item["output"],
            }
        )

    out_path = f"logs/{project}_fever_filter.csv"
    write_csv(out_path, rows)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
