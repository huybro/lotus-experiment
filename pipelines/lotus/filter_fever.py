"""Pipeline: sem_filter (relevance) — LOTUS only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/..')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/../projects/lotus")

import time
from experiment_utils_lotus import (
    logger,
    FILTER_RELEVANCE,
    )
from data_utils import write_csv, load_fever

_empty = {"input": "", "output": "", "claim": "", "content": ""}

print("\n" + "=" * 60)
print("  PIPELINE: filter only (relevance)")
print("=" * 60)

project = 'lotus'
# ── LOTUS ──
t0 = time.time()
joined_df = load_fever("data/fever_claims_with_evidence.csv")
df = joined_df.sem_filter(FILTER_RELEVANCE)
lotus_time = time.time() - t0
lotus_cap = logger
print(f"  LOTUS: {len(df)}/{len(joined_df)} passed ({lotus_time:.1f}s)")


# ── Log ──
rows = []
for i in range(len(joined_df)):
    row = joined_df.iloc[i]
    lm = lotus_cap[i] if i < len(lotus_cap) else _empty
    rows.append({
        "tuple": i, "claim": row["claim"][:80], "evidence": row["content"][:80],
        "lotus_input": lm["input"], "lotus_output": lm["output"],
    })

write_csv(f"logs/{project}filter_fever.csv", rows)
print(f"  Saved logs/filter_fever.csv")

