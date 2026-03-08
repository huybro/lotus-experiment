"""Pipeline: sem_filter (relevance) — LOTUS vs Palimpzest comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/..')

import time
from experiment_utils_lotus import (
    state, joined_df,
    FILTER_RELEVANCE,
    write_csv,
)

_empty = {"input": "", "output": "", "claim": "", "content": ""}

print("\n" + "=" * 60)
print("  PIPELINE: filter only (relevance)")
print("=" * 60)

project = 'lotus'
# ── LOTUS ──
state.rewrite_mode = False
state.captured.clear()

joined_df['claim'] = "[Claim] " + joined_df['claim']
joined_df['content'] = "[Evidence] " + joined_df['content']

t0 = time.time()
df_f = joined_df.copy().sem_filter(FILTER_RELEVANCE)
lotus_time = time.time() - t0
lotus_cap = list(state.captured)
print(f"  LOTUS: {len(df_f)}/{len(joined_df)} passed ({lotus_time:.1f}s)")


# ── Log ──
rows = []
for i in range(len(joined_df)):
    row = joined_df.iloc[i]
    lm = lotus_cap[i] if i < len(lotus_cap) else _empty
    rows.append({
        "tuple": i, "claim": row["claim"][:80], "evidence": row["content"][:80],
        "lotus_input": lm["input"], "lotus_output": lm["output"],
    })
write_csv(f"logs/{project}_filter_fever.csv", rows)

