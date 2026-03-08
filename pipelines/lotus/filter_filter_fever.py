"""Pipeline: sem_filter → sem_filter (relevance → support) — LOTUS only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/..')

import time
from experiment_utils_lotus import (
    state, joined_df,
    FILTER_RELEVANCE, FILTER_SUPPORT,
    write_csv,
)

_empty = {"input": "", "output": "", "claim": "", "content": ""}

print("\n" + "=" * 60)
print("  PIPELINE: filter → filter (relevance → support)")
print("=" * 60)

# ── LOTUS ──
state.rewrite_mode = False
state.captured.clear()
t0 = time.time()
df_ff1 = joined_df.copy().sem_filter(FILTER_RELEVANCE)
lotus_f1_cap = list(state.captured)
state.captured.clear()
df_ff2 = df_ff1.sem_filter(FILTER_SUPPORT)
lotus_f2_cap = list(state.captured)
lotus_time = time.time() - t0
print(f"  LOTUS: {len(joined_df)}→{len(df_ff1)}→{len(df_ff2)} ({lotus_time:.1f}s)")

# ── Log ──
rows = []
for i in range(len(joined_df)):
    row = joined_df.iloc[i]
    lf1 = lotus_f1_cap[i] if i < len(lotus_f1_cap) else _empty
    lf2 = lotus_f2_cap[i] if i < len(lotus_f2_cap) else _empty
    rows.append({
        "tuple": i, "claim": row["claim"][:80], "evidence": row["content"][:80],
        "lotus_f1_input": lf1["input"], "lotus_f1_output": lf1["output"],
        "lotus_f2_input": lf2["input"], "lotus_f2_output": lf2["output"],
    })
write_csv("logs/lotusfilter_filter_fever.csv", rows)
print(f"  Saved logs/lotusfilter_filter_fever.csv")
