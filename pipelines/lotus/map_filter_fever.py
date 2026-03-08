"""Pipeline: sem_map → sem_filter (verdict → verify) — LOTUS only.

Tests data accumulation across chained operators: after sem_map adds a 'verdict' column,
sem_filter should see the original data (content, claim) PLUS the new verdict column.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/..')

import time
from experiment_utils_lotus import (
    state, joined_df,
    MAP_VERDICT, FILTER_VERDICT,
    write_csv,
)

_empty = {"input": "", "output": "", "claim": "", "content": ""}

print("\n" + "=" * 60)
print("  PIPELINE: map → filter (verdict → verify)")
print("=" * 60)

# ── LOTUS: map (generate verdict), then filter (verify verdict) ──
state.rewrite_mode = False
state.captured.clear()
t0 = time.time()

# Step 1: sem_map — adds "verdict" column
df_map = joined_df.copy().sem_map(MAP_VERDICT, suffix="verdict")
lotus_map_cap = list(state.captured)

# Step 2: sem_filter — should see content + claim + verdict
state.captured.clear()
df_filtered = df_map.sem_filter(FILTER_VERDICT)
lotus_filter_cap = list(state.captured)
lotus_time = time.time() - t0
print(f"  LOTUS: map={len(df_map)}, filter={len(df_filtered)}/{len(df_map)} ({lotus_time:.1f}s)")

# ── Log ──
rows = []
for i in range(len(joined_df)):
    row = joined_df.iloc[i]
    lm = lotus_map_cap[i] if i < len(lotus_map_cap) else _empty
    lf = lotus_filter_cap[i] if i < len(lotus_filter_cap) else _empty
    rows.append({
        "tuple": i, "claim": row["claim"][:80], "evidence": row["content"][:80],
        "lotus_map_input": lm["input"], "lotus_map_output": lm["output"],
        "lotus_filter_input": lf["input"], "lotus_filter_output": lf["output"],
    })
write_csv("logs/lotusmap_filter_fever.csv", rows)
print(f"  Saved logs/lotusmap_filter_fever.csv")
