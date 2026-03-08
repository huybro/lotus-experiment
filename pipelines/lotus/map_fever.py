"""Pipeline: sem_map (verdict) — LOTUS only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/..')

import time
from experiment_utils_lotus import (
    state, joined_df,
    MAP_VERDICT,
    write_csv,
)

_empty = {"input": "", "output": "", "claim": "", "content": ""}

print("\n" + "=" * 60)
print("  PIPELINE: map only (verdict)")
print("=" * 60)

# ── LOTUS ──
state.rewrite_mode = False
state.captured.clear()
t0 = time.time()
df_m = joined_df.copy().sem_map(MAP_VERDICT, suffix="verdict")
lotus_time = time.time() - t0
lotus_cap = list(state.captured)
print(f"  LOTUS: {len(df_m)} rows ({lotus_time:.1f}s)")

# ── Log ──
rows = []
for i in range(len(joined_df)):
    row = joined_df.iloc[i]
    lm = lotus_cap[i] if i < len(lotus_cap) else _empty
    rows.append({
        "tuple": i, "claim": row["claim"][:80], "evidence": row["content"][:80],
        "lotus_input": lm["input"], "lotus_output": lm["output"],
    })
write_csv("logs/lotusmap_fever.csv", rows)
print(f"  Saved logs/lotusmap_fever.csv")
