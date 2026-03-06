"""Pipeline: sem_map (verdict) — LOTUS vs Palimpzest comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from experiment_utils import (
    state, joined_df,
    MAP_VERDICT,
    write_csv, find_match, pz_map_with_fallback,
)

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

# ── PZ ──
state.rewrite_mode = True
state.current_filter_instruction = None
state.captured.clear()
t0 = time.time()
pz_m_df = pz_map_with_fallback(
    MAP_VERDICT, joined_df, "verdict",
    "TRUE if the claim is supported by the evidence, FALSE otherwise.",
    ["content", "claim"],
)
pz_time = time.time() - t0
pz_cap = list(state.captured)
state.rewrite_mode = False
print(f"  PZ:    {len(pz_m_df)} rows ({pz_time:.1f}s)")

# ── Log ──
rows = []
for i in range(len(joined_df)):
    row = joined_df.iloc[i]
    lm = find_match(lotus_cap, row["claim"], row["content"])
    pm = find_match(pz_cap, row["claim"], row["content"])
    rows.append({
        "tuple": i, "claim": row["claim"][:80], "evidence": row["content"][:80],
        "lotus_input": lm["input"], "lotus_output": lm["output"],
        "pz_input": pm["input"], "pz_output": pm["output"],
    })
write_csv("logs/map_fever.csv", rows)
print(f"  Saved logs/map_fever.csv")
