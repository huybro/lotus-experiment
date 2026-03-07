"""Pipeline: sem_map → sem_filter (verdict → verify) — LOTUS vs Palimpzest comparison.

Tests data accumulation across chained operators: after sem_map adds a 'verdict' column,
sem_filter should see the original data (content, claim) PLUS the new verdict column.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
import palimpzest as pz
from experiment_utils_palimpzest import (
    state, joined_df, pz_config,
    MAP_VERDICT, FILTER_VERDICT,
    write_csv, pz_map_with_fallback,
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

# ── PZ: map then filter ──
state.rewrite_mode = True
state.current_filter_instruction = None
state.current_map_instruction = MAP_VERDICT
state.captured.clear()
t0 = time.time()

# Step 1: PZ sem_map
pz_map_df = pz_map_with_fallback(
    MAP_VERDICT, joined_df, "verdict",
    "TRUE if the claim is supported by the evidence, FALSE otherwise.",
    ["content", "claim"],
)
pz_map_cap = list(state.captured)

# Step 2: PZ sem_filter — should see content + claim + verdict
state.current_filter_instruction = FILTER_VERDICT
state.current_filter_cols = ["content", "claim", "verdict"]
state.current_map_instruction = None
state.captured.clear()
if len(pz_map_df) > 0:
    ds = pz.MemoryDataset(id="cmp-mf", vals=pz_map_df.to_dict("records"))
    ds = ds.sem_filter(
        "Based on the evidence and verdict, the claim is correctly assessed."
        " Evidence: {content} Claim: {claim} Verdict: {verdict}",
        depends_on=["content", "claim", "verdict"],
    )
    pz_filter_df = ds.run(config=pz_config).to_df()
else:
    pz_filter_df = pd.DataFrame()
pz_filter_cap = list(state.captured)
pz_time = time.time() - t0
state.rewrite_mode = False
print(f"  PZ:    map={len(pz_map_df)}, filter={len(pz_filter_df)}/{len(pz_map_df)} ({pz_time:.1f}s)")

# ── Log ──
rows = []
for i in range(len(joined_df)):
    row = joined_df.iloc[i]
    lm = lotus_map_cap[i] if i < len(lotus_map_cap) else _empty
    pm = pz_map_cap[i] if i < len(pz_map_cap) else _empty
    lf = lotus_filter_cap[i] if i < len(lotus_filter_cap) else _empty
    pf = pz_filter_cap[i] if i < len(pz_filter_cap) else _empty
    rows.append({
        "tuple": i, "claim": row["claim"][:80], "evidence": row["content"][:80],
        "lotus_map_input": lm["input"], "lotus_map_output": lm["output"],
        "pz_map_input": pm["input"], "pz_map_output": pm["output"],
        "lotus_filter_input": lf["input"], "lotus_filter_output": lf["output"],
        "pz_filter_input": pf["input"], "pz_filter_output": pf["output"],
    })
write_csv("logs/map_filter_fever.csv", rows)
print(f"  Saved logs/map_filter_fever.csv")
