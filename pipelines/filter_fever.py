"""Pipeline: sem_filter (relevance) — LOTUS vs Palimpzest comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import palimpzest as pz
from experiment_utils import (
    state, joined_df, pz_config,
    FILTER_RELEVANCE,
    write_csv, find_match,
)

print("\n" + "=" * 60)
print("  PIPELINE: filter only (relevance)")
print("=" * 60)

# ── LOTUS ──
state.rewrite_mode = False
state.captured.clear()
t0 = time.time()
df_f = joined_df.copy().sem_filter(FILTER_RELEVANCE)
lotus_time = time.time() - t0
lotus_cap = list(state.captured)
print(f"  LOTUS: {len(df_f)}/{len(joined_df)} passed ({lotus_time:.1f}s)")

# ── PZ ──
state.rewrite_mode = True
state.current_filter_instruction = FILTER_RELEVANCE
state.current_filter_cols = ["content", "claim"]
state.captured.clear()
t0 = time.time()
ds = pz.MemoryDataset(id="cmp-f1", vals=joined_df.to_dict("records"))
ds = ds.sem_filter(
    "The following evidence is relevant to the claim. Evidence: {content} Claim: {claim}",
    depends_on=["content", "claim"],
)
pz_df = ds.run(config=pz_config).to_df()
pz_time = time.time() - t0
pz_cap = list(state.captured)
state.rewrite_mode = False
print(f"  PZ:    {len(pz_df)}/{len(joined_df)} passed ({pz_time:.1f}s)")

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
write_csv("logs/filter_fever.csv", rows)
print(f"  Saved logs/filter_fever.csv")
