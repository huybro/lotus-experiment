

"""Pipeline: sem_filter (relevance) — PZ only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/..')

import time
import palimpzest as pz
from experiment_utils_palimpzest import (
    state, joined_df, pz_config,
    FILTER_RELEVANCE,
    write_csv,
)

_empty = {"input": "", "output": "", "claim": "", "content": ""}

print("\n" + "=" * 60)
print("  PIPELINE: filter only (relevance)")
print("=" * 60)

# ── PZ ──
state.rewrite_mode = True
state.current_filter_instruction = FILTER_RELEVANCE
state.current_filter_cols = ["claim", "content"]
state.captured.clear()
t0 = time.time()
ds = pz.MemoryDataset(id="cmp-f1", vals=joined_df.to_dict("records"))
ds = ds.sem_filter(
    "The following evidence is relevant to the claim. Claim: {claim} Evidence: {content}",
    depends_on=["claim", "content"],
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
    pm = pz_cap[i] if i < len(pz_cap) else _empty
    rows.append({
        "tuple": i, "claim": row["claim"][:80], "evidence": row["content"][:80],
        "pz_input": pm["input"], "pz_output": pm["output"],
    })
write_csv("logs/pzfilter_fever.csv", rows)
print(f"  Saved logs/pzfilter_fever.csv")
