"""Pipeline: sem_filter → sem_filter (relevance → support) — PZ only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/..')

import time
import pandas as pd
import palimpzest as pz
from experiment_utils_palimpzest import (
    state, joined_df, pz_config,
    FILTER_RELEVANCE, FILTER_SUPPORT,
    write_csv, find_match
)

_empty = {"input": "", "output": "", "claim": "", "content": ""}

print("\n" + "=" * 60)
print("  PIPELINE: filter → filter (relevance → support)")
print("=" * 60)

# ── PZ ──
state.rewrite_mode = True
state.current_filter_instruction = FILTER_RELEVANCE
state.current_filter_cols = ["claim", "content"]
state.captured.clear()
t0 = time.time()
ds1 = pz.MemoryDataset(id="cmp-ff1", vals=joined_df.to_dict("records"))
ds1 = ds1.sem_filter(
    "The following evidence is relevant to the claim. Claim: {claim} Evidence: {content}",
    depends_on=["claim", "content"],
)
pz_ff1_df = ds1.run(config=pz_config).to_df()
pz_f1_cap = list(state.captured)

# ── PZ F2 ──
state.current_filter_instruction = FILTER_SUPPORT
state.captured.clear()
if len(pz_ff1_df) > 0:
    ds2 = pz.MemoryDataset(id="cmp-ff2", vals=pz_ff1_df.to_dict("records"))
    ds2 = ds2.sem_filter(
        "{content}\nBased on the above evidence, the following claim is supported: {claim}",
        depends_on=["content", "claim"],
    )
    pz_ff2_df = ds2.run(config=pz_config).to_df()
else:
    pz_ff2_df = pd.DataFrame()
pz_f2_cap = list(state.captured)
pz_time = time.time() - t0
state.rewrite_mode = False
print(f"  PZ:    {len(joined_df)}→{len(pz_ff1_df)}→{len(pz_ff2_df)} ({pz_time:.1f}s)")

# ── Log ──
rows = []
for i in range(len(joined_df)):
    row = joined_df.iloc[i]
    pf1 = find_match(row, pz_f1_cap)
    pf2 = find_match(row, pz_f2_cap)
    rows.append({
        "tuple": i, "claim": row["claim"][:80], "evidence": row["content"][:80],
        "pz_f1_input": pf1["input"], "pz_f1_output": pf1["output"],
        "pz_f2_input": pf2["input"], "pz_f2_output": pf2["output"],
    })
write_csv("logs/pzfilter_filter_fever.csv", rows)
print(f"  Saved logs/pzfilter_filter_fever.csv")
