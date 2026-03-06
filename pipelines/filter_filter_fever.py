"""Pipeline: sem_filter → sem_filter (relevance → support) — LOTUS vs Palimpzest comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
import palimpzest as pz
from experiment_utils import (
    state, joined_df, pz_config,
    FILTER_RELEVANCE, FILTER_SUPPORT,
    write_csv, find_match,
)

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

# ── PZ F1 ──
state.rewrite_mode = True
state.current_filter_instruction = FILTER_RELEVANCE
state.current_filter_cols = ["content", "claim"]
state.captured.clear()
t0 = time.time()
ds1 = pz.MemoryDataset(id="cmp-ff1", vals=joined_df.to_dict("records"))
ds1 = ds1.sem_filter(
    "The following evidence is relevant to the claim. Evidence: {content} Claim: {claim}",
    depends_on=["content", "claim"],
)
pz_ff1_df = ds1.run(config=pz_config).to_df()
pz_f1_cap = list(state.captured)

# ── PZ F2 ──
state.current_filter_instruction = FILTER_SUPPORT
state.captured.clear()
if len(pz_ff1_df) > 0:
    ds2 = pz.MemoryDataset(id="cmp-ff2", vals=pz_ff1_df.to_dict("records"))
    ds2 = ds2.sem_filter(
        "Based on the evidence, the following claim is supported. {content} {claim}",
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
    lf1 = find_match(lotus_f1_cap, row["claim"], row["content"])
    pf1 = find_match(pz_f1_cap, row["claim"], row["content"])
    lf2 = find_match(lotus_f2_cap, row["claim"], row["content"])
    pf2 = find_match(pz_f2_cap, row["claim"], row["content"])
    rows.append({
        "tuple": i, "claim": row["claim"][:80], "evidence": row["content"][:80],
        "lotus_f1_input": lf1["input"], "lotus_f1_output": lf1["output"],
        "pz_f1_input": pf1["input"], "pz_f1_output": pf1["output"],
        "lotus_f2_input": lf2["input"], "lotus_f2_output": lf2["output"],
        "pz_f2_input": pf2["input"], "pz_f2_output": pf2["output"],
    })
write_csv("logs/filter_filter_fever.csv", rows)
print(f"  Saved logs/filter_filter_fever.csv")
