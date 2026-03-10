"""Pipeline: sem_filter → sem_map (relevance → verdict) — PZ only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/..')

import time
import pandas as pd
import palimpzest as pz
from experiment_utils_palimpzest import (
    state, joined_df, pz_config,
    FILTER_RELEVANCE, MAP_VERDICT,
    write_csv, pz_map_with_fallback, find_match
)

_empty = {"input": "", "output": "", "claim": "", "content": ""}

print("\n" + "=" * 60)
print("  PIPELINE: filter → map (relevance → verdict)")
print("=" * 60)

# ── PZ ──
state.rewrite_mode = True
state.current_filter_instruction = FILTER_RELEVANCE
state.current_filter_cols = ["content", "claim"]
state.captured.clear()
t0 = time.time()
ds = pz.MemoryDataset(id="cmp-fm", vals=joined_df.to_dict("records"))
ds = ds.sem_filter(
    "The following evidence is relevant to the claim. Evidence: {content} Claim: {claim}",
    depends_on=["content", "claim"],
)
pz_fm_f_df = ds.run(config=pz_config).to_df()
pz_f_cap = list(state.captured)

state.current_filter_instruction = None  # next op is map
state.captured.clear()
if len(pz_fm_f_df) > 0:
    pz_fm_m_df = pz_map_with_fallback(
        MAP_VERDICT, pz_fm_f_df, "verdict",
        "TRUE if the claim is supported by the evidence, FALSE otherwise.",
        ["claim", "content"],
    )
else:
    pz_fm_m_df = pd.DataFrame()
pz_m_cap = list(state.captured)
pz_time = time.time() - t0
state.rewrite_mode = False
print(f"  PZ:    filter={len(pz_fm_f_df)}/{len(joined_df)}, map={len(pz_fm_m_df)} rows ({pz_time:.1f}s)")

# ── Log ──
rows = []
for i in range(len(joined_df)):
    row = joined_df.iloc[i]
    pf = find_match(row, pz_f_cap)
    pm = find_match(row, pz_m_cap)
    rows.append({
        "tuple": i, "claim": row["claim"][:80], "evidence": row["content"][:80],
        "pz_filter_input": pf["input"], "pz_filter_output": pf["output"],
        "pz_map_input": pm["input"], "pz_map_output": pm["output"],
    })
write_csv("logs/pzfilter_map_fever.csv", rows)
print(f"  Saved logs/pzfilter_map_fever.csv")
