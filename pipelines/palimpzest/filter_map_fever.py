"""Pipeline: sem_filter → sem_map (relevance → verdict) — LOTUS vs Palimpzest comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
import palimpzest as pz
from experiment_utils_palimpzest import (
    state, joined_df, pz_config,
    FILTER_RELEVANCE, MAP_VERDICT,
    write_csv, pz_map_with_fallback,
)

_empty = {"input": "", "output": "", "claim": "", "content": ""}

print("\n" + "=" * 60)
print("  PIPELINE: filter → map (relevance → verdict)")
print("=" * 60)

# ── LOTUS: filter relevant evidence, then generate verdict ──
state.rewrite_mode = False
state.captured.clear()
t0 = time.time()
df_fm_f = joined_df.copy().sem_filter(FILTER_RELEVANCE)
lotus_f_cap = list(state.captured)
state.captured.clear()
df_fm_m = df_fm_f.sem_map(MAP_VERDICT, suffix="verdict")
lotus_m_cap = list(state.captured)
lotus_time = time.time() - t0
print(f"  LOTUS: filter={len(df_fm_f)}/{len(joined_df)}, map={len(df_fm_m)} rows ({lotus_time:.1f}s)")

# ── PZ: filter then map ──
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
    lf = lotus_f_cap[i] if i < len(lotus_f_cap) else _empty
    pf = pz_f_cap[i] if i < len(pz_f_cap) else _empty
    lm = lotus_m_cap[i] if i < len(lotus_m_cap) else _empty
    pm = pz_m_cap[i] if i < len(pz_m_cap) else _empty
    rows.append({
        "tuple": i, "claim": row["claim"][:80], "evidence": row["content"][:80],
        "lotus_filter_input": lf["input"], "lotus_filter_output": lf["output"],
        "pz_filter_input": pf["input"], "pz_filter_output": pf["output"],
        "lotus_map_input": lm["input"], "lotus_map_output": lm["output"],
        "pz_map_input": pm["input"], "pz_map_output": pm["output"],
    })
write_csv("logs/filter_map_fever.csv", rows)
print(f"  Saved logs/filter_map_fever.csv")
