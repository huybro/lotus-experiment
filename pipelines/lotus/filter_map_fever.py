"""Pipeline: sem_filter → sem_map (relevance → verdict) — LOTUS only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/..')

import time
from experiment_utils_lotus import (
    state, joined_df,
    FILTER_RELEVANCE, MAP_VERDICT,
    write_csv,
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

# ── Log ──
rows = []
for i in range(len(joined_df)):
    row = joined_df.iloc[i]
    lf = lotus_f_cap[i] if i < len(lotus_f_cap) else _empty
    lm = lotus_m_cap[i] if i < len(lotus_m_cap) else _empty
    rows.append({
        "tuple": i, "claim": row["claim"][:80], "evidence": row["content"][:80],
        "lotus_filter_input": lf["input"], "lotus_filter_output": lf["output"],
        "lotus_map_input": lm["input"], "lotus_map_output": lm["output"],
    })
write_csv("logs/lotusfilter_map_fever.csv", rows)
print(f"  Saved logs/lotusfilter_map_fever.csv")
