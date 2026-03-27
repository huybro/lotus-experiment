"""Pipeline: sem_map (verdict) — PZ only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/..')

import time
from experiment_utils_palimpzest import (
    logger, joined_df,
    MAP_VERDICT,
    write_csv, pz_map_with_fallback, find_match
)

_empty = {"input": "", "output": "", "claim": "", "content": ""}

print("\n" + "=" * 60)
print("  PIPELINE: map only (verdict)")
print("=" * 60)

# ── PZ ──
logger.clear()
t0 = time.time()
pz_m_df = pz_map_with_fallback(
    MAP_VERDICT, joined_df, "verdict",
    "TRUE if the claim is supported by the evidence, FALSE otherwise.",
    ["claim", "content"],
)
pz_time = time.time() - t0
pz_cap = list(logger)
print(f"  PZ:    {len(pz_m_df)} rows ({pz_time:.1f}s)")

# ── Log ──
rows = []
for i in range(len(joined_df)):
    row = joined_df.iloc[i]
    pm = find_match(row, pz_cap)
    rows.append({
        "tuple": i, "claim": row["claim"][:80], "evidence": row["content"][:80],
        "pz_input": pm["input"], "pz_output": pm["output"],
    })
write_csv("logs/pzmap_fever.csv", rows)
print(f"  Saved logs/pzmap_fever.csv")
