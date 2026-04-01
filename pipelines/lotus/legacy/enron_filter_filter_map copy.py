"""Pipeline: sem_filter -> sem_filter -> sem_map on Enron (LOTUS only)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

import time
from experiment_utils_lotus import (
    logger,
    load_enron,
    FILTER_ENRON_FRAUD,
    FILTER_ENRON_NOT_NEWS,
    MAP_ENRON_EXPLANATION,
)
from data_utils import write_csv

_empty = {"input": "", "output": "", "filename": "", "contents": ""}

print("\n" + "=" * 60)
print("  PIPELINE: filter -> filter -> map (enron)")
print("=" * 60)

project = "lotus"

# -- LOTUS --
t0 = time.time()
joined_df = load_enron("projects/palimpzest/testdata/enron-eval")
logger.clear()
df_filter1 = joined_df.sem_filter(FILTER_ENRON_FRAUD)
df_filter2 = df_filter1.sem_filter(FILTER_ENRON_NOT_NEWS)
df_map = df_filter2.sem_map(MAP_ENRON_EXPLANATION, suffix="fraud_explanation")
lotus_time = time.time() - t0
print(f"  LOTUS: {len(joined_df)}->{len(df_filter1)}->{len(df_filter2)} ({lotus_time:.1f}s)")

# Slice logger into stages (one LM call per row per op)
f1_len = len(df_filter1)
f2_len = len(df_filter2)
m_len = len(df_map)
lotus_f1_cap = logger[0:len(joined_df)]
lotus_f2_cap = logger[len(joined_df):len(joined_df) + f1_len]
lotus_m_cap = logger[len(joined_df) + f1_len:len(joined_df) + f1_len + m_len]

# -- Log (wide format: one row per original email, stage outputs as columns) --
rows = []
base_rows = []
row_pos_by_filename = {}

for i in range(len(joined_df)):
    row = joined_df.iloc[i]
    lm = lotus_f1_cap[i] if i < len(lotus_f1_cap) else _empty
    base = {
        "tuple": i,
        "filename": row["filename"],
        "contents": row["contents"][:120],
        "filter1_input": lm["input"],
        "filter1_output": lm["output"],
        "filter2_input": "",
        "filter2_output": "",
        "map_input": "",
        "map_output": "",
    }
    base_rows.append(base)
    row_pos_by_filename[row["filename"]] = i

for i in range(f1_len):
    row = df_filter1.iloc[i]
    lm = lotus_f2_cap[i] if i < len(lotus_f2_cap) else _empty
    pos = row_pos_by_filename.get(row["filename"])
    if pos is not None:
        base_rows[pos]["filter2_input"] = lm["input"]
        base_rows[pos]["filter2_output"] = lm["output"]

for i in range(m_len):
    row = df_map.iloc[i]
    lm = lotus_m_cap[i] if i < len(lotus_m_cap) else _empty
    pos = row_pos_by_filename.get(row["filename"])
    if pos is not None:
        base_rows[pos]["map_input"] = lm["input"]
        base_rows[pos]["map_output"] = lm["output"]

rows.extend(base_rows)

write_csv(f"logs/{project}enron_filter_filter_map.csv", rows)
print(f"  Saved logs/{project}enron_filter_filter_map.csv")
