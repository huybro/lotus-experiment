#!/usr/bin/env python3
"""Compare input prompts from Lotus vs Palimpzest CSV files."""
import csv
import sys


def load_csv(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def main():
    lotus_path = "lotus_filter_fever.csv"
    pz_path = "palimpzestfilter_fever.csv"
    if len(sys.argv) >= 3:
        lotus_path, pz_path = sys.argv[1], sys.argv[2]

    lotus = load_csv(lotus_path)
    pz = load_csv(pz_path)

    print("=" * 70)
    print("ANALYSIS: Lotus vs Palimpzest Input Prompts")
    print("=" * 70)
    print(f"\nLotus: {len(lotus)} rows from {lotus_path}")
    print(f"PZ:    {len(pz)} rows from {pz_path}")

    # Row 0 comparison
    l0 = lotus[0].get("lotus_input", "")
    p0 = pz[0].get("pz_input", "")

    print("\n--- ROW 0 ---")
    print(f"Lotus claim (first 60): {lotus[0]['claim'][:60]}...")
    print(f"PZ claim (first 60):    {pz[0]['claim'][:60]}...")

    print("\n--- STRUCTURE ---")
    sys_same = (
        "You are a helpful assistant for executing semantic operators" in l0
        and "You are a helpful assistant for executing semantic operators" in p0
    )
    print(f"1. System prompt: {'SAME' if sys_same else 'DIFFERENT'}")

    l_user = l0.count("<|start_header_id|>user<|end_header_id|>")
    p_user = p0.count("<|start_header_id|>user<|end_header_id|>")
    print(f"2. User message count: Lotus={l_user}, PZ={p_user}")

    l_ctx_format = "flat" if "[Claim]" in l0 or "text\":" in l0 else "other"
    p_ctx_format = "json" if '"claim"' in p0 and '"content"' in p0 else "flat"
    print(f"3. CONTEXT format: Lotus={l_ctx_format}, PZ={p_ctx_format}")

    print("\n--- CONTEXT Snippet (Lotus) ---")
    if "CONTEXT:" in l0 and "TASK:" in l0:
        ctx = l0.split("CONTEXT:")[1].split("TASK:")[0][:400]
        print(repr(ctx))
    else:
        print("(not found)")

    print("\n--- CONTEXT Snippet (PZ) ---")
    if "CONTEXT:" in p0 and "TASK:" in p0:
        ctx = p0.split("CONTEXT:")[1].split("TASK:")[0][:400]
        print(repr(ctx))
    else:
        print("(not found)")

    print("\n--- TASK Snippet (both) ---")
    if "TASK:" in l0:
        t = l0.split("TASK:")[1][:250]
        print("Lotus:", repr(t[:250]))
    if "TASK:" in p0:
        t = p0.split("TASK:")[1][:250]
        print("PZ:   ", repr(t[:250]))

    # Check if identical when comparing same row by claim
    print("\n--- IDENTICAL ROWS? ---")
    identical = 0
    for i in range(min(len(lotus), len(pz))):
        if lotus[i].get("lotus_input") == pz[i].get("pz_input"):
            identical += 1
    print(f"Rows with identical input prompts: {identical}/{min(len(lotus), len(pz))}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
- System + TASK structure: SAME (both use prompt_utils with [system, user])
- CONTEXT data format: DIFFERENT
  * Lotus: "text": [Claim] X [Evidence] Y (flat string from df2text)
  * PZ:    "text": {"claim": "X", "content": "Y"} (JSON from PZ record serialization)

To make prompts identical, PZ would need to serialize context the same way
as Lotus (flat [Claim]/[Evidence] format) before passing to get_prompt.
""")


if __name__ == "__main__":
    main()
