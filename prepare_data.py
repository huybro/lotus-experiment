"""
One-time data preparation script.

Loads FEVER claims, builds the oracle Wikipedia KB, retrieves top-K
evidence per claim using sentence-transformers, and saves the joined
DataFrame to CSV. After running this once, experiment scripts can
simply load from the CSV.

Usage:
    python prepare_data.py [--n_claims 20] [--k 3]
"""
import argparse
import os

import pandas as pd

from data_loader import load_fever_claims, load_oracle_wiki_kb
from retrieval import retrieve_for_claims


def main():
    parser = argparse.ArgumentParser(description="Pre-retrieve FEVER evidence and save to CSV")
    parser.add_argument("--n_claims", type=int, default=20, help="Number of claims to load")
    parser.add_argument("--k", type=int, default=3, help="Top-K evidence per claim")
    parser.add_argument("--output", type=str, default="data/fever_claims_with_evidence.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    # Load claims
    claims_df = load_fever_claims(n=args.n_claims)
    claims_df["true_label"] = claims_df["label"].apply(lambda l: l == "SUPPORTS")

    # Load oracle Wikipedia KB
    wiki_df = load_oracle_wiki_kb(claims_split="labelled_dev", n_claims=args.n_claims)

    # Retrieve top-K evidence per claim
    joined_df = retrieve_for_claims(claims_df, wiki_df, query_col="claim", K=args.k)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joined_df.to_csv(args.output, index=False)
    print(f"\n✅ Saved {len(joined_df)} rows to {args.output}")
    print(f"   {args.n_claims} claims × {args.k} evidence = {len(joined_df)} pairs")
    print(f"   Columns: {list(joined_df.columns)}")


if __name__ == "__main__":
    main()
