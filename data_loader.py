"""
Data loader for the FEVER (Fact Extraction and VERification) dataset.

Loads claims from the labelled_dev split and builds an oracle Wikipedia
knowledge base using only the pages referenced in the claims' evidence.
"""
import pandas as pd
from datasets import load_dataset


def load_fever_claims(n: int = 20, split: str = "labelled_dev") -> pd.DataFrame:
    """
    Load claims from the FEVER dataset.

    Args:
        n: Number of claims to load (unique claims, after deduplication).
        split: Which split to use. Default is 'labelled_dev'.

    Returns:
        DataFrame with columns: id, claim, label
    """
    print(f"Loading FEVER claims from '{split}' split...")

    raw = load_dataset("fever", "v1.0", split=f"{split}[:{n * 5}]", trust_remote_code=True)
    df = pd.DataFrame(raw)

    # Deduplicate: keep one row per unique claim id
    claims_df = df.drop_duplicates(subset=["id"])[["id", "claim", "label"]].reset_index(drop=True)
    claims_df = claims_df.head(n)

    print(f"  Loaded {len(claims_df)} unique claims")
    print(f"  Label distribution: {dict(claims_df['label'].value_counts())}")
    return claims_df


def load_oracle_wiki_kb(claims_split: str = "labelled_dev", n_claims: int = 20) -> pd.DataFrame:
    """
    Build an oracle Wikipedia KB from the FEVER dataset.

    Loads only the Wikipedia articles that are referenced as evidence
    for the given claims. This is a standard evaluation shortcut that
    avoids loading the full 5.4M article corpus.

    Args:
        claims_split: Which claims split to extract evidence page IDs from.
        n_claims: How many claims worth of evidence pages to load.

    Returns:
        DataFrame with columns: page_id, content
        Each row is one sentence from a referenced Wikipedia page.
    """
    print("Building oracle Wikipedia KB...")

    # Step 1: Get all evidence wiki page IDs from the claims
    raw = load_dataset("fever", "v1.0", split=f"{claims_split}[:{n_claims * 5}]", trust_remote_code=True)
    df = pd.DataFrame(raw)
    wiki_urls = set(df[df["evidence_wiki_url"] != ""]["evidence_wiki_url"].tolist())
    print(f"  Found {len(wiki_urls)} unique Wikipedia pages referenced in evidence")

    if not wiki_urls:
        raise ValueError("No evidence pages found — all claims may be NOT ENOUGH INFO")

    # Step 2: Load wiki_pages and filter to only referenced pages
    print("  Loading FEVER wiki_pages (cached after first download ~1.7GB)...")
    wiki_ds = load_dataset("fever", "wiki_pages", split="wikipedia_pages", trust_remote_code=True)
    wiki_df = pd.DataFrame(wiki_ds)

    # Filter to only pages referenced by our claims
    oracle_pages = wiki_df[wiki_df["id"].isin(wiki_urls)]
    print(f"  Matched {len(oracle_pages)} pages out of {len(wiki_urls)} referenced")

    # Step 3: Extract individual sentences from the 'lines' field
    # Format: "0\tFirst sentence.\n1\tSecond sentence.\n..."
    sentences = []
    for _, page in oracle_pages.iterrows():
        page_id = page["id"]
        lines = page.get("lines", "")
        if not lines:
            continue
        for line in lines.split("\n"):
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[1].strip():
                sentence_text = parts[1].strip()
                # Only include meaningful sentences (skip very short/empty ones)
                if len(sentence_text) > 10:
                    readable_id = page_id.replace("_", " ").replace("-LRB-", "(").replace("-RRB-", ")")
                    sentences.append({
                        "page_id": page_id,
                        "content": f"{readable_id}: {sentence_text}"
                    })

    kb_df = pd.DataFrame(sentences)
    print(f"  Extracted {len(kb_df)} sentences for the knowledge base")
    return kb_df