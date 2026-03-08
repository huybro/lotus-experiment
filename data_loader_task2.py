"""
Data loader for Task 2: research abstracts + categories.

Loads data for: sem_map("Summarize...") + sem_filter("Is the research paper related...")
Supports HuggingFace dataset or CSV.
"""
import pandas as pd


def load_abstracts_with_categories(
    n: int = 100,
    source: str = "huggingface",
    csv_path: str | None = None,
) -> pd.DataFrame:
    """
    Load research abstracts with categories.

    Args:
        n: Number of samples to load.
        source: "huggingface", "csv", or "fever".
        csv_path: Path to CSV. For "csv": must have abstract,category. For "fever": uses content→abstract, claim→category.

    Returns:
        DataFrame with columns: abstract, category
    """
    if source == "fever" and csv_path:
        df = pd.read_csv(csv_path)
        if "content" not in df.columns or "claim" not in df.columns:
            raise ValueError(f"FEVER CSV must have 'content' and 'claim' columns. Found: {list(df.columns)}")
        df = df.rename(columns={"content": "abstract", "claim": "category"})
        return df[["abstract", "category"]].head(n).reset_index(drop=True)

    if source == "csv" and csv_path:
        df = pd.read_csv(csv_path)
        if "abstract" not in df.columns or "category" not in df.columns:
            raise ValueError(f"CSV must have 'abstract' and 'category' columns. Found: {list(df.columns)}")
        return df.head(n).reset_index(drop=True)

    if source == "huggingface":
        from datasets import load_dataset
        ds = load_dataset("ccdv/pubmed-summarization", "section", split="train", trust_code=True)
        df = ds.to_pandas()
        categories = ["Medicine", "Biology", "Genetics", "Neuroscience", "Oncology", "Immunology"]
        import numpy as np
        df["category"] = np.random.choice(categories, size=len(df))
        df = df.rename(columns={"article": "abstract"})
        df["abstract"] = df["abstract"].astype(str).str[:800]
        return df[["abstract", "category"]].head(n).reset_index(drop=True)

    raise ValueError("source must be 'huggingface', 'csv', or 'fever'. For csv/fever, provide csv_path.")
