"""
Shared retrieval module for FEVER experiments.

Provides embedding-based retrieval using sentence-transformers + FAISS,
independent of LOTUS or Palimpzest. Both systems use this to get identical
retrieved evidence for fair comparison.
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# Global cache for the embedding model
_model_cache = {}


def get_embedder(model_name: str = "intfloat/e5-base-v2") -> SentenceTransformer:
    """Get (or cache) a SentenceTransformer embedding model."""
    if model_name not in _model_cache:
        print(f"  Loading embedding model: {model_name} (on CPU)")
        _model_cache[model_name] = SentenceTransformer(model_name, device="cpu")
    return _model_cache[model_name]


def build_index(wiki_df: pd.DataFrame, content_col: str = "content",
                model_name: str = "intfloat/e5-base-v2") -> np.ndarray:
    """
    Encode all wiki articles into embeddings.

    Returns:
        np.ndarray of shape (n_articles, embed_dim)
    """
    embedder = get_embedder(model_name)
    texts = wiki_df[content_col].tolist()
    print(f"  Encoding {len(texts)} documents...")
    embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings


def retrieve(queries: list[str], wiki_df: pd.DataFrame, wiki_embeddings: np.ndarray,
             K: int = 3, model_name: str = "intfloat/e5-base-v2") -> pd.DataFrame:
    """
    Retrieve top-K wiki articles for each query using cosine similarity.

    Args:
        queries: List of query strings (claims or search queries).
        wiki_df: DataFrame with wiki articles (must have 'content' column).
        wiki_embeddings: Pre-computed embeddings for wiki_df rows.
        K: Number of top results per query.
        model_name: Embedding model name.

    Returns:
        DataFrame with one row per (query_index, retrieved_article) pair,
        containing the original query info and retrieved content.
    """
    embedder = get_embedder(model_name)
    query_embeddings = embedder.encode(queries, normalize_embeddings=True)

    # Cosine similarity (embeddings are normalized, so dot product = cosine sim)
    similarities = query_embeddings @ wiki_embeddings.T  # (n_queries, n_docs)

    results = []
    for i, query in enumerate(queries):
        top_k_indices = np.argsort(similarities[i])[::-1][:K]
        for idx in top_k_indices:
            results.append({
                "query_idx": i,
                "content": wiki_df.iloc[idx]["content"],
                "page_id": wiki_df.iloc[idx].get("page_id", ""),
                "similarity": float(similarities[i][idx]),
            })

    return pd.DataFrame(results)


def retrieve_for_claims(claims_df: pd.DataFrame, wiki_df: pd.DataFrame,
                        query_col: str = "claim", K: int = 3,
                        model_name: str = "intfloat/e5-base-v2") -> pd.DataFrame:
    """
    Retrieve top-K evidence for each claim and return a joined DataFrame.

    Args:
        claims_df: DataFrame with claims (must have 'id', 'claim', 'label', 'true_label').
        wiki_df: DataFrame with wiki articles.
        query_col: Column in claims_df to use as the query.
        K: Number of retrieved articles per claim.
        model_name: Embedding model name.

    Returns:
        DataFrame with claims joined to their retrieved evidence.
        Each claim appears K times (once per retrieved article).
    """
    print(f"\nRetrieving top-{K} evidence for {len(claims_df)} claims (query_col={query_col})...")

    # Build wiki embeddings (cached per call — could be improved with global cache)
    wiki_embeddings = build_index(wiki_df, model_name=model_name)

    # Retrieve
    queries = claims_df[query_col].tolist()
    retrieved = retrieve(queries, wiki_df, wiki_embeddings, K=K, model_name=model_name)

    # Join back with claims
    rows = []
    for _, claim_row in claims_df.iterrows():
        claim_idx = claim_row.name  # positional index
        matches = retrieved[retrieved["query_idx"] == claim_idx]
        for _, match in matches.iterrows():
            rows.append({
                "id": claim_row["id"],
                "claim": claim_row["claim"],
                "label": claim_row["label"],
                "true_label": claim_row["true_label"],
                "content": match["content"],
                "similarity": match["similarity"],
            })

    joined_df = pd.DataFrame(rows)
    print(f"  Retrieved {len(joined_df)} total (claim, evidence) pairs")
    return joined_df
