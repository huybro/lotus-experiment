import os
import pytest

from docetl.runner import DSLRunner


pytest.importorskip("lancedb")


def test_lancedb_retriever_fts_only(tmp_path):
    kb = [
        {"id": 1, "text": "alpha beta"},
        {"id": 2, "text": "gamma delta"},
        {"id": 3, "text": "epsilon zeta"},
    ]

    config = {
        "default_model": "gpt-4o-mini",
        "datasets": {
            "kb": {
                "type": "memory",
                "path": kb,
            },
        },
        "retrievers": {
            "kb_r": {
                "type": "lancedb",
                "dataset": "kb",
                "index_dir": str(tmp_path / "idx"),
                "build_index": "always",
                "index_types": ["fts"],
                "fts": {
                    "index_phrase": "{{ input.text }}",
                    "query_phrase": "{{ input.q }}",
                },
                "query": {"top_k": 2, "mode": "fts"},
            }
        },
        "operations": [],
        "pipeline": {"steps": [], "output": {"path": str(tmp_path / "out.json")}},
    }

    runner = DSLRunner(config, max_threads=8)
    # Ensure datasets loaded for retriever indexing
    runner.load()
    r = runner.retrievers["kb_r"]
    r.ensure_index()

    res = r.retrieve({"input": {"q": "alpha"}})
    assert res.docs, "Expected at least one FTS hit"
    ctx = res.rendered_context
    assert "alpha" in ctx, f"retrieval_context should include indexed text, got: {ctx}"

@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Requires OpenAI embeddings (set OPENAI_API_KEY)",
)
def test_lancedb_retriever_embedding_only(tmp_path):
    kb = [
        {"id": 1, "text": "alpha beta"},
        {"id": 2, "text": "gamma delta"},
        {"id": 3, "text": "epsilon zeta"},
    ]

    config = {
        "default_model": "gpt-4o-mini",
        "datasets": {
            "kb": {
                "type": "memory",
                "path": kb,
            },
        },
        "retrievers": {
            "kb_r": {
                "type": "lancedb",
                "dataset": "kb",
                "index_dir": str(tmp_path / "idx2"),
                "build_index": "always",
                "index_types": ["embedding"],
                "embedding": {
                    "model": "text-embedding-3-small",
                    "index_phrase": "{{ input.text }}",
                    "query_phrase": "{{ input.q }}",
                },
                "query": {"top_k": 2, "mode": "embedding"},
            }
        },
        "operations": [],
        "pipeline": {"steps": [], "output": {"path": str(tmp_path / "out2.json")}},
    }

    runner = DSLRunner(config, max_threads=8)
    runner.load()
    r = runner.retrievers["kb_r"]
    r.ensure_index()

    res = r.retrieve({"input": {"q": "alpha"}})
    assert res.docs, "Expected at least one embedding hit"
    assert "alpha" in res.rendered_context or "beta" in res.rendered_context


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Requires OpenAI embeddings (set OPENAI_API_KEY)",
)
def test_lancedb_retriever_hybrid(tmp_path):
    kb = [
        {"id": 1, "text": "alpha beta"},
        {"id": 2, "text": "gamma delta"},
        {"id": 3, "text": "epsilon zeta"},
    ]

    config = {
        "default_model": "gpt-4o-mini",
        "datasets": {
            "kb": {
                "type": "memory",
                "path": kb,
            },
        },
        "retrievers": {
            "kb_r": {
                "type": "lancedb",
                "dataset": "kb",
                "index_dir": str(tmp_path / "idx3"),
                "build_index": "always",
                "index_types": ["fts", "embedding"],
                "fts": {
                    "index_phrase": "{{ input.text }}",
                    "query_phrase": "{{ input.q }}",
                },
                "embedding": {
                    "model": "text-embedding-3-small",
                    "index_phrase": "{{ input.text }}",
                    "query_phrase": "{{ input.q }}",
                },
                "query": {"top_k": 2, "mode": "hybrid"},
            }
        },
        "operations": [],
        "pipeline": {"steps": [], "output": {"path": str(tmp_path / "out3.json")}},
    }

    runner = DSLRunner(config, max_threads=8)
    runner.load()
    r = runner.retrievers["kb_r"]
    r.ensure_index()

    res = r.retrieve({"input": {"q": "alpha"}})
    assert res.docs, "Expected at least one hybrid hit"
    assert "alpha" in res.rendered_context


def test_lancedb_retriever_different_contexts_per_query(tmp_path):
    """Test that different queries return different retrieval contexts."""
    kb = [
        {"id": 1, "text": "The quick brown fox jumps over the lazy dog"},
        {"id": 2, "text": "Python is a programming language used for data science"},
        {"id": 3, "text": "Machine learning models require training data"},
        {"id": 4, "text": "Cats and dogs are popular household pets"},
    ]

    config = {
        "default_model": "gpt-4o-mini",
        "datasets": {
            "kb": {
                "type": "memory",
                "path": kb,
            },
        },
        "retrievers": {
            "kb_r": {
                "type": "lancedb",
                "dataset": "kb",
                "index_dir": str(tmp_path / "idx_diff"),
                "build_index": "always",
                "index_types": ["fts"],
                "fts": {
                    "index_phrase": "{{ input.text }}",
                    "query_phrase": "{{ input.query }}",
                },
                "query": {"top_k": 1, "mode": "fts"},
            }
        },
        "operations": [],
        "pipeline": {"steps": [], "output": {"path": str(tmp_path / "out_diff.json")}},
    }

    runner = DSLRunner(config, max_threads=8)
    runner.load()
    r = runner.retrievers["kb_r"]
    r.ensure_index()

    # Query for "fox" should return the fox document
    res_fox = r.retrieve({"input": {"query": "fox"}})
    assert res_fox.docs, "Expected at least one FTS hit for 'fox'"
    ctx_fox = res_fox.rendered_context

    # Query for "Python" should return the Python document
    res_python = r.retrieve({"input": {"query": "Python programming"}})
    assert res_python.docs, "Expected at least one FTS hit for 'Python'"
    ctx_python = res_python.rendered_context

    # Query for "machine learning" should return the ML document
    res_ml = r.retrieve({"input": {"query": "machine learning"}})
    assert res_ml.docs, "Expected at least one FTS hit for 'machine learning'"
    ctx_ml = res_ml.rendered_context

    # Query for "cats dogs pets" should return the pets document
    res_pets = r.retrieve({"input": {"query": "cats dogs pets"}})
    assert res_pets.docs, "Expected at least one FTS hit for 'pets'"
    ctx_pets = res_pets.rendered_context

    # Verify each context contains the expected content
    assert "fox" in ctx_fox.lower(), f"Expected 'fox' in context, got: {ctx_fox}"
    assert (
        "python" in ctx_python.lower()
    ), f"Expected 'python' in context, got: {ctx_python}"
    assert (
        "machine" in ctx_ml.lower() or "learning" in ctx_ml.lower()
    ), f"Expected ML terms in context, got: {ctx_ml}"
    assert (
        "cats" in ctx_pets.lower() or "dogs" in ctx_pets.lower()
    ), f"Expected pet terms in context, got: {ctx_pets}"

    # Verify contexts are different from each other
    contexts = [ctx_fox, ctx_python, ctx_ml, ctx_pets]
    for i, ctx1 in enumerate(contexts):
        for j, ctx2 in enumerate(contexts):
            if i < j:
                assert (
                    ctx1 != ctx2
                ), f"Contexts {i} and {j} should be different but both are: {ctx1}"
