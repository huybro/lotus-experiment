#!/usr/bin/env python3
"""
Task 2 sweep: filter → map with varying sample sizes.
Collects throughput and prefix cache hit rate (vLLM metrics).

Usage:
  python hit_rate_throughput_script.py --csv-path data/fever_claims_with_evidence.csv
  python hit_rate_throughput_script.py --csv-path data/my_data.csv --samples 10 50 100 200
  python hit_rate_throughput_script.py --csv-path data/my_data.csv --no-relaunch

"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
import urllib.request

import litellm
import pandas as pd

import lotus
from lotus.models import LM

# Paths and config
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

os.environ.setdefault("VLLM_API_BASE", "http://localhost:8003/v1")
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
VLLM_PORT = 8003

lotus.settings.configure(lm=LM(
    model=f"hosted_vllm/{MODEL_NAME}",
    api_base=os.environ.get("VLLM_API_BASE", f"http://127.0.0.1:{VLLM_PORT}/v1"),
    max_tokens=512,
    temperature=0,
))

MAP_VERDICT = (
    "Explain how the claim can be supported by the evidence.\n"
    "Abstract: {abstract}\nCategory: {category}\n"
    "Provide a short explanation in natural language."
)
FILTER_INSTR = (
    "The sentence can determine whether the claim is true or false.\n"
    "Abstract: {abstract}\nCategory: {category}\n"
    "Answer TRUE if the context is sufficient to judge the claim, and FALSE otherwise.\n"
    "Output TRUE or FALSE only."
)


def load_data(csv_path: str, n: int) -> pd.DataFrame:
    """Load CSV and return DataFrame with abstract, category columns."""
    df = pd.read_csv(csv_path)
    if "content" in df.columns and "claim" in df.columns:
        df = df.rename(columns={"content": "abstract", "claim": "category"})
    if "abstract" not in df.columns or "category" not in df.columns:
        raise ValueError(
            f"CSV must have (abstract, category) or (content, claim). Found: {list(df.columns)}"
        )
    return df[["abstract", "category"]].head(n).reset_index(drop=True)


def start_vllm(port: int = 8003, model: str = MODEL_NAME) -> subprocess.Popen:
    """Start vLLM server in background."""
    return subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
            "--dtype", "float16",
            "--tensor-parallel-size", "1",
            "--max-model-len", "4096",
            "--enable-prefix-caching",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def wait_for_vllm(base_url: str, max_wait: int = 300, verbose: bool = True) -> bool:
    """Poll /health until vLLM responds or max_wait is exceeded."""
    url = base_url.rstrip("/").replace("/v1", "") + "/health"
    start = time.time()
    last_msg = 0
    while time.time() - start < max_wait:
        elapsed = int(time.time() - start)
        if verbose and elapsed > 0 and elapsed - last_msg >= 15:
            print(f"    Waiting for vLLM... ({elapsed}s)")
            last_msg = elapsed
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def warm_up_vllm(api_base: str, n: int = 20) -> None:
    """Send a few dummy completions to warm up vLLM (CUDA graphs, JIT)."""
    api_url = api_base.rstrip("/")
    if not api_url.endswith("/v1"):
        api_url = api_url + "/v1"
    for _ in range(n):
        litellm.completion(
            model=f"hosted_vllm/{MODEL_NAME}",
            messages=[{"role": "user", "content": "Say OK" * 100}],
            max_tokens=5,
            temperature=0,
            api_base=api_url,
        )


def stop_vllm(proc: subprocess.Popen) -> None:
    """Stop vLLM server."""
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def fetch_vllm_metrics(base_url: str, debug: bool = False) -> dict[str, float]:
    """Fetch /metrics and parse prefix_cache_hits, prefix_cache_queries."""
    url = base_url.rstrip("/").replace("/v1", "") + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            text = resp.read().decode()
    except Exception as e:
        print(f"  [WARN] Could not fetch metrics: {e}")
        return {"prefix_cache_hits": 0, "prefix_cache_queries": 0}

    hits = 0.0
    queries = 0.0
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") or not s:
            continue
        if debug and "prefix_cache" in s.lower():
            print(f"    [DEBUG] {s}")
        m = re.search(r'vllm:prefix_cache_hits(?:_total)?(?:\{[^}]*\})?\s+(\d+(?:\.\d+)?)', s)
        if m:
            hits = float(m.group(1))
        m = re.search(r'vllm:prefix_cache_queries(?:_total)?(?:\{[^}]*\})?\s+(\d+(?:\.\d+)?)', s)
        if m:
            queries = float(m.group(1))

    if debug:
        print(f"    [DEBUG] hits={hits}, queries={queries}")
    return {"prefix_cache_hits": hits, "prefix_cache_queries": queries}


def run_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, float, int]:
    """Run filter → map, return (result_df, elapsed_sec, num_requests)."""
    t0 = time.time()
    df_f = df.copy().sem_filter(FILTER_INSTR)
    df_out = df_f.sem_map(MAP_VERDICT)
    elapsed = time.time() - t0
    num_requests = len(df) + len(df_f)  # filter + map calls
    return df_out, elapsed, num_requests