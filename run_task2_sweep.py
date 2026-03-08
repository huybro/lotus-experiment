#!/usr/bin/env python3
"""
Task 2 sweep: map → filter with varying sample sizes.
Collects throughput (pipeline timing) and prefix cache hit rate (vLLM metrics).

Relaunches vLLM for each experiment (fresh cache state per run).

Usage:
  python run_task2_sweep.py
  python run_task2_sweep.py --samples 10 50 100 200
  python run_task2_sweep.py --no-relaunch   # Use if vLLM is already running (uses delta metrics)

Output: logs/task2_metrics.csv and logs/task2_cache_vs_throughput.png
"""
import argparse
import os
import re
import subprocess
import sys
import time
import csv
import urllib.request

import pandas as pd

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Lotus setup (before loading experiment_utils)
os.environ.setdefault("VLLM_API_BASE", "http://localhost:8000/v1")
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
VLLM_PORT = 8000

# Lotus config
import lotus
from lotus.models import LM

lotus.settings.configure(lm=LM(
    model=f"hosted_vllm/{MODEL_NAME}",
    api_base=os.environ.get("VLLM_API_BASE", f"http://127.0.0.1:{VLLM_PORT}/v1"),
    max_tokens=512,
    temperature=0,
))

from data_loader_task2 import load_abstracts_with_categories

MAP_INSTR = (
    "Summarize the research abstract and explain how it is related to the category.\n"
    "Abstract: {abstract}\nCategory: {category}"
)
FILTER_INSTR = (
    "Is the research paper related to the given category?\n"
    "Abstract: {abstract}\nCategory: {category}"
)


def start_vllm(port: int = 8000, model: str = "meta-llama/Llama-3.2-1B-Instruct") -> subprocess.Popen:
    """Start vLLM server in background, return process handle."""
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
            "--dtype", "float16",
            "--tensor-parallel-size", "1",
            "--max-model-len", "4096",
            "--enable-metrics",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return proc


def wait_for_vllm(base_url: str, max_wait: int = 300) -> bool:
    """Wait for vLLM /health to succeed. Returns True if ready."""
    url = base_url.rstrip("/").replace("/v1", "") + "/health"
    start = time.time()
    while time.time() - start < max_wait:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def stop_vllm(proc: subprocess.Popen) -> None:
    """Stop vLLM server."""
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def fetch_vllm_metrics(base_url: str) -> dict[str, float]:
    """Fetch vLLM /metrics and parse prefix cache counters."""
    url = base_url.rstrip("/").replace("/v1", "") + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            text = resp.read().decode()
    except Exception as e:
        print(f"  [WARN] Could not fetch vLLM metrics from {url}: {e}")
        return {"prefix_cache_hits": 0, "prefix_cache_queries": 0}

    hits = 0
    queries = 0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        m = re.search(r'vllm:prefix_cache_hits(?:\{[^}]*\})?\s+(\d+(?:\.\d+)?)', line)
        if m:
            hits = float(m.group(1))
            continue
        m = re.search(r'vllm:prefix_cache_queries(?:\{[^}]*\})?\s+(\d+(?:\.\d+)?)', line)
        if m:
            queries = float(m.group(1))
    return {"prefix_cache_hits": hits, "prefix_cache_queries": queries}


def run_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, float, int]:
    """Run map → filter, return (result_df, elapsed_sec, num_requests)."""
    t0 = time.time()
    df_map = df.copy().sem_map(MAP_INSTR, suffix="summary")
    df_filtered = df_map.sem_filter(FILTER_INSTR)
    elapsed = time.time() - t0
    num_requests = len(df) * 2  # map + filter per row
    return df_filtered, elapsed, num_requests


def main():
    parser = argparse.ArgumentParser(description="Task 2 sweep: throughput vs prefix cache hit rate")
    parser.add_argument("--samples", type=int, nargs="+", default=[10, 25, 50, 100, 200],
                        help="Sample sizes to sweep")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000",
                        help="vLLM server base URL (e.g. http://localhost:8000)")
    parser.add_argument("--output-dir", type=str, default="logs",
                        help="Output directory for CSV and plot")
    parser.add_argument("--data-source", type=str, default="huggingface",
                        choices=["huggingface", "csv", "fever"],
                        help="Data source (fever: maps content→abstract, claim→category)")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to CSV with abstract,category columns (if --data-source csv)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting")
    parser.add_argument("--no-relaunch", action="store_true",
                        help="Assume vLLM already running; use delta metrics instead of relaunch")
    parser.add_argument("--vllm-port", type=int, default=8000,
                        help="vLLM server port")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_url = args.vllm_url
    port = args.vllm_port
    relaunch = not args.no_relaunch
    results = []

    print("\n" + "=" * 60)
    print("  Task 2 sweep: map → filter (throughput vs prefix cache hit rate)")
    print("=" * 60)
    print(f"  Sample sizes: {args.samples}")
    print(f"  vLLM: {base_url}")
    print(f"  Relaunch per run: {relaunch}")
    print()

    max_n = max(args.samples)
    df_all = load_abstracts_with_categories(n=max_n, source=args.data_source, csv_path=args.csv_path)
    print(f"  Loaded {len(df_all)} samples")

    for n in args.samples:
        df = df_all.head(n).copy()
        print(f"\n  --- n={n} ---")

        vllm_proc = None
        if relaunch:
            print("    Starting vLLM...")
            vllm_proc = start_vllm(port=port, model=MODEL_NAME)
            run_base_url = f"http://localhost:{port}"
            if not wait_for_vllm(run_base_url, max_wait=300):
                print("    [ERROR] vLLM failed to start within 300s")
                if vllm_proc:
                    stop_vllm(vllm_proc)
                sys.exit(1)
            print("    vLLM ready.")
            base_url = run_base_url

        metrics_before = fetch_vllm_metrics(base_url) if not relaunch else None

        df_out, elapsed, num_requests = run_pipeline(df)
        throughput = num_requests / elapsed if elapsed > 0 else 0

        metrics_after = fetch_vllm_metrics(base_url)
        if relaunch:
            hits = metrics_after["prefix_cache_hits"]
            queries = metrics_after["prefix_cache_queries"]
            cache_hit_rate = hits / queries if queries > 0 else 0.0
        else:
            delta_hits = metrics_after["prefix_cache_hits"] - metrics_before["prefix_cache_hits"]
            delta_queries = metrics_after["prefix_cache_queries"] - metrics_before["prefix_cache_queries"]
            cache_hit_rate = delta_hits / delta_queries if delta_queries > 0 else 0.0

        if relaunch and vllm_proc:
            print("    Stopping vLLM...")
            stop_vllm(vllm_proc)
            vllm_proc = None

        print(f"    Throughput: {throughput:.1f} req/s")
        print(f"    Prefix cache hit rate: {cache_hit_rate*100:.1f}%")
        print(f"    Elapsed: {elapsed:.2f}s")

        results.append({
            "n_samples": n,
            "throughput": round(throughput, 2),
            "cache_hit_rate": round(cache_hit_rate, 4),
            "elapsed_sec": round(elapsed, 2),
            "num_requests": num_requests,
        })

    csv_path = os.path.join(args.output_dir, "task2_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Saved {csv_path}")

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            df_res = pd.DataFrame(results)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(df_res["cache_hit_rate"] * 100, df_res["throughput"], s=100, c=df_res["n_samples"], cmap="viridis")
            for _, r in df_res.iterrows():
                ax.annotate(f"n={int(r['n_samples'])}", (r["cache_hit_rate"]*100, r["throughput"]),
                            fontsize=9, ha="left")
            ax.set_xlabel("Prefix cache hit rate (%)")
            ax.set_ylabel("Throughput (requests/sec)")
            ax.set_title("Task 2: Cache hit rate vs throughput")
            plt.tight_layout()
            plot_path = os.path.join(args.output_dir, "task2_cache_vs_throughput.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  Saved {plot_path}")
        except ImportError:
            print("  [WARN] matplotlib not installed, skipping plot")

    print("\n  Done.")


if __name__ == "__main__":
    main()
