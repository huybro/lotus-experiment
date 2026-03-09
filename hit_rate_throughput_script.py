#!/usr/bin/env python3
"""
Task 2 sweep: map → filter with varying sample sizes.
Collects throughput (pipeline timing) and prefix cache hit rate (vLLM metrics).

Relaunches vLLM for each experiment (fresh cache state per run).

Usage:
  python run_task2_sweep.py --csv-path data/fever_claims_with_evidence.csv
  python run_task2_sweep.py --csv-path data/my_data.csv --samples 10 50 100 200
  python run_task2_sweep.py --csv-path data/my_data.csv --no-relaunch

  CSV must have (abstract, category) or (content, claim) columns.

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
import litellm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Lotus setup (before loading experiment_utils)
os.environ.setdefault("VLLM_API_BASE", "http://localhost:8003/v1")
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
VLLM_PORT = 8003

# Lotus config
import lotus
from lotus.models import LM

lotus.settings.configure(lm=LM(
    model=f"hosted_vllm/{MODEL_NAME}",
    api_base=os.environ.get("VLLM_API_BASE", f"http://127.0.0.1:{VLLM_PORT}/v1"),
    max_tokens=512,
    temperature=0,
))


def load_data(csv_path: str, n: int) -> pd.DataFrame:
    """Load CSV and return DataFrame with abstract, category columns."""
    df = pd.read_csv(csv_path)
    if "content" in df.columns and "claim" in df.columns:
        df = df.rename(columns={"content": "abstract", "claim": "category"})
    if "abstract" not in df.columns or "category" not in df.columns:
        raise ValueError(f"CSV must have (abstract, category) or (content, claim). Found: {list(df.columns)}")
    return df[["abstract", "category"]].head(n).reset_index(drop=True)


MAP_VERDICT = (
    "Explain how the claim can be supported by the evidence.\n"
    "Provide a short explanation in natural language."
)

FILTER_INSTR = (
    "The sentence can determine whether the claim is true or false.\n"
    "Answer TRUE if the context is sufficient to judge the claim, and FALSE otherwise.\n"
    "Output TRUE or FALSE only."
)


def start_vllm(port: int = 8003, model: str = "meta-llama/Llama-3.2-1B-Instruct") -> subprocess.Popen:
    """Start vLLM server in background, return process handle."""
    proc = subprocess.Popen(
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
    return proc


def wait_for_vllm(base_url: str, max_wait: int = 300, verbose: bool = True) -> bool:
    """Wait for vLLM /health to succeed. Returns True if ready."""
    url = base_url.rstrip("/").replace("/v1", "") + "/health"
    start = time.time()
    last_msg = 0
    while time.time() - start < max_wait:
        elapsed = int(time.time() - start)
        if verbose and elapsed > 0 and elapsed - last_msg >= 15:
            print(f"    Waiting for vLLM... ({elapsed}s, can take 2–5 min on first load)")
            last_msg = elapsed
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def warm_up_vllm(api_base: str, num_requests: int = 2) -> None:
    api_url = api_base.rstrip("/")
    if not api_url.endswith("/v1"):
        api_url = api_url + "/v1"
    for _ in range(num_requests):
        litellm.completion(
            model=f"hosted_vllm/{MODEL_NAME}",
            messages=[{"role": "user", "content": "Say OK"}],
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
        line_stripped = line.strip()
        if line_stripped.startswith("#") or not line_stripped:
            continue
        if "prefix_cache" in line_stripped.lower():
            if debug:
                print(f"    [DEBUG] {line_stripped}")
        # vLLM 0.15+ uses _total suffix (e.g. prefix_cache_hits_total)
        m = re.search(r'vllm:prefix_cache_hits(?:_total)?(?:\{[^}]*\})?\s+(\d+(?:\.\d+)?)', line_stripped)
        if m:
            hits = float(m.group(1))
            continue
        m = re.search(r'vllm:prefix_cache_queries(?:_total)?(?:\{[^}]*\})?\s+(\d+(?:\.\d+)?)', line_stripped)
        if m:
            queries = float(m.group(1))
    if debug:
        print(f"    [DEBUG] Parsed: prefix_cache_hits={hits}, prefix_cache_queries={queries}")
    return {"prefix_cache_hits": hits, "prefix_cache_queries": queries}


def run_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, float, int]:
    """Run map → filter, return (result_df, elapsed_sec, num_requests)."""
    t0 = time.time()
    df_filtered = df.copy().sem_filter(FILTER_INSTR).sem_map(MAP_VERDICT)
    elapsed = time.time() - t0
    num_requests = len(df)
    return df_filtered, elapsed, num_requests


def main():
    parser = argparse.ArgumentParser(description="Task 2 sweep: throughput vs prefix cache hit rate")
    parser.add_argument("--samples", type=int, nargs="+", default=[10, 25, 50, 100, 200],
                        help="Sample sizes to sweep")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8003",
                        help="vLLM server base URL (e.g. http://localhost:8003)")
    parser.add_argument("--output-dir", type=str, default="logs",
                        help="Output directory for CSV and plot")
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path to CSV. Use (abstract, category) or (content, claim) columns")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting")
    parser.add_argument("--no-relaunch", action="store_true",
                        help="Assume vLLM already running; use delta metrics instead of relaunch")
    parser.add_argument("--debug-metrics", action="store_true",
                        help="Print raw prefix cache metric lines from /metrics")
    parser.add_argument("--vllm-port", type=int, default=8003,
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
    df_all = load_data(args.csv_path, n=max_n)
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
            print("    Warming up vLLM (2 dummy requests)...")
            warm_up_vllm(f"{base_url}/v1")
            print("    Warm-up done.")

        metrics_before = fetch_vllm_metrics(base_url, debug=args.debug_metrics) if not relaunch else None

        df_out, elapsed, num_requests = run_pipeline(df)
        throughput = num_requests / elapsed if elapsed > 0 else 0

        metrics_after = fetch_vllm_metrics(base_url, debug=args.debug_metrics)
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

            # X axis: number of tuples (n_samples)
            x = df_res["n_samples"]

            fig, ax1 = plt.subplots(figsize=(8, 5))

            # Left Y axis: throughput
            color1 = "tab:blue"
            ax1.set_xlabel("Number of tuples (n_samples)")
            ax1.set_ylabel("Throughput (requests/sec)", color=color1)
            ax1.plot(x, df_res["throughput"], marker="o", color=color1, label="Throughput")
            ax1.tick_params(axis="y", labelcolor=color1)

            # Right Y axis: cache hit rate (%)
            ax2 = ax1.twinx()
            color2 = "tab:green"
            ax2.set_ylabel("Prefix cache hit rate (%)", color=color2)
            ax2.plot(x, df_res["cache_hit_rate"] * 100, marker="s", linestyle="--", color=color2, label="Cache hit rate")
            ax2.tick_params(axis="y", labelcolor=color2)

            fig.suptitle("Task 2: Throughput and cache hit rate vs number of tuples")
            fig.tight_layout()

            plot_path = os.path.join(args.output_dir, "task2_cache_vs_throughput.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  Saved {plot_path}")
        except ImportError:
            print("  [WARN] matplotlib not installed, skipping plot")

    print("\n  Done.")


if __name__ == "__main__":
    main()
