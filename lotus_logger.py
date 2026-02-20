"""
LOTUS Request Logger

A reusable module that logs every LLM call made by LOTUS operators to CSV.
Produces one CSV per experiment (a specific pipeline of operators).

Usage:
    from lotus_logger import LotusLogger

    # Each experiment gets its own CSV
    logger = LotusLogger(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        dataset_name="fever",
        experiment_name="filter_map_join",   # describes the pipeline
    )
    logger.install()

    # ... run your LOTUS pipeline (filter → map → join) ...

    logger.summary()
    # CSV saved to: logs/fever__qwen2.5-1.5b__filter_map_join.csv

    # For another experiment, create a new logger:
    logger2 = LotusLogger(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        dataset_name="fever",
        experiment_name="filter_topk_join",
    )
    logger2.install()
    # ... run a different pipeline ...
    # CSV saved to: logs/fever__qwen2.5-1.5b__filter_topk_join.csv
"""

import csv
import os
import time
from functools import wraps


class LotusLogger:
    """Logs every LOTUS LLM call per tuple to CSV. One CSV per experiment."""

    # Map LOTUS progress_bar_desc → clean operator name
    OPERATOR_MAP = {
        "Mapping": "map",
        "Filtering": "filter",
        "Aggregating": "agg",
        "Join comparisons": "join",
        "Heap comparisons": "topk",
        "Mapping examples": "join",
        "Running helper LM": "filter",
        "Running oracle for threshold learning": "filter",
        "Running predicate evals with oracle LM": "filter",
    }

    # Default CSV columns
    DEFAULT_COLUMNS = [
        "request_id", "operator", "input", "output",
        "input_token_len", "output_token_len",
    ]

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        experiment_name: str = "default",
        output_dir: str = "logs",
        output_path: str | None = None,
        columns: list[str] | None = None,
        debug: bool = False,
        debug_max_chars: int = 500,
    ):
        """
        Args:
            model_name:       Name of the LLM (e.g. "Qwen/Qwen2.5-1.5B-Instruct")
            dataset_name:     Name of the dataset (e.g. "fever")
            experiment_name:  Pipeline description (e.g. "filter_map_join")
            output_dir:       Directory for CSV files (default: "logs/")
            output_path:      Override full CSV path (ignores output_dir + auto-naming)
            columns:          CSV columns (default: request_id, operator, input, output, tokens)
            debug:            Print prompts/responses to console
            debug_max_chars:  Truncate console output (0 = no limit)
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.columns = columns or self.DEFAULT_COLUMNS
        self.debug = debug
        self.debug_max_chars = debug_max_chars

        # Auto-generate CSV path: logs/fever__qwen2.5-1.5b__filter_map_join.csv
        if output_path:
            self.output_path = output_path
        else:
            short_model = model_name.split("/")[-1].lower()
            filename = f"{dataset_name}__{short_model}__{experiment_name}.csv"
            os.makedirs(output_dir, exist_ok=True)
            self.output_path = os.path.join(output_dir, filename)

        self._request_counter = 0
        self._operator_stats = {}
        self._installed = False
        self._tokenizer = None  # lazy-loaded

    def install(self):
        """Monkey-patch LM.__call__ to intercept all LOTUS LLM calls."""
        if self._installed:
            return

        from lotus.models import LM
        original_call = LM.__call__
        logger = self  # capture for closure

        # Write CSV header
        with open(self.output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)

        @wraps(original_call)
        def patched_call(lm_self, messages, *args, **kwargs):
            desc = kwargs.get("progress_bar_desc", "")
            operator = logger.OPERATOR_MAP.get(desc, desc if desc else "unknown")

            # Console debug
            if logger.debug:
                logger._print_batch(operator, lm_self.model, messages)

            # Time the actual call
            t0 = time.time()
            result = original_call(lm_self, messages, *args, **kwargs)
            elapsed = time.time() - t0

            # Extract outputs
            outputs = result.outputs if hasattr(result, "outputs") and result.outputs else []

            # Log each tuple to CSV
            with open(logger.output_path, "a", newline="") as f:
                writer = csv.writer(f)
                for i, msg_list in enumerate(messages):
                    logger._request_counter += 1
                    full_input = logger._messages_to_text(msg_list)
                    output_text = outputs[i].strip() if i < len(outputs) else ""
                    input_tokens = logger._count_tokens(full_input)
                    output_tokens = logger._count_tokens(output_text)

                    # Build row based on configured columns
                    row_data = {
                        "request_id": logger._request_counter,
                        "operator": operator,
                        "input": full_input,
                        "output": output_text,
                        "input_token_len": input_tokens,
                        "output_token_len": output_tokens,
                        "batch_time_sec": f"{elapsed:.3f}",
                        "model": logger.model_name,
                        "dataset": logger.dataset_name,
                        "experiment": logger.experiment_name,
                    }
                    writer.writerow([row_data.get(col, "") for col in logger.columns])

            # Update stats
            if operator not in logger._operator_stats:
                logger._operator_stats[operator] = {
                    "count": 0, "total_time": 0.0,
                    "input_tokens": 0, "output_tokens": 0,
                }
            stats = logger._operator_stats[operator]
            stats["count"] += len(messages)
            stats["total_time"] += elapsed
            for i, msg_list in enumerate(messages):
                stats["input_tokens"] += logger._count_tokens(logger._messages_to_text(msg_list))
                stats["output_tokens"] += logger._count_tokens(
                    outputs[i].strip() if i < len(outputs) else ""
                )

            # Console debug: responses
            if logger.debug and outputs:
                logger._print_responses(outputs)

            print(f"  >> Logged {len(messages)} rows to {logger.output_path} "
                  f"(op={operator}, time={elapsed:.2f}s)")

            return result

        LM.__call__ = patched_call
        self._installed = True

    def summary(self):
        """Print a summary of all logged requests."""
        print(f"\n{'='*60}")
        print(f"  LOTUS Logger Summary")
        print(f"  Model:      {self.model_name}")
        print(f"  Dataset:    {self.dataset_name}")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  CSV:        {self.output_path}")
        print(f"  Total requests: {self._request_counter}")
        print(f"{'='*60}")

        if self._operator_stats:
            print(f"  {'Operator':<12} {'Count':>6} {'Time (s)':>10} {'In Tok':>8} {'Out Tok':>8}")
            print(f"  {'─'*48}")
            for op, s in sorted(self._operator_stats.items()):
                print(f"  {op:<12} {s['count']:>6} {s['total_time']:>10.2f} "
                      f"{s['input_tokens']:>8} {s['output_tokens']:>8}")
        print()

    # ── Internal helpers ──────────────────────────────────────

    @staticmethod
    def _extract_text(content):
        """Extract plain text from a message content field."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item["text"])
            return "\n".join(parts)
        return str(content)

    @classmethod
    def _messages_to_text(cls, msg_list):
        """Convert a single message list to a flat text string."""
        parts = []
        for part in msg_list:
            role = part.get("role", "?")
            text = cls._extract_text(part.get("content", ""))
            parts.append(f"[{role}] {text}")
        return "\n".join(parts)

    def _get_tokenizer(self):
        """Lazy-load tokenizer from model name."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                print(f"  [logger] Loaded tokenizer for {self.model_name}")
            except Exception as e:
                print(f"  [logger] Could not load tokenizer for {self.model_name}: {e}")
                print(f"  [logger] Falling back to whitespace split")
                self._tokenizer = "fallback"
        return self._tokenizer

    def _count_tokens(self, text):
        """Count tokens using the model's tokenizer."""
        if not text:
            return 0
        tokenizer = self._get_tokenizer()
        if tokenizer == "fallback":
            return len(text.split())
        return len(tokenizer.encode(text, add_special_tokens=False))

    def _print_batch(self, operator, model, messages):
        """Print batch info to console."""
        n = len(messages)
        print(f"\n{'─'*70}")
        print(f"[{operator}] Batch ({n} message{'s' if n != 1 else ''} to {model})")
        print(f"{'─'*70}")
        for i, msg_list in enumerate(messages):
            if n > 3 and 2 <= i < n - 1:
                if i == 2:
                    print(f"  ... ({n - 3} more) ...")
                continue
            print(f"  -- Message [{i}] --")
            for part in msg_list:
                role = part.get("role", "?")
                text = self._extract_text(part.get("content", ""))
                display = text
                if self.debug_max_chars and len(display) > self.debug_max_chars:
                    display = display[:self.debug_max_chars] + f" ... [{len(text)} chars]"
                print(f"    [{role}]")
                for line in display.split("\n"):
                    print(f"      {line}")
        print(f"{'─'*70}")

    def _print_responses(self, outputs):
        """Print responses to console."""
        print(f"  Responses ({len(outputs)}):")
        for i, out in enumerate(outputs):
            if len(outputs) > 5 and 3 <= i < len(outputs) - 1:
                if i == 3:
                    print(f"    ... ({len(outputs) - 4} more) ...")
                continue
            display = out.strip()
            if self.debug_max_chars and len(display) > self.debug_max_chars:
                display = display[:self.debug_max_chars] + "..."
            print(f"    [{i}] {display}")
        print()
    