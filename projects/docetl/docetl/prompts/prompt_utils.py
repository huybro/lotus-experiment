from __future__ import annotations

from typing import Any


class SemanticOp:
    FILTER = "sem_filter"
    JOIN = "sem_join"
    CLASSIFY = "sem_classify"
    MAP = "sem_map"
    AGG = "sem_agg"


SYSTEM_PROMPT = (
    "You are a helpful assistant for executing semantic operators.\n"
    "You will be given data and an operation description.\n"
    "Apply the operation to the provided data exactly as specified and return only the required result.\n"
)


def _build_operation_prompt(
    instruction: str, op: str = SemanticOp.FILTER
) -> str:
    instruction = instruction.strip()
    if op == SemanticOp.FILTER:
        return (
            "You will be presented with a context and a filter condition. Output TRUE if the context satisfies the filter condition, and FALSE otherwise.\n"
            "Remember, your answer must be TRUE or FALSE. Finish your response with a newline character\n"
            "Output TRUE or FALSE only.\n"
            f"Condition:{instruction}\n"
        )
    if op == SemanticOp.MAP:
        return (
            "You  are presented with a context and a mapping instruction.\n"
            "Apply the instruction to the context and produce the mapped output.\n"
            "The output must strictly follow the instruction and contain no extra commentary.\n"
            f"Map Instruction:{instruction}\n"
        )
    if op == SemanticOp.AGG:
        return (
            "You are presented with multiple contexts.\n"
            "Aggregate them according to the aggregation instruction.\n"
            "The output must be a single aggregated result.\n"
            "Do not include explanations or commentary.\n"
            f"Instruction:{instruction}\n"
        )
    if op == SemanticOp.JOIN:
        return (
            "You are presented with two contexts.\n"
            "Determine whether the two contexts A, B together satisfy the condition.\n"
            "Remember, your answer must be TRUE or FALSE. Finish your response with a newline character\n"
            "The output must strictly follow the condition and contain no extra commentary.\n"
            f"Condition:{instruction}\n"
        )
    if op in (SemanticOp.CLASSIFY,):
        return (
            "You are presented with a context and a classification instruction.\n"
            "Classify the context into exactly one of the provided groups.\n"
            "The output must be one group name only.\n"
            "Do not include explanations or extra text.\n"
            f"Instruction:{instruction}\n"
        )

    raise ValueError(f"Unsupported semantic operation: {op}")


def get_system_prompt() -> list[dict[str, str]]:
    messages = [{"role": "system", "type": "text", "content": SYSTEM_PROMPT}]
    return messages


def get_data_prompt(data: Any, right_data: list[Any] | None = None) -> list[dict[str, str]]:
    if right_data:
        context = (
            "CONTEXT:\n"
            "  {\n"
            f"    \"text\": {data}\n"
            "  }\n"
        )
        for item in right_data:
            context += (
                "\n\n"
                f"CONTEXT:\n"
                "  {\n"
                f"    \"text\": {item}\n"
                "  }\n"
            )
    else:
        context = (
            "CONTEXT:\n"
            "  {\n"
            f"    \"text\": {data}\n"
            "  }\n"
        )

    return [{"role": "user", "type": "text", "content": context}]


def get_task_prompt(instruction: str, op: str = SemanticOp.FILTER) -> list[dict[str, str]]:
    operation = _build_operation_prompt(instruction, op=op)
    return [
        {
            "role": "user",
            "type": "text",
            "content": (
                "TASK:\n"
                f"{operation}\n\n"
                "ANSWER:\n"
            ),
        }
    ]


def get_prompt(
    instruction: str,
    data: list[dict[str, Any]],
    op: str = SemanticOp.FILTER,
) -> list[dict[str, Any]]:
    """Merge system prompt, CONTEXT user message(s), and TASK user message(s)."""
    if "system" in data[0]["role"]:
        return data + get_task_prompt(instruction=instruction, op=op)
    return get_system_prompt() + data + get_task_prompt(instruction=instruction, op=op)


def context_text_from_item(
    item: dict[str, Any],
    columns: list[str] | None,
) -> str:
    r"""
    Build CONTEXT body text as ``"".join(str(item[c]) for c in columns)``.

    Pass ``semantic_context_columns`` in pipeline config in the desired column order.
    If omitted, uses ``sorted(item.keys())``.
    """
    if columns is None:
        columns = sorted(item.keys())
    return "".join(str(item[c]) for c in columns if c in item)


def build_semantic_prompt_messages(
    instruction: str,
    context_text: Any,
    op: str,
) -> list[dict[str, str]]:
    """Full message list: ``get_prompt(instruction, get_data_prompt(context_text), op)``."""
    data_prompt = get_data_prompt(context_text)
    return get_prompt(instruction, data_prompt, op=op)


def add_assistant_prompt(prompt, output_text):
    output_template = [{
        "role": "assistant",
        "content": output_text
    }]
    appended_prompt = prompt + output_template
    return appended_prompt