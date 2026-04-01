SYSTEM_PROMPT = (
    "You are a helpful assistant for executing semantic operators.\n"
    "You will be given data and an operation description.\n"
    "Apply the operation to the provided data exactly as specified and return only the required result.\n"
)


def build_filter_messages(condition: str, context_text: str) -> list[dict]:
    """Build filter prompt messages in SYSTEM + CONTEXT + TASK format."""
    normalized_condition = condition.strip()
    context_prompt = (
        "CONTEXT:\n"
        "  {\n"
        f'    "text": {context_text}\n'
        "  }\n"
    )
    task_prompt = (
        "TASK:\n"
        "You will be presented with a context and a filter condition. "
        "Output TRUE if the context satisfies the filter condition, and FALSE otherwise.\n"
        "Remember, your answer must be TRUE or FALSE. Finish your response with a newline character\n"
        "Output TRUE or FALSE only.\n"
        f"Condition:{normalized_condition}\n\n"
        "ANSWER:\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context_prompt},
        {"role": "user", "content": task_prompt},
    ]
