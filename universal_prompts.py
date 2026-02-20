"""
Universal Prompts for LOTUS operators.
"""

# Shared system prompt — same for all operators
SYSTEM_PROMPT = (
    "You are a precise data processing assistant. "
    "Follow the user's instructions exactly and respond concisely."
)

# Per-operator instruction templates
# {placeholders} are filled in with dataframe column values by LOTUS

OPERATOR_PROMPTS = {
    "filter": {
        "system": SYSTEM_PROMPT,
        "description": "Determine if a claim/condition is true given context.",
        "instruction": (
            "Context:\n{context}\n\n"
            "Claim: {claim}\n\n"
            "Is the claim supported by the context? "
            "Respond with exactly one word: True or False."
        ),
    },
    "map": {
        "system": SYSTEM_PROMPT,
        "description": "Transform or extract information from data.",
        "instruction": (
            "Data:\n{data}\n\n"
            "Instruction: {instruction}\n\n"
            "Provide a concise answer based only on the data."
        ),
    },
    "agg": {
        "system": SYSTEM_PROMPT,
        "description": "Summarize or aggregate multiple data items.",
        "instruction": (
            "Data items:\n{data}\n\n"
            "Instruction: {instruction}\n\n"
            "Provide a concise summary."
        ),
    },
    "join": {
        "system": SYSTEM_PROMPT,
        "description": "Determine if two items are related.",
        "instruction": (
            "Item A: {left}\n"
            "Item B: {right}\n\n"
            "Condition: {condition}\n\n"
            "Are these items related per the condition? "
            "Respond with exactly one word: True or False."
        ),
    },
    "topk": {
        "system": SYSTEM_PROMPT,
        "description": "Compare items to select the best one.",
        "instruction": (
            "Item A: {item_a}\n"
            "Item B: {item_b}\n\n"
            "Criteria: {criteria}\n\n"
            "Which item is better? Respond with exactly: A or B."
        ),
    },
}


def get_prompt(operator: str, **kwargs) -> dict:
    """
    Get formatted system + user messages for an operator.

    Args:
        operator: One of 'filter', 'map', 'agg', 'join', 'topk'
        **kwargs: Template values (e.g., context, claim, data, instruction)

    Returns:
        dict with 'system' and 'user' keys containing the formatted messages

    Example:
        >>> prompt = get_prompt("filter", context="...", claim="...")
        >>> prompt["system"]
        "You are a precise data processing assistant..."
        >>> prompt["user"]
        "Context: ...\\nClaim: ...\\nIs the claim supported..."
    """
    if operator not in OPERATOR_PROMPTS:
        raise ValueError(f"Unknown operator '{operator}'. Valid: {list(OPERATOR_PROMPTS.keys())}")

    config = OPERATOR_PROMPTS[operator]
    return {
        "system": config["system"],
        "user": config["instruction"].format(**kwargs),
    }
