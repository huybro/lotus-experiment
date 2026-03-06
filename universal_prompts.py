import json


def get_prompt(instruction, data, data2=None, op='sem_filter'):
    """Build a standardized prompt for any semantic operation."""

    messages = []
    system_prompt = (
        "You are a helpful assistant for executing semantic operators.\n"
        "You will be given data and an operation description.\n"
        "Apply the operation to the provided data exactly as specified and return only the required result.\n"
    )

    if system_prompt is not None:
        messages.append(
            {"role": "system", "type": "text", "content": system_prompt}
        )
    if op == 'sem_filter':
        operation = (
            "You will be presented with a context and a filter condition. Output TRUE if the context satisfies the filter condition, and FALSE otherwise.\n" + \
            "Remember, your answer must be TRUE or FALSE. Finish your response with a newline character\n" + \
            "Output TRUE or FALSE only.\n" + \
            f"Condition:{instruction}\n"
        )
    elif op == 'sem_map':
        operation = (
            "You  are presented with a context and a mapping instruction.\n"
            "Apply the instruction to the context and produce the mapped output.\n"
            "The output must strictly follow the instruction and contain no extra commentary.\n"
            f"Map Instruction:{instruction}\n"
        )
    elif op == 'sem_agg':
        operation = (
            "You are presented with multiple contexts.\n"
            "Aggregate them according to the aggregation instruction.\n"
            "The output must be a single aggregated result.\n"
            "Do not include explanations or commentary.\n"
            f"Instruction:{instruction}\n"
        )
    elif op == 'sem_join':
        operation = (
            "You are presented with two contexts.\n"
            "Determine whether the two contexts A, B together satisfy the condition.\n"
            "Remember, your answer must be TRUE or FALSE. Finish your response with a newline character\n" + \
            "The output must strictly follow the condition and contain no extra commentary.\n"
            f"Condition:{instruction}\n"
        )

    if data2 is not None:
        user_messages = [
            {
                "role": "user",
                "type": "text",
                "content": (
                    "CONTEXT_A:\n"
                    "  {\n"
                    f"    \"text\": {data}\n"
                    "  }\n"
                    "\n\n"
                    "CONTEXT_B:\n"
                    "  {\n"
                    f"    \"text\": {data2}\n"
                    "  }\n"
                    "\n\n"
                    "TASK:\n"
                    f"{operation}\n\n"
                    "ANSWER:\n"
                ),
            }
        ]
    else:
        user_messages = [
            {
                "role": "user",
                "type": "text",
                "content": (
                    "CONTEXT:\n"
                    "  {\n"
                    f"    \"text\": {data}\n"
                    "  }\n"
                    "\n\n"
                    "TASK:\n"
                    f"{operation}\n\n"
                    "ANSWER:\n"
                ),
            }
        ]

    messages.extend(user_messages)
    return messages


def lotus_df2text_row(row_dict, cols):
    """Replicate LOTUS's df2text format: [Column]: «value» for each column."""
    return "".join(f"[{col.capitalize()}]: «{row_dict[col]}»\n" for col in cols)


def install_prompt_overrides():
    """Monkey-patch LOTUS's filter_formatter and map_formatter to use get_prompt().
    """
    import lotus.templates.task_instructions as task_instr

    def custom_filter_formatter(model, multimodal_data, user_instruction, *args, **kwargs):
        """Replacement filter_formatter matching LOTUS's real signature.
        - multimodal_data: dict with "text" key (serialized row data)
        - user_instruction: raw instruction template (e.g., "{content}\n...")
        """
        if isinstance(multimodal_data, dict):
            data = multimodal_data.get("text", str(multimodal_data))
        else:
            data = str(multimodal_data)
        return get_prompt(user_instruction, data, op='sem_filter')

    def custom_map_formatter(model, multimodal_data, user_instruction, *args, **kwargs):
        """Replacement map_formatter matching LOTUS's real signature."""
        if isinstance(multimodal_data, dict):
            data = multimodal_data.get("text", str(multimodal_data))
        else:
            data = str(multimodal_data)
        return get_prompt(user_instruction, data, op='sem_map')

    task_instr.filter_formatter = custom_filter_formatter
    task_instr.map_formatter = custom_map_formatter


def install_pz_prompt_overrides():

    from palimpzest.constants import Model

    # Patch __init__ to accept hosted_vllm models not in PZ's known list
    _original_init = Model.__init__
    def _patched_init(self, value, *args, **kwargs):
        if isinstance(value, str) and 'hosted_vllm' in value:
            self._value_ = value
            self.model_id = value
            self.model_name = value
            return
        return _original_init(self, value, *args, **kwargs)
    Model.__init__ = _patched_init

    # Patch is_vllm_model to recognize hosted_vllm models
    _original_is_vllm = Model.is_vllm_model
    def _patched_is_vllm(self):
        if hasattr(self, 'value') and 'hosted_vllm' in str(self.value):
            return True
        return _original_is_vllm(self)
    Model.is_vllm_model = _patched_is_vllm
