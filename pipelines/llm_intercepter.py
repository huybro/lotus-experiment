import litellm

_original_completion = None


def set_intercept(**params):
    global _original_completion
    if _original_completion is None:
        _original_completion = litellm.completion
    log = params['log']
    MAX_TOKENS = params['max_tokens']
    tokenizer = params['tokenizer']

    def _interceptor(*args, **kwargs):
        kwargs.setdefault("max_tokens", MAX_TOKENS)
        kwargs.setdefault("temperature", 0)
        kwargs.setdefault("seed", 42)

        messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        result = _original_completion(*args, **kwargs)
        output_text = result.choices[0].message.content if result.choices else ""

        log.append({"input": prompt_text, "output": output_text})

        return result

    litellm.completion = _interceptor
