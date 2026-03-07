import sys

# Add lotus source AFTER stdlib init
sys.path.append("/home/hojaeson_umass_edu/project/vllm-test/ref/lotus")

from lotus.models import LM
import lotus 
from pathlib import Path
import pandas as pd
import time


import json

def get_prompt(instruction, data, data2=None, op='sem_filter'):
    

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


lm = LM(
    model='hosted_vllm/meta-llama/Llama-3.1-8B-Instruct',
    api_base='http://localhost:8003/v1',
    max_ctx_len=8000,
    max_tokens=256
)

lotus.settings.configure(lm=lm)


def load_txt_folder_to_df(folder_path: str) -> pd.DataFrame:
    texts = []

    for p in sorted(Path(folder_path).glob("*.txt"), key=lambda x: int(x.stem)):
        with p.open("r", encoding="utf-8") as f:
            texts.append(f.read().strip())

    return pd.DataFrame({"Abstract": texts})

data = load_txt_folder_to_df("/home/hojaeson_umass_edu/.cache/kagglehub/datasets/spsayakpaul/arxiv-paper-abstracts/versions/2/arxiv_txt")
    
df = pd.DataFrame(data)


t_start = time.perf_counter()

t0 = time.perf_counter()
user_instruction = "{Abstract:left} Is the research paper related to the given category {Category:right}?"
df2 = pd.DataFrame({"Category": ['ai', 'biology', 'chemistry', 'geology', 'math', 'phyics']})

df = df.sem_join(df2, user_instruction)
t1 = time.perf_counter()
print(f"sem_join took {t1 - t0:.4f} seconds")

user_instruction = "Summarize the research abstract and explain how it is related to the category."


map_instruction = "You are presented with a context and a mapping instruction.\n" + \
                   "Apply the instruction to the context and produce the mapped output.\n" + \
                   "The output must strictly follow the instruction and contain no extra commentary.\n" + \
                   f"Map Instruction:{user_instruction}\n"


df["document"] = df.apply(
    lambda row: [
            "CONTEXT_A:\n"
            "  {\n"
            f"    \"text\": {json.dumps(row['Abstract'])}\n"
            "  }\n"
            "\n\n"
            "CONTEXT_B:\n"
            "  {\n"
            f"    \"text\": {json.dumps(row['Category'])}\n"
            "  }\n"
            "\n\n"
            "TASK:\n"
            f"{map_instruction}\n\n"
            "ANSWER:\n"
    ],
    axis=1,
)



t0 = time.perf_counter()
out = df.sem_map("{document}")
t1 = time.perf_counter()
print(f"sem_map took {t1 - t0:.4f} seconds")

t_end = time.perf_counter()
print(f"Total time taken: {t_end - t_start:.4f} seconds")

print(out)



