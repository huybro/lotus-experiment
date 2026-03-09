import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/..')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/../palimpzest/src")
import json

import palimpzest as pz
from palimpzest.core.lib.schemas import TextFile
# python3 demos/enron-demo.py 

class EnronValidator(pz.Validator):
    def __init__(self, labels_file: str):
        super().__init__()

        self.filename_to_labels = {}
        if labels_file:
            with open(labels_file) as f:
                self.filename_to_labels = json.load(f)

    def map_score_fn(self, fields: list[str], input_record: dict, output: dict) -> float | None:
        filename = input_record["filename"]
        labels = self.filename_to_labels[filename]
        if len(labels) == 0:
            return None

        labels = labels[0]
        return (float(labels["sender"] == output["sender"]) + float(labels["subject"] == output["subject"])) / 2.0


class EnronDataset(pz.IterDataset):
    def __init__(self, dir: str, labels_file: str | None = None, split: str = "test"):
        super().__init__(id="enron", schema=TextFile)
        self.filepaths = [os.path.join(dir, filename) for filename in os.listdir(dir)]
        self.filepaths = self.filepaths[:50] if split == "train" else self.filepaths
        self.filename_to_labels = {}
        if labels_file:
            with open(labels_file) as f:
                self.filename_to_labels = json.load(f)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        # get input fields
        filepath = self.filepaths[idx]
        filename = os.path.basename(filepath)
        with open(filepath) as f:
            contents = f.read()

        # create item with fields
        item = {"filename": filename, "contents": contents}

        return item


if __name__ == "__main__":
        
    from palimpzest.constants import Model
    from palimpzest.query.processor.config import QueryProcessorConfig
    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    MAX_TOKENS = 512
    VLLM_API_BASE = "http://localhost:8003/v1"
    PZ_MODEL = Model(f"hosted_vllm/{MODEL_NAME}")
    PZ_MODEL.api_base = VLLM_API_BASE
    pz_config = QueryProcessorConfig(
        policy=pz.MaxQuality(),
        execution_strategy="parallel",
        api_base=VLLM_API_BASE,
        available_models=[PZ_MODEL],
        allow_model_selection=False,
        allow_bonded_query=False,  # Use direct LLM (LLMConvertBonded), not RAG
        allow_rag_reduction=False,  # Disable RAG (needs OpenAI embeddings)
        allow_mixtures=False,
        allow_critic=False,
        allow_split_merge=False,
        sample_budget=100,
        max_workers=20,
        progress=True,
        verbose=True,
    )


    # create validator and train_dataset
    validator = EnronValidator(labels_file="testdata/enron-eval-medium-labels.json")
    train_dataset = EnronDataset(dir="testdata/enron-eval", split="train")

    # construct plan
    plan = EnronDataset(dir="testdata/enron-eval", split="test")
    plan = plan.sem_map([
        {"name": "subject", "type": str, "desc": "The subject of the email"},
        {"name": "sender", "type": str, "desc": "The email address of the email's sender"},
    ])
    plan = plan.sem_filter(
        'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")',
        depends_on=["contents"],
    )
    plan = plan.sem_filter(
        "The email is not quoting from a news article or an article written by someone outside of Enron",
        depends_on=["contents"],
    )

    # execute pz plan
    config = pz.QueryProcessorConfig(
        api_base=VLLM_API_BASE,
        available_models=[PZ_MODEL],
        policy=pz.MaxQuality(),
        execution_strategy="parallel",
        k=5,
        j=6,
        sample_budget=100,
        max_workers=20,
        # allow_bonded_query=False,  # Use direct LLM (LLMConvertBonded), not RAG
        allow_rag_reduction=False,  # Disable RAG (needs OpenAI embeddings)
        allow_model_selection=False,
        allow_mixtures=False,
        allow_critic=False,
        allow_split_merge=False,
        progress=True,
        verbose=True,
    )
    print('asdf')
    # output = plan.optimize_and_run(train_dataset=train_dataset, validator=validator, config=config)
    output = plan.optimize_and_run(train_dataset=train_dataset, config=config)
    print('asdf')

    # print output dataframe
    print(output.to_df())

    # print precision and recall
    with open("testdata/enron-eval-medium-labels.json") as f:
        filename_to_labels = json.load(f)
        test_filenames = os.listdir("testdata/enron-eval")
        filename_to_labels = {k: v for k, v in filename_to_labels.items() if k in test_filenames}

    target_filenames = set(filename for filename, labels in filename_to_labels.items() if labels != [])
    pred_filenames = set(output.to_df()["filename"])
    tp = sum(filename in target_filenames for filename in pred_filenames)
    fp = len(pred_filenames) - tp
    fn = len(target_filenames) - tp

    print(f"PRECISION: {tp/(tp + fp) if tp + fp > 0 else 0.0:.3f}")
    print(f"RECALL: {tp/(tp + fn) if tp + fn > 0 else 0.0:.3f}")
