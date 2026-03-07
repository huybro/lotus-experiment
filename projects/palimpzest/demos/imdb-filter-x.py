import json
import os

import palimpzest as pz
from palimpzest.core.lib.schemas import TextFile
from palimpzest.constants import Model


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
        self.filepaths = self.filepaths[:50] if split == "train" else self.filepaths[50:150]
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
    # create validator and train_dataset 


    # construct plan
    # plan = EnronDataset(dir="testdata/enron-eval-medium", split="test")
    plan = pz.TextFileDataset(id="enron", path="/home/hojaeson_umass_edu/.cache/kagglehub/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/versions/1/imdb_sample_500_texts")
    # # map - filter -filter
    # plan = plan.sem_map([
    #     {"name": "summary", "type": str, "desc": "Summarize the review"},
    # ])
    
    # plan = plan.sem_filter(
    #     "The review contains substantive content, meaning it is not short or vague and expresses a concrete opinion about the movie",
    #     depends_on=["contents"],
    # )

    # plan = plan.sem_filter(
    #     "The review criticizes the movie’s plot, storytelling, or narrative structure, such as issues with pacing, coherence, or resolution",
    #     depends_on=["contents"],
    # )
    
    # # filter - filter - map
    # plan = plan.sem_filter(
    #     "The review contains substantive content, meaning it is not short (less than three sentences) or vague and expresses a concrete opinion about the movie",
    #     depends_on=["contents"],
    # )

    # plan = plan.sem_filter(
    #     "The review criticizes the movie’s plot, storytelling, or narrative structure, such as issues with pacing, coherence, or resolution",
    #     depends_on=["contents"],
    # )
    
    # plan = plan.sem_map([
    #     {"name": "summary", "type": str, "desc": "Summarize the review"},
    # ])
            
    #TODO IT doesnt work
    # plan = plan.sem_agg(
    #     col={'name': 'overall_sentiment', 'type': str, 'desc': 'The top-3 most common complaints mentioned in the reviews'},
    #     agg="Compute the top-3 most common reviews mentioned in the reviews",
    #     depends_on=["contents"],
    # )
    # Sem join
    
    # paper = pz.TextFileDataset(id="abstract", path="/home/hojaeson_umass_edu/.cache/kagglehub/datasets/spsayakpaul/arxiv-paper-abstracts/versions/2/arxiv_txt") 
    # cat = pz.TextFileDataset(id="category", path="/home/hojaeson_umass_edu/.cache/kagglehub/datasets/spsayakpaul/arxiv-paper-abstracts/versions/2/category")
    # paper = paper.sem_join(cat, condition="Is the research paper related to the given category?")
    # paper = paper.sem_map([
    #     {"name": "summary", "type": str, "desc": "Summarize the research abstract and explain how it is related to the category"},
    # ])


    # plan = plan.sem_join(plan2, condition="Are two contexts relevant?")
    # Sem Topk
    # Sem group by - it doesnt't support
    
    

    # sem join, sem topk, sem groupby
    # plan = plan.sem_map([
    #     {'name': 'summary', 'desc': 'Summarize the review', 'type': str},
    # ])
    

    # execute pz plan
    config = pz.QueryProcessorConfig(
        api_base="http://localhost:8003/v1",
        available_models=[Model.VLLM_LLAMA3_2_3B],
        # available_models=[Model.VLLM_LLAMA3_1_8B],
        verbose=False,
        policy=pz.MinTime(),
        # policy=pz.MaxQuality(),
        execution_strategy='sequential',
        # execution_strategy='parallel',
    )
    # config = pz.QueryProcessorConfig(
        
    #     api_base="http://localhost:8003/v1",
    #     available_models=[Model.VLLM_LLAMA3_2_3B],
    #     allow_model_selection=False, 
    #     policy=pz.MaxQuality(),
    #     execution_strategy="parallel",
    #     k=5,
    #     j=6,
    #     sample_budget=100,
    #     max_workers=20,
    #     progress=True,
    # )
    data_record_collection = plan.run(config)
    print()
    # output = plan.optimize_and_run(train_dataset=train_dataset, validator=validator, config=config)

    # print output dataframe 

    # print precision and recall
    # with open("testdata/enron-eval-medium-labels.json") as f:
    #     filename_to_labels = json.load(f)
    #     test_filenames = os.listdir("testdata/enron-eval-medium")[50:150]
    #     filename_to_labels = {k: v for k, v in filename_to_labels.items() if k in test_filenames}

    # target_filenames = set(filename for filename, labels in filename_to_labels.items() if labels != [])
    # pred_filenames = set(output.to_df()["filename"])
    # tp = sum(filename in target_filenames for filename in pred_filenames)
    # fp = len(pred_filenames) - tp
    # fn = len(target_filenames) - tp

    # print(f"PRECISION: {tp/(tp + fp) if tp + fp > 0 else 0.0:.3f}")
    # print(f"RECALL: {tp/(tp + fn) if tp + fn > 0 else 0.0:.3f}")
