import palimpzest as pz
from palimpzest.core.lib.schemas import TextFile
from palimpzest.constants import Model


if __name__ == "__main__":
    # create validator and train_dataset 


    # construct plan
    plan = pz.TextFileDataset(id="enron", path="/home/hojaeson_umass_edu/.cache/kagglehub/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/versions/1/imdb_sample_500_texts")
    
    # filter - filter
    plan = plan.sem_filter(
        "The review contains substantive content, meaning it is not short (less than three sentences) or vague and expresses a concrete opinion about the movie",
        depends_on=["contents"],
    )

    plan = plan.sem_filter(
        "The review criticizes the movie’s plot, storytelling, or narrative structure, such as issues with pacing, coherence, or resolution",
        depends_on=["contents"],
    )

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
    data_record_collection = plan.run(config)
    print()
