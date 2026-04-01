
# FEVER
FEVER_FILTER = "{claim}{content}Given a claim and evidence, decide if the evidence is enough to determine whether the claim is true or false."
FEVER_MAP = "{claim}{content}Explain how the evidence supports or unsupports the claim."


# ENRON 1
FILTER_ENRON_FRAUD = (
    '{contents}The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'

)
FILTER_ENRON_NOT_NEWS = (
    '{contents}The email is not quoting from a news article or an article written by someone outside of Enron.'
)

MAP_ENRON_EXPLANATION = (
    '{contents}Explain briefly why this email is related to a fraudulent scheme, using the email contents provided in the context.'
)


FILTER_ENRON_FRAUD_2 = (
    '{contents}The email is normal scheme (i.e., Not "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'

)
FILTER_ENRON_NOT_NEWS_2 = (
    '{contents}The email includes people associated with Enron.'
)

MAP_ENRON_EXPLANATION_2 = (
    '{contents}Summarize the email briefly.'
)
# ARXIV

# TOPK - MAP
CASE_1_TOPK_ARXIV = ("{abstract} Provide the most relevant papers to Image Semantic Segmentation research")
CASE_1_MAP_ARXIV = (
    '{abstract}Explain why is the paper relevant to Image Semantic Segmentation research'
    )
CASE_1_AGG_ARXIV = (
    '{abstract}"Summarize the key common ideas and contributions across the paper abstracts"'
    )



# FILTER - JOIN - FILTER - MAP -

CASE_2_FILTER_ARXIV = ("{abstract}Is the paper relevant to AI robotics research?")
CASE_2_JOIN_ARXIV = (
    '{abstract}{robotic_abstract}Do these two papers study the same problem or topic?'
)
CASE_2_MAP_ARXIV = (
    '{abstract}Explain their relationship in terms of shared topics, methods, and differences.'
)

CASE_3_FILTER_1 = ("{abstract}Is this abstract about AI or machine learning?")
CASE_3_FILTER_2 = ("{abstract}Is this abstract describing theoretical work?")
CASE_3_FILTER_3 = ("{abstract}Does this abstract mention using an image dataset?")
CASE_3_MAP_ARXIV = ("{abstract}Summarize the abstract in 1–2 sentences, focusing on the main method and key result.")