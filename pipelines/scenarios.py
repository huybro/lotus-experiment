
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

