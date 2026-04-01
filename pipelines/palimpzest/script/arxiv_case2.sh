base='..'
project='palimpzest'
mkdir -p logs
python3 -u ${base}/arxiv_case_2_filter.py > logs/${project}_arxiv_case_2_filter.log
python3 -u ${base}/arxiv_case_2_filter_join.py > logs/${project}_arxiv_case_2_filter_join.log
python3 -u ${base}/arxiv_case_2_filter_join_map.py > logs/${project}_arxiv_case_2_filter_join_map.log
