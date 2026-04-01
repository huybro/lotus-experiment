base='..'
project='palimpzest'
mkdir -p logs
python3 -u ${base}/arxiv_filter.py > logs/${project}_arxiv_filter.log
python3 -u ${base}/arxiv_filter_filter.py > logs/${project}_arxiv_filter_filter.log
python3 -u ${base}/arxiv_filter_filter_filter.py > logs/${project}_arxiv_filter_filter_filter.log
python3 -u ${base}/arxiv_filter_filter_filter_map.py > logs/${project}_arxiv_filter_filter_filter_map.log