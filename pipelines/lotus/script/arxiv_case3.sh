base='..'
mkdir -p logs
python3 ${base}/arxiv_filter.py > logs/lotus_arxiv_filter.log
python3 ${base}/arxiv_filter_filter.py > logs/lotus_arxiv_filter_filter.log
python3 ${base}/arxiv_filter_filter_filter.py > logs/lotus_arxiv_filter_filter_filter.log
python3 ${base}/arxiv_filter_filter_filter_map.py > logs/lotus_arxiv_filter_filter_filter_map.log