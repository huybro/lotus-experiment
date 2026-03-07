#!/bin/bash
#
#
#
module load conda/latest
conda activate /home/hojaeson_umass_edu/hojae_workspace/miniconda3/envs/py312

conda activate py312
export PYTHONPATH=$HOME/project/lotus:$PYTHONPATH
