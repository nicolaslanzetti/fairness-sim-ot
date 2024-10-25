#!/bin/bash

conda init
source /root/.bashrc
source /root/miniconda3/etc/profile.d/conda.sh && conda activate fairness-sim-ot-lab
jupyter notebook --ip=0.0.0.0 --allow-root --no-browser