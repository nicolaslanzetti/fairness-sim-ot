# use similar files to extract and create 'graph_edges.csv' and
# 'graph_node_features.csv' from dataset dumps

import os

from utils import housekeeping_utils

PROJECT_ROOT, global_logger = housekeeping_utils.hk_init()


def create_and_dump_graph_info():
    curr_dir = os.getcwd()
    print("Creating graph info csv files from dumps at", curr_dir)

    pass  # stub
