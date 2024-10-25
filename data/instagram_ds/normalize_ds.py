# use similar files to extract and create 'graph_edges.csv' and
# 'graph_node_features.csv' from dataset dumps

import os

from utils import housekeeping_utils
from utils.data_utils import create_and_dump_graph_info_helper

PROJECT_ROOT, global_logger = housekeeping_utils.hk_init()

DS_NAME = "instagram_ds"
RAW_NODE_FEAT_FILENAME = "graph_gender_anon.csv"
RAW_EDGES_FILENAME = "graph_edges_anon.csv"
NODE_FEAT_FILENAME = "graph_node_features.csv"
EDGES_FILENAME = "graph_edges.csv"

FEAT_TO_GROUP = {"F": 0, "M": 1}


def create_and_dump_graph_info(num_truncated_nodes=50000,
                               raw_root=PROJECT_ROOT + "/data/" + DS_NAME +
                               "/raw/"):
    raw_features_file = os.path.join(raw_root, RAW_NODE_FEAT_FILENAME)
    raw_edges_file = os.path.join(raw_root, RAW_EDGES_FILENAME)
    features_file = os.path.join(PROJECT_ROOT + "/data/" + DS_NAME,
                                 NODE_FEAT_FILENAME)
    edges_file = os.path.join(PROJECT_ROOT + "/data/" + DS_NAME,
                              EDGES_FILENAME)
    raw_data_files = {"nodes": raw_features_file, "edges": raw_edges_file}
    new_data_files = {"nodes": features_file, "edges": edges_file}

    local_logger = housekeeping_utils.get_local_logger(
        DS_NAME, PROJECT_ROOT + "/data/" + DS_NAME + "/metadata.txt")
    create_and_dump_graph_info_helper(num_truncated_nodes,
                                      raw_data_files=raw_data_files,
                                      new_data_files=new_data_files,
                                      feat_to_grp_map=FEAT_TO_GROUP,
                                      logger=local_logger)
