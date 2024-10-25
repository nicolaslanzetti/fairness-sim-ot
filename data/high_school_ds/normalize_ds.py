# use similar files to extract and create 'graph_edges.csv' and
# 'graph_node_features.csv' from dataset dumps

import csv
import os

import networkx as nx

from utils import housekeeping_utils
from utils.data_utils import create_and_dump_graph_info_from_obj
from utils.graph_utils import log_graph_properties

PROJECT_ROOT, global_logger = housekeeping_utils.hk_init()

DS_NAME = "high_school_ds"
RAW_EDGES_FILENAME = "Friendship-network_data_2013.csv"
RAW_NODE_FEAT_FILENAME = "metadata_2013.txt"
NODE_FEAT_FILENAME = "graph_node_features.csv"
EDGES_FILENAME = "graph_edges.csv"

FEAT_TO_GROUP = {"F": 0, "M": 1}


# unused num_truncated_nodes for now as the largest cc is of
# decent size
def create_and_dump_graph_info(raw_root=PROJECT_ROOT + "/data/" + DS_NAME +
                               "/raw/"):
    raw_edges_file = os.path.join(raw_root, RAW_EDGES_FILENAME)
    G = nx.read_edgelist(raw_edges_file)

    local_logger = housekeeping_utils.get_local_logger(
        DS_NAME, PROJECT_ROOT + "/data/" + DS_NAME + "/metadata.txt")

    node_feats = {}
    raw_feats_file = os.path.join(raw_root, RAW_NODE_FEAT_FILENAME)
    feats_file = open(raw_feats_file)
    feats_reader = csv.reader(feats_file)

    for row in feats_reader:
        node, _, feat = row[0].split('\t')
        if feat == "Unknown":
            try:
                G.remove_node(node)
            except nx.NetworkXError:
                pass  # tried removing an unncessary node
        else:
            node_feats[node] = FEAT_TO_GROUP[feat]

    log_graph_properties(G, logger=local_logger)

    features_file = os.path.join(PROJECT_ROOT + "/data/" + DS_NAME,
                                 NODE_FEAT_FILENAME)
    edges_file = os.path.join(PROJECT_ROOT + "/data/" + DS_NAME,
                              EDGES_FILENAME)

    create_and_dump_graph_info_from_obj(G,
                                        node_feats,
                                        dump_files=(features_file, edges_file),
                                        logger=local_logger)
