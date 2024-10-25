# use similar files to extract and create 'graph_edges.csv' and
# 'graph_node_features.csv' from dataset dumps

import os
from collections import defaultdict

import networkx as nx
import pandas as pd

from utils import housekeeping_utils
from utils.data_utils import create_and_dump_graph_info_from_obj

PROJECT_ROOT, global_logger = housekeeping_utils.hk_init()

DS_NAME = "add_health_ds"
NODE_FEAT_FILENAME = "graph_node_features.csv"
EDGES_FILENAME = "graph_edges.csv"

# communities to analyze; chose those with over 4-500 students in them
TARGET_COMMS = [
    '6', '9', '13', '14', '15', '16', '17', '41', '46', '49', '50', '56', '58',
    '60', '62', '72', '76'
    '77', '78', '82', '86', '87', '91', '92', '93', '259', '268', '269', '271'
]
TARGET_COMM = '49'
TARGET_COMM_FILE = 'AddHealth/comm' + TARGET_COMM + '.paj'
RACE_FILE = 'AddHealth/comm' + TARGET_COMM + 'pajrace.csv'

FEAT_TO_GROUP = defaultdict(lambda: 0)
FEAT_TO_GROUP[1] = 1


def create_and_dump_graph_info(raw_root=PROJECT_ROOT + "/data/" + DS_NAME +
                               "/raw/"):
    features_file = os.path.join(PROJECT_ROOT + "/data/" + DS_NAME,
                                 NODE_FEAT_FILENAME)
    edges_file = os.path.join(PROJECT_ROOT + "/data/" + DS_NAME,
                              EDGES_FILENAME)
    local_logger = housekeeping_utils.get_local_logger(
        DS_NAME, PROJECT_ROOT + "/data/" + DS_NAME + "/metadata.txt")

    orig_G = nx.Graph(nx.read_pajek(os.path.join(raw_root, TARGET_COMM_FILE)))
    largest_cc = sorted(nx.connected_components(orig_G), key=len,
                        reverse=True)[0]
    largest_cc_G = orig_G.subgraph(largest_cc)

    # target_G_with_loop = nx.Graph(largest_cc_G)
    # target_G_with_loop.add_edges_from([(node, node)
    #    for node in target_G_with_loop.nodes()])

    race_info = pd.read_csv(os.path.join(raw_root, RACE_FILE), header=None)

    race_list = list(race_info[0])

    node_feats = {
        node: FEAT_TO_GROUP[race_list[i]]
        for i, node in enumerate(largest_cc_G.nodes())
    }

    create_and_dump_graph_info_from_obj(largest_cc_G,
                                        node_feats,
                                        dump_files=(features_file, edges_file),
                                        logger=local_logger)
