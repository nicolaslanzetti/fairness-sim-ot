# use similar files to extract and create 'graph_edges.csv' and
# 'graph_node_features.csv' from dataset dumps

import os
import pickle

from utils import housekeeping_utils
from utils.data_utils import create_and_dump_graph_info_from_obj
from utils.graph_utils import log_graph_properties

PROJECT_ROOT, global_logger = housekeeping_utils.hk_init()

DS_NAME = "antelope_valley_ds"
# DS_IDX = 22  # 0 to 23
RAW_GRAPH_PREFIX = "graph_spa_500_"  # typed "*.pickle"
NODE_FEAT_FILENAME = "graph_node_features.csv"
EDGES_FILENAME = "graph_edges.csv"

FEAT_KEY = "gender"
FEAT_TO_GROUP = {"female": 0, "male": 1}


def create_and_dump_graph_info(ds_idx,
                               raw_root=PROJECT_ROOT + "/data/" + DS_NAME +
                               "/raw/"):
    raw_graph_file = os.path.join(raw_root,
                                  RAW_GRAPH_PREFIX + str(ds_idx) + ".pickle")
    raw_G = pickle.load(open(raw_graph_file, 'rb'))
    G = raw_G.to_undirected().copy()

    featless_nodes = []
    node_feats = {
        node_id: FEAT_TO_GROUP[raw_G.nodes[node_id][FEAT_KEY]]
        for node_id in raw_G.nodes if FEAT_KEY in raw_G.nodes[node_id].keys()
        or not featless_nodes.append(node_id)
    }

    G.remove_nodes_from(featless_nodes)

    features_file = os.path.join(PROJECT_ROOT + "/data/" + DS_NAME,
                                 NODE_FEAT_FILENAME)
    edges_file = os.path.join(PROJECT_ROOT + "/data/" + DS_NAME,
                              EDGES_FILENAME)
    local_logger = housekeeping_utils.get_local_logger(
        DS_NAME, PROJECT_ROOT + "/data/" + DS_NAME + "/metadata.txt")

    log_graph_properties(G, logger=local_logger)

    create_and_dump_graph_info_from_obj(G,
                                        node_feats,
                                        (features_file, edges_file),
                                        logger=local_logger)

    # return G
