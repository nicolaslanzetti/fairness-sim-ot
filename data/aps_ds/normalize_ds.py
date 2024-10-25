# use similar files to extract and create 'graph_edges.csv' and
# 'graph_node_features.csv' from dataset dumps

import csv
import os

import networkx as nx

from utils import housekeeping_utils
from utils.graph_utils import log_graph_properties

PROJECT_ROOT, global_logger = housekeeping_utils.hk_init()

DS_NAME = "aps_ds"
RAW_GRAPH_FILENAME = "sampled_APS_pacs052030-dir.gexf"
NODE_FEAT_FILENAME = "graph_node_features.csv"
EDGES_FILENAME = "graph_edges.csv"

FEAT_TO_GROUP = {"05.30.-d": 0, "05.20.-y": 1}


# unused num_truncated_nodes for now as the largest cc is of
# decent size
def create_and_dump_graph_info(num_truncated_nodes=50000,
                               raw_root=PROJECT_ROOT + "/data/" + DS_NAME +
                               "/raw/"):
    raw_graph_file = os.path.join(raw_root, RAW_GRAPH_FILENAME)
    G_orig = nx.read_gexf(raw_graph_file)
    Gcc = max(nx.weakly_connected_components(G_orig), key=len)
    G = nx.subgraph(G_orig, Gcc)

    local_logger = housekeeping_utils.get_local_logger(
        DS_NAME, PROJECT_ROOT + "/data/" + DS_NAME + "/metadata.txt")
    log_graph_properties(G.to_undirected(), logger=local_logger)

    features_file = os.path.join(PROJECT_ROOT + "/data/" + DS_NAME,
                                 NODE_FEAT_FILENAME)
    edges_file = os.path.join(PROJECT_ROOT + "/data/" + DS_NAME,
                              EDGES_FILENAME)

    with open(features_file,
              'w') as node_writefile, open(edges_file, 'w') as edg_writefile:
        node_writer = csv.writer(node_writefile,
                                 delimiter=',',
                                 quotechar='"',
                                 quoting=csv.QUOTE_MINIMAL)
        edg_writer = csv.writer(edg_writefile,
                                delimiter=',',
                                quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)

        for node in G.nodes():
            node_writer.writerow([node, FEAT_TO_GROUP[G.nodes[node]['pacs']]])

        for edge in G.edges():
            edg_writer.writerow([str(edge[0]) + " " + str(edge[1])])
