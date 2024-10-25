import json
import os
import random

import networkx as nx
import pandas as pd

from utils import housekeeping_utils
from utils.data_utils import create_and_dump_graph_info_from_obj

PROJECT_ROOT, global_logger = housekeeping_utils.hk_init()

DS_NAME = "dblp_ds"
RAW_NODE_FEAT_FILENAME = "DBLP_v10_dictionary_authornames_to_gender_largestwcc.txt"
RAW_EDGES_FILENAME = "dblpv10_largestwcc_graph_edges_unique.csv"
NODE_FEAT_FILENAME = "graph_node_features.csv"
EDGES_FILENAME = "graph_edges.csv"

CHUNK_SIZE = 500000
FEAT_TO_GROUP = {"male": 0, "female": 1}


def convert_whitespaces_to(target_str, replace_char='_'):
    return target_str.replace(' ', replace_char)


def create_and_dump_graph_info(num_truncated_nodes=50000,
                               raw_root=PROJECT_ROOT + "/data/" + DS_NAME +
                               "/raw/"):
    raw_features_file = os.path.join(raw_root, RAW_NODE_FEAT_FILENAME)
    raw_edges_file = os.path.join(raw_root, RAW_EDGES_FILENAME)
    features_file = os.path.join(PROJECT_ROOT + "/data/" + DS_NAME,
                                 NODE_FEAT_FILENAME)
    edges_file = os.path.join(PROJECT_ROOT + "/data/" + DS_NAME,
                              EDGES_FILENAME)
    local_logger = housekeeping_utils.get_local_logger(
        DS_NAME, PROJECT_ROOT + "/data/" + DS_NAME + "/metadata.txt")

    with open(raw_features_file) as raw_featfile:
        data_featfile = raw_featfile.read()

    # reconstructing the data as a dictionary
    raw_node_feats = json.loads(data_featfile)
    node_entries = list(raw_node_feats.keys())
    num_node_entries = len(node_entries)
    if num_node_entries < num_truncated_nodes:
        global_logger.info(
            "Total sampled nodes in sub-graph asked, %d, are greater than "
            "total nodes in actual graph, %d. "
            "Moving ahead with all the nodes.", num_truncated_nodes,
            num_node_entries)
        num_truncated_nodes = num_node_entries
    sampled_node_indices = random.sample(range(num_node_entries),
                                         num_truncated_nodes)
    node_feats = {
        convert_whitespaces_to(node_entries[idx]):
        FEAT_TO_GROUP[raw_node_feats[node_entries[idx]]]
        for idx in sampled_node_indices
    }

    graph_edges = []
    involved_nodes = set()
    for chunk in pd.read_csv(raw_edges_file,
                             header='infer',
                             chunksize=CHUNK_SIZE,
                             on_bad_lines='skip'):
        # Filter rows with exactly 2 columns and collect the edges
        valid_edges = chunk[chunk.count(axis=1) == 2].values
        # graph_edges.extend([
        #     (elem1, elem2)
        #     for (elem1,
        #          elem2) in [(convert_whitespaces_to(str(row[0]).strip()),
        #                      convert_whitespaces_to(str(row[1]).strip()))
        #                     for row in valid_edges]
        #     if elem1 in node_feats.keys() and elem2 in node_feats.keys()
        # ])
        # hack to update a set in the same comprehension
        graph_edges.extend([
            (elem1, elem2) for row in valid_edges
            if (elem1 := convert_whitespaces_to(str(row[0]).strip())
                ) in node_feats.keys() and (elem2 := convert_whitespaces_to(
                    str(row[1]).strip())) in node_feats.keys() and
            not (involved_nodes.add(elem1)) and not (involved_nodes.add(elem2))
        ])

    G = nx.from_edgelist(graph_edges)
    node_feats = {k: v for k, v in node_feats.items() if k in involved_nodes}

    create_and_dump_graph_info_from_obj(G,
                                        node_feats,
                                        dump_files=(features_file, edges_file),
                                        logger=local_logger)
