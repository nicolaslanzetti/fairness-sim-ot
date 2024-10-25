# use similar files to extract and create 'graph_edges.csv' and
# 'graph_node_features.csv' from dataset dumps

import csv
import os
from os.path import exists

import networkx as nx
import pandas as pd

from utils import housekeeping_utils
from utils.data_utils import create_and_dump_graph_info_from_obj

PROJECT_ROOT, global_logger = housekeeping_utils.hk_init()

DS_NAME = "ind_villages_ds"
DEMO_DF_FILENAME = "indianvillages/Demographics/individual_characteristics.dta"
VILLAGE_REL_FILENAME_PREFIX = 'indianvillages/Adjacency Matrices/adj_allVillageRelationships_vilno_'
NODE_FEAT_FILENAME = "graph_node_features.csv"
EDGES_FILENAME = "graph_edges.csv"

TOT_VILLAGES = 77  # numbered 1 to 77
VILL_IDX_CHOICE = 75  # apt choices [5, 59, 75]
DEMO_ATTR = "religion"
REL_VALUES = ['ISLAM', 'HINDUISM']
REL_TO_IDX = {"ISLAM": 0, "HINDUISM": 1}

FEAT_VALUES = ["TELUGU", "KANNADA"]
FEAT_COND_TO_GROUP = {True: 0, False: 1}


def create_and_dump_graph_info(raw_root=PROJECT_ROOT + "/data/" + DS_NAME +
                               "/raw/"):
    local_logger = housekeeping_utils.get_local_logger(
        DS_NAME, PROJECT_ROOT + "/data/" + DS_NAME + "/metadata.txt")
    demo_df = pd.read_stata(raw_root + DEMO_DF_FILENAME)
    found_village_indices = []
    religion_ratio_in_villages = {}

    # vilalge index starts from 1
    for vill_idx in range(1, TOT_VILLAGES + 1):
        village_file = raw_root + VILLAGE_REL_FILENAME_PREFIX + str(
            vill_idx) + '.csv'
        if exists(village_file):
            found_village_indices.append(vill_idx)

            # list_of_known_religions = list(
            # demo_df[demo_df['village'] == vill_idx]['adjmatrix_key'])

            # village_df = pd.read_csv(village_file, header=None)
            # village_np = village_df.to_numpy()
            # village_orig_G = nx.from_numpy_matrix(village_np)
            # village_known_rel_G = village_orig_G.subgraph(
            #     list_of_known_religions)
            # village_known_rel_largest_cc = sorted(
            #     nx.connected_components(village_known_rel_G),
            #     key=len,
            #     reverse=True)[0]
            # village_known_rel_largest_cc_G = village_known_rel_G.subgraph(
            # village_known_rel_largest_cc)

            # village_target_G = nx.Graph(village_known_rel_largest_cc_G)

            # nodes = list(village_target_G.nodes())
            religion_nodes = list(
                demo_df[demo_df['village'] == vill_idx][DEMO_ATTR])

            rel_val_count = [0] * len(REL_VALUES)

            for religion_node in religion_nodes:
                if religion_node not in REL_VALUES:
                    continue
                rel_val_count[REL_TO_IDX[religion_node]] += 1

            # assumption/nature of dataset processing-- 2 religion assumption
            rel_1_count = rel_val_count[REL_TO_IDX[REL_VALUES[1]]]
            rel_0_count = rel_val_count[REL_TO_IDX[REL_VALUES[0]]]
            religion_ratio_in_villages[vill_idx] = rel_1_count / (rel_1_count +
                                                                  rel_0_count)

    # use religion_ratio_in_villages to decide any village's node label (village node's mothertongue)
    list_of_known_religions = list(
        demo_df[demo_df['village'] == VILL_IDX_CHOICE]['adjmatrix_key'])

    village_file = raw_root + VILLAGE_REL_FILENAME_PREFIX + str(
        vill_idx) + '.csv'
    village_df = pd.read_csv(village_file, header=None)
    village_np = village_df.to_numpy()
    village_orig_G = nx.from_numpy_array(village_np)
    village_known_rel_G = village_orig_G.subgraph(list_of_known_religions)
    village_known_rel_largest_cc = sorted(
        nx.connected_components(village_known_rel_G), key=len, reverse=True)[0]
    village_known_rel_largest_cc_G = village_known_rel_G.subgraph(
        village_known_rel_largest_cc)

    village_target_G = nx.Graph(village_known_rel_largest_cc_G)
    village_target_G_with_loops = nx.Graph(village_target_G)
    village_target_G_with_loops.add_edges_from([
        (node, node) for node in village_target_G_with_loops.nodes()
    ])

    target_nodes = list(village_target_G_with_loops.nodes())
    mothertongue_nodes = list(
        demo_df[demo_df['village'] == VILL_IDX_CHOICE]['mothertongue'])

    minmaj_mothertonge = FEAT_VALUES[0] if religion_ratio_in_villages[
        VILL_IDX_CHOICE] > 0.5 else FEAT_VALUES[1]

    node_feats = {
        node: FEAT_COND_TO_GROUP[mothertongue_nodes[i] == minmaj_mothertonge]
        for i, node in enumerate(target_nodes)
    }

    features_file = os.path.join(PROJECT_ROOT + "/data/" + DS_NAME,
                                 NODE_FEAT_FILENAME)
    edges_file = os.path.join(PROJECT_ROOT + "/data/" + DS_NAME,
                              EDGES_FILENAME)

    create_and_dump_graph_info_from_obj(village_target_G_with_loops,
                                        node_feats,
                                        dump_files=(features_file, edges_file),
                                        logger=local_logger)
