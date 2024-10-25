import csv
import json
import os
import pickle
import random

import networkx as nx
import numpy as np

from utils import housekeeping_utils

PROJECT_ROOT, global_logger = housekeeping_utils.hk_init()


def load_graph_from_edgelist(
    edgelist_file=PROJECT_ROOT + '/data/sample_data/graph_edges.csv'
) -> nx.Graph:
    assert os.path.exists(edgelist_file)
    global_logger.debug("Loading graph from %s", edgelist_file)

    nx_graph = nx.read_edgelist(edgelist_file, comments=None)

    return nx_graph


def load_graph_node_features(
    features_file=PROJECT_ROOT + '/data/sample_data/graph_node_features.csv'
) -> dict:
    assert os.path.exists(features_file)
    global_logger.debug("Loading node features from %s", features_file)

    file_handle = open(features_file, 'r')

    node_features = {}
    reader = csv.reader(file_handle)

    for row in reader:
        # TODO(schowdhary): Should ideally be a vector read
        node_features[row[0]] = int(row[1])

    return node_features


def get_distinct_random_row_idxs_from_csv(num_samples, reader, file_len):
    if num_samples >= file_len:
        global_logger.info(
            "Total sampled nodes in sub-graph asked, %d, are equal/greater"
            "than total nodes in actual graph, %d. "
            "Moving ahead with all the nodes.", num_samples, file_len)
        selected_rows = [row for idx, row in enumerate(reader)]
    else:
        offsets = random.sample(range(file_len), num_samples)
        selected_rows = [
            row for idx, row in enumerate(reader) if idx in offsets
        ]

    return selected_rows


def create_and_dump_graph_info_helper(num_truncated_nodes,
                                      raw_data_files,
                                      new_data_files,
                                      feat_to_grp_map,
                                      logger=global_logger):
    selected_nodes = set()
    selected_nodes_info = {}  # node_id: node_feat, assumption to be single int

    raw_features_file = raw_data_files["nodes"]
    raw_edges_file = raw_data_files["edges"]
    features_file = new_data_files["nodes"]
    edges_file = new_data_files["edges"]

    with open(raw_features_file) as readfile, open(
            raw_features_file) as readfile_2:
        reader = csv.reader(readfile, delimiter=',')
        reader_2 = csv.reader(readfile_2, delimiter=',')

        file_length = len(list(reader_2))

        for row in get_distinct_random_row_idxs_from_csv(
                num_truncated_nodes, reader, file_length):
            node_id = row[0]
            assert bool(selected_nodes.intersection([node_id])) is False
            selected_nodes.add(node_id)
            group_id = feat_to_grp_map[row[1]]
            selected_nodes_info[node_id] = group_id

    with open(raw_edges_file) as readfile, open(edges_file,
                                                'w') as edg_writefile, open(
                                                    features_file,
                                                    'w') as node_writefile:
        reader = csv.reader(readfile, delimiter=',')
        edg_writer = csv.writer(edg_writefile,
                                delimiter=',',
                                quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
        node_writer = csv.writer(node_writefile,
                                 delimiter=',',
                                 quotechar='"',
                                 quoting=csv.QUOTE_MINIMAL)

        # we do not need independent nodes in ...
        # ... the original dataset we mine ...
        # ... (see data_utils.load_graph_from_edgelist)
        relevant_nodes = set()
        num_edges_selected = 0

        for row in reader:
            if row[0] not in selected_nodes or row[1] not in selected_nodes:
                continue
            edge_nodes = [row[0], row[1]]
            edg_writer.writerow([" ".join(edge_nodes)])
            num_edges_selected += 1
            relevant_nodes.add(row[0])
            relevant_nodes.add(row[1])

        for rel_node in relevant_nodes:
            # implicit string conversion for group_id
            node_writer.writerow([rel_node, selected_nodes_info[rel_node]])

        logger.info("Total nodes, edges in sub-graph, %d, %d",
                    len(list(relevant_nodes)), num_edges_selected)


def create_and_dump_graph_info_from_obj(G, node_feats, dump_files, logger):
    features_file, edges_file = dump_files

    with open(features_file, 'w') as node_writefile:
        node_writer = csv.writer(node_writefile,
                                 delimiter=',',
                                 quotechar='"',
                                 quoting=csv.QUOTE_MINIMAL)

        for node in G.nodes():
            node_writer.writerow([node, node_feats[node]])

    with open(edges_file, 'w') as edg_writefile:
        edg_writer = csv.writer(edg_writefile,
                                delimiter=',',
                                quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)

        for edg in G.edges():
            edg_writer.writerow([" ".join([str(edg[0]), str(edg[1])])])

    logger.info("Total nodes, edges in sub-graph, %d, %d", G.number_of_nodes(),
                G.number_of_edges())


def dump_config_to_json(config, filename):
    with open(filename, 'w') as fp:
        json.dump(config, fp, indent=4)


def dump_obj(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_to_obj(filename):
    obj = None
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj


def dump_vectors_to_csv(filename, *args):
    args_lens = [len(arg) for arg in args]
    max_len = max(args_lens)
    num_args = len(args_lens)

    with open(filename, 'w') as f:
        write = csv.writer(f)
        for i in range(max_len):
            row = [
                args[j][i] if i < args_lens[j] else "" for j in range(num_args)
            ]
            write.writerow(row)


def load_vectors_from_csv(filename):
    vectors = None
    num_vectors = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if num_vectors == 0 and vectors is None:
                vectors = [[] for _ in range(len(row))]
                num_vectors = len(vectors)
            for j, elem in enumerate(row):
                vectors[j].append(
                    float(elem) if housekeeping_utils.is_float(elem) else elem)

    return tuple(vectors)


def convert_2d_matrix_to_vector_features(mat):
    x_dim, y_dim = mat.shape
    x_vec = []
    y_vec = []
    feat_data = []

    for i in range(x_dim):
        for j in range(y_dim):
            x_vec.append(i)
            y_vec.append(j)
            feat_data.append(mat[i, j])

    return (x_vec, y_vec, feat_data)


def calculate_sample_mu_sigma(samples):
    return np.mean(samples), np.std(samples)
