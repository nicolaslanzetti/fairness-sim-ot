import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from utils import housekeeping_utils

PROJECT_ROOT, global_logger = housekeeping_utils.hk_init()

SMALL_NODE_SIZE = 300
LARGE_NODE_SIZE = 1000
NODE_COLORS = ['red', 'blue']
MAX_DIAM_PROP_NAME = "prop_max_diam"


def edge_conducts(prob):
    return random.random() < prob


def draw_random_graph(G, prob, flagged_nodes):
    values = [1 if i in flagged_nodes else 0 for i in G.nodes()]
    pos = nx.spring_layout(G)

    edge_labels = {}
    for elem in G.edges:
        edge_labels[(elem[0], elem[1])] = prob

    nx.draw(G,
            pos=pos,
            cmap=plt.get_cmap('viridis'),
            node_color=values,
            with_labels=True,
            font_color='white')
    nx.draw_networkx_edge_labels(G,
                                 pos=pos,
                                 edge_labels=edge_labels,
                                 font_color='red')
    plt.show()


def draw_social_graph(G,
                      prob,
                      seedset,
                      node_feats,
                      plot_plain=True,
                      fig_name=None):
    # based on admission to seedset
    node_sizes = [
        LARGE_NODE_SIZE if i in seedset else SMALL_NODE_SIZE
        for i in G.nodes()
    ]
    # ideally should support vector node_feats, based on group ids
    node_colors = [NODE_COLORS[node_feats[i]] for i in G.nodes()]

    pos = nx.spring_layout(G)
    edge_labels = {}
    for elem in G.edges:
        edge_labels[(elem[0], elem[1])] = prob

    nx.draw(G,
            pos=pos,
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=(not plot_plain),
            font_color='white')
    nx.draw_networkx_edge_labels(
        G,
        pos=pos,
        edge_labels=({} if plot_plain else edge_labels),
        font_color='brown')

    if fig_name:
        plt.savefig(str(fig_name) + '_social_graph.png')

    else:
        plt.show()


# works for graphs with several connected components
def get_largest_graph_diameter(G: nx.graph):
    if MAX_DIAM_PROP_NAME in G.graph.keys():
        return G.graph[MAX_DIAM_PROP_NAME]

    comp_diams = [
        nx.diameter(G.subgraph(c)) for c in nx.connected_components(G)
    ]
    assert len(comp_diams) > 0
    comp_diams.sort()  # ascending order

    max_diam = comp_diams[-1]
    G.graph[MAX_DIAM_PROP_NAME] = max_diam

    return max_diam


def log_graph_properties(G: nx.graph, logger=global_logger):
    logger.info("Graph properties:")
    logger.info("Total nodes, %d", G.number_of_nodes())
    logger.info("Total edges, %d", G.number_of_edges())
    logger.info("Avg node degree, %f",
                np.mean([G.degree[node] for node in G.nodes()]))
    logger.info("Graph largest diameter, %d", get_largest_graph_diameter(G))


def log_social_graph_data_summary(G, node_feats, logger=global_logger):
    logger.info("Loaded graph dataset summary:")
    log_graph_properties(G, logger=logger)

    # assumption: binary scalar groups
    total_nodes_in_group_0 = len(
        [key for key, val in node_feats.items() if val == 0])
    total_nodes_in_group_1 = len(
        [key for key, val in node_feats.items() if val == 1])

    logger.info("Total nodes from Group 0: %d", total_nodes_in_group_0)
    logger.info("Total nodes from Group 1, %d", total_nodes_in_group_1)

    # assumption, groups are binary
    frac_intergroup_edges = len([
        edg for edg in G.edges() if node_feats[edg[0]] != node_feats[edg[1]]
    ]) * 1.0 / G.number_of_edges()

    logger.info("Fraction of intergroup Edges, %.5f", frac_intergroup_edges)


def get_group_ratio_in_nbrs(G, node_feats, group_id=0):
    # assumption 2 groups based ratio for each node calculated
    group_ratio_in_nbrs_per_node = {
        elem[0]: (1.0 * len([
            nbr for nbr in G.neighbors(elem[0]) if node_feats[nbr] == group_id
        ])) / elem[1]
        for elem in G.degree
    }

    return group_ratio_in_nbrs_per_node
