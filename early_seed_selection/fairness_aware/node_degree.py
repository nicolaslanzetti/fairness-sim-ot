# Implements hrt_d from our paper

import networkx as nx

from utils import housekeeping_utils
from utils.graph_utils import get_group_ratio_in_nbrs

PROJECT_ROOT, global_logger = housekeeping_utils.hk_init()


def propose_seedset(G: nx.Graph,
                    node_feats: dict,
                    num_seeds: int,
                    edge_prob: float,
                    num_realizations=1000,
                    min_fair_ratio=0.4,
                    max_fair_ratio=0.6,
                    logger=global_logger) -> list:
    """Degree based seedset selection on heuristic

    Args:
        G (nx.Graph): social network object
        node_feats (dict): node features
        num_seeds (int): total seeds to be selected
        edge_prob (float): edge conduction probability in the random graph
        num_realizations (int, optional): # samples of the random graph.
          Defaults to 1000.
        min_fair_ratio (float, optional): heuristic fairness min score.
          Defaults to 0.4.
        max_fair_ratio (float, optional): heuristic fairness max score.
          Defaults to 0.6.
        logger (_type_, optional): logger used. Defaults to global_logger.

    Returns:
        list: seedset selected from the random social network
    """
    try:
        assert (G.number_of_nodes() >= num_seeds)
    except AssertionError as err:
        logger.error(
            "Total nodes in graph, %d, should atleast be "
            "total seeds expected, %d", G.number_of_nodes(), num_seeds)
        raise err

    # sorting based on degree from all edges is same as ones using stochastic
    # edges, as all edges have same probability of occuring in this model

    # assumption: binary integral group ids
    group_ratios = get_group_ratio_in_nbrs(G, node_feats, group_id=0)
    potential_seedset = [
        elem[0] for elem in list(
            sorted(G.degree, key=lambda elem: elem[1], reverse=True))
        if min_fair_ratio <= group_ratios[elem[0]] <= max_fair_ratio
    ]
    tot_potential_seeds = len(potential_seedset)
    try:
        assert tot_potential_seeds >= num_seeds
    except AssertionError as err:
        logger.error(
            "Total fairly-defined nodes in graph, %d, should atleast be "
            "total seeds expected, %d", tot_potential_seeds, num_seeds)
        raise err

    seedset = potential_seedset[:num_seeds]

    return seedset
