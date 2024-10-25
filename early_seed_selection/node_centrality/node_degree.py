# Implements bas_d from our paper

import networkx as nx

from utils import housekeeping_utils

PROJECT_ROOT, global_logger = housekeeping_utils.hk_init()


def propose_seedset(G: nx.Graph,
                    node_feats: dict,
                    num_seeds: int,
                    edge_prob: float,
                    num_realizations=1000,
                    logger=global_logger) -> list:
    """Degree based seedset selection
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
    seedset = list(sorted(G.degree, key=lambda elem: elem[1],
                          reverse=True))[:num_seeds]
    seedset = [seed_info[0] for seed_info in seedset]
    return seedset
