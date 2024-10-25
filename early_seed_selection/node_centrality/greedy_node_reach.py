# Implements bas_g from our paper

from queue import PriorityQueue

import networkx as nx

from utils import housekeeping_utils
from utils.graph_utils import edge_conducts

PROJECT_ROOT, global_logger = housekeeping_utils.hk_init()

# for calculating debug info during seed selection
# assumption, binary group characterization in graphs
DEFAULT_DEBUG_INFO = {
    "avg_reach_per_seed": [],
    "avg_group_0_reach_per_seed": [],
    "total_avg_reach": 0,
    "total_avg_group_0_reach": 0,
    # trajectory of total values from above, as we select a new seed each time
    "total_avg_reach_in_seed_sel": [],
    "total_avg_group_0_reach_in_seed_sel": []
}


def process_debug_info(seedset, debug_info, logger=global_logger):
    """logs statistics due to a seedset selection

    Args:
        seedset (list): list of seeds selected as seedset
        debug_info (dict): of the format DEFAULT_DEBUG_INFO defined above
        logger (Logger, optional): logger used to log the debug info.
          Defaults to global_logger.
    """
    assert debug_info.keys() == DEFAULT_DEBUG_INFO.keys()

    logger.debug("==================== <GREEDY SEED SELECTION METRICS"
                 "====================")
    logger.debug("Total average node reach due to all seeds, %f",
                 debug_info["total_avg_reach"])
    logger.debug("Total average Group 0 node reach due to all seeds, %f",
                 debug_info["total_avg_group_0_reach"])

    for i, seed in enumerate(seedset):
        logger.debug("\n")
        logger.debug("<For seed %s: ====================", seed)
        logger.debug("\n")
        logger.debug("Average node reach, %f",
                     debug_info["avg_reach_per_seed"][i])
        logger.debug("Average Group 0 node reach, %f",
                     debug_info["avg_group_0_reach_per_seed"][i])
        logger.debug(
            "Total average node reached until this seed selection, %f",
            debug_info["total_avg_reach_in_seed_sel"][i])
        logger.debug(
            "Total average Group 0 nodes reached until"
            "this seed selection, %f",
            debug_info["total_avg_group_0_reach_in_seed_sel"][i])
        logger.debug("\n")
        logger.debug("For seed %s> ====================", seed)
    logger.debug("==================== GREEDY SEED SELECTION METRICS>"
                 "====================")


def propose_seedset(G: nx.Graph,
                    node_feats: dict,
                    num_seeds: int,
                    edge_prob: float,
                    num_realizations=1000,
                    logger=global_logger) -> list:
    """Greedy seedset selection

    Args:
        G (nx.Graph): social network object
        node_feats (dict): node features
        num_seeds (int): total seeds to be selected
        edge_prob (float): edge conduction probability in the random graph
        num_realizations (int, optional): # samples of the random graph.
          Defaults to 1000.
        logger (Logger, optional): logger used. Defaults to global_logger.

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

    seedset = []  # a list in order

    # for performing seedset selection process
    node_reachability = {node: 0 for node in G.nodes()}
    uniq_comp_id = 1  # across all realizations
    comp_ids_per_node = {node: set() for node in G.nodes()}
    uniq_comp_id_info = {}  # uniq_comp_id: (comp_len, num_group_0_in_comp)

    debug_info = DEFAULT_DEBUG_INFO

    for realization in range(num_realizations):
        logger.debug("Greedy Node Reach, Realization %d", realization)
        realization_edges = [
            edge for edge in G.edges() if edge_conducts(edge_prob)
        ]

        realized_sub_G = nx.Graph()
        # independent nodes won't make it in this sub-graph
        # Remember, nodes are string ids
        realized_sub_G.add_edges_from(realization_edges)
        comps_in_realization = [
            realized_sub_G.subgraph(c)
            for c in nx.connected_components(realized_sub_G)
        ]

        for realization_comp_itr in range(len(comps_in_realization)):
            comp_len = len(comps_in_realization[realization_comp_itr])
            comp_nodes = comps_in_realization[realization_comp_itr].nodes()
            # for characterization of uniq_comp_id,
            # assumption, binary group characterization in graphs
            num_group_0_in_comp = len([
                comp_node for comp_node in comp_nodes
                if node_feats[comp_node] == 0
            ])

            for comp_node in comp_nodes:
                node_reachability[comp_node] += comp_len
                comp_ids_per_node[comp_node].add(uniq_comp_id)

            uniq_comp_id_info[uniq_comp_id] = (comp_len, num_group_0_in_comp)

            uniq_comp_id += 1

    node_pqueue_on_reach = PriorityQueue()  # min-heap
    for node in G.nodes():
        node_pqueue_on_reach.put((-node_reachability[node], node))

    covered_comp_ids = set()

    for _ in range(num_seeds):
        reach, seed = node_pqueue_on_reach.get()
        reach = -1 * reach

        while len(covered_comp_ids.intersection(comp_ids_per_node[seed])) != 0:
            comp_ids_per_node[seed] = comp_ids_per_node[seed].difference(
                covered_comp_ids)

            adjusted_node_reach = sum([
                uniq_comp_id_info[comp_id][0]
                for comp_id in comp_ids_per_node[seed]
            ])
            node_pqueue_on_reach.put((-adjusted_node_reach, seed))

            reach, seed = node_pqueue_on_reach.get()
            reach = -1 * reach

        covered_comp_ids.update(comp_ids_per_node[seed])

        # extract performance of the selected seed (for debugging)
        # assumption, binary group characterization in graphs
        seed_reach_group_0 = sum([
            uniq_comp_id_info[comp_id][1]
            for comp_id in comp_ids_per_node[seed]
        ])

        seedset.append(seed)

        # debug info
        debug_info["avg_reach_per_seed"].append(reach * 1.0 / num_realizations)
        debug_info["avg_group_0_reach_per_seed"].append(seed_reach_group_0 *
                                                        1.0 / num_realizations)
        debug_info["total_avg_reach"] += (reach * 1.0 / num_realizations)
        debug_info["total_avg_reach_in_seed_sel"].append(
            debug_info["total_avg_reach"])
        debug_info["total_avg_group_0_reach"] += (seed_reach_group_0 * 1.0 /
                                                  num_realizations)
        debug_info["total_avg_group_0_reach_in_seed_sel"].append(
            debug_info["total_avg_group_0_reach"])

    process_debug_info(seedset, debug_info, logger=logger)
    return seedset
