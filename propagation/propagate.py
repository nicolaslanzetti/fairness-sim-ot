# Implementation of the Independent Cascade information propagation algo

from collections import deque

import networkx as nx

from utils import housekeeping_utils
from utils.graph_utils import edge_conducts

PROJECT_ROOT, global_logger = housekeeping_utils.hk_init()


def propagate(G: nx.Graph,
              node_feats: dict,
              edge_prob: float,
              seedset: list,
              time_horizon: int,
              logger=global_logger) -> tuple:
    """ simulates the process of a single cascade in a random graph

    Args:
        G (nx.Graph): underlying random graph with all random edges present
        (they are sampled later)
        node_feats (dict): node features for the corresponding graph, G
        edge_prob (float): probability of each edge's conduction
        seedset (list): the proposed seedset with the information
        time_horizon (int): time-step until which propagation takes place
        logger (Logger, optional): logger used for logging. Defaults to
        global_logger.

    Returns:
        tuple: 1. dict: nodes with values as
        (boolean, (underlying_seed, time_step)), denoting whether the node got
        reached, and if so, then due to which seed of the seedset, and by
        what aggregated time step
        2. dict: for every seed,
        [total_nodes_reached, {group_id: [total_group_i_nodes_reached,
        total_time_taken],...}]
        3. dict: for every group,
        (total_group_i_nodes_reached, total_time_taken)

    """

    prop_info = {}
    seed_info = {}
    group_info = {}

    ts_max = time_horizon
    uniq_seedset = list(set(seedset))

    uniq_groups = set(node_feats.values())
    for node in G.nodes:
        prop_info[node] = (False, None)

    for group in uniq_groups:
        group_info[group] = (0, 0)

    for seed in uniq_seedset:
        prop_info[seed] = (True, (seed, 0))  # node reached at t=0
        seed_info[seed] = [
            1,
            dict(
                zip(list(uniq_groups),
                    [[0, 0] for _ in range(len(uniq_groups))]))
        ]
        seed_info[seed][1][node_feats[seed]] = [1, 0]
        group_info[node_feats[seed]] = (group_info[node_feats[seed]][0] + 1, 0)

    bfs_node_queue = deque(uniq_seedset)

    ts = 0  # propagation timestamp
    reached_nodes_set = set(uniq_seedset)

    while len(bfs_node_queue) > 0 and ts < ts_max:
        ts += 1
        curr_parent_length = len(bfs_node_queue)
        for _ in range(curr_parent_length):
            curr_parent = bfs_node_queue.popleft()
            for curr_child in set(
                    G.neighbors(curr_parent)).difference(reached_nodes_set):
                if edge_conducts(edge_prob):
                    assert prop_info[curr_parent][0]  # already visited
                    parent_seed = prop_info[curr_parent][1][0]
                    prop_info[curr_child] = (True, (parent_seed, ts))

                    group_info[node_feats[curr_child]] = (
                        group_info[node_feats[curr_child]][0] + 1,
                        group_info[node_feats[curr_child]][1] + ts)

                    # seed covered one more node in total
                    seed_info[parent_seed][0] += 1
                    seed_info[parent_seed][1][node_feats[curr_child]][0] += 1
                    seed_info[parent_seed][1][node_feats[curr_child]][1] += ts

                    bfs_node_queue.append(curr_child)
                    reached_nodes_set.add(curr_child)

    return prop_info, seed_info, group_info
