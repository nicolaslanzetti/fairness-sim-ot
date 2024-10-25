# Optimal Transport score (1 - beta-fairness, from paper) metrics

import numpy as np

from metrics.group_metrics import calculate_groupwise_dist
from plots.plot_2d import plot_groupwise_outreach_histogram
from propagation.multiple_propagate import repeated_propagate
from propagation.propagate import propagate
from utils.graph_utils import get_largest_graph_diameter
from utils.housekeeping_utils import hk_init
from utils.ot_utils import (calculate_emd_matrix,
                            create_fair_and_efficient_2d_cost_matrix,
                            create_ideal_fair_and_efficient_2d_target_dist,
                            create_most_unfair_2d_target_dist,
                            unroll_dist_from_hist_bins)

PROJECT_ROOT, global_logger = hk_init()

MAX_FAIRNESS_OT_SCORE_MEM = None  # should be (bin_size, score)


# OT metric here is calculated using propagation.propagate
def reach_transport_score_helper(seedset,
                                 G,
                                 node_feats,
                                 edge_prob,
                                 ot_refs,
                                 time_horizon_scale=1,
                                 realizations=1000,
                                 bin_size=100):
    target_dist, M, scale_k = ot_refs["target_dist"], ot_refs[
        "cost_mat"], ot_refs["scale"]
    _, _, runs_group_info = repeated_propagate(
        propagate,
        realizations,
        G=G,
        node_feats=node_feats,
        edge_prob=edge_prob,
        seedset=seedset,
        time_horizon=get_largest_graph_diameter(G) // time_horizon_scale)

    reach_per_group_per_run = calculate_groupwise_dist(runs_group_info,
                                                       node_feats)
    plot_hist_in_bins = plot_groupwise_outreach_histogram(
        reach_per_group_per_run, bin_size, return_only_bins=True)

    score, _ = calculate_emd_matrix(
        unroll_dist_from_hist_bins(plot_hist_in_bins[0]), target_dist[2], M,
        scale_k)

    return score


def seed_reach_transport_score(eval_seed, *args, **kwargs):
    """routine for OT score due to outreach from a single seed as a seedset
    """
    lone_seedset = [eval_seed]
    if "time_horizon_scale" not in kwargs.keys():
        kwargs["time_horizon_scale"] = 8  # default value
    return reach_transport_score_helper(lone_seedset, *args, **kwargs)


def seedset_reach_transport_score(seedset, *args, **kwargs):
    """routine for OT score due to outreach from a given seedset
    """
    if "time_horizon_scale" not in kwargs.keys():
        kwargs["time_horizon_scale"] = 1  # default value
    return reach_transport_score_helper(seedset, *args, **kwargs)


def translate_ot_scores_to_reverse_metrics(scores, bin_size):
    """normalize and linearly invert OT scores to get beta-fairness
    """
    max_ot_score = calculate_max_fairness_ot_score(bin_size)
    if isinstance(scores, list):
        reverse_metrics = ((max_ot_score - np.array(scores)) /
                           max_ot_score).tolist()

    elif isinstance(scores, dict):
        reverse_metrics = {
            key: (max_ot_score - val) / max_ot_score
            for key, val in scores.items()
        }

    return reverse_metrics


def calculate_max_fairness_ot_score(bin_size):
    """for normalization purposes, calculate the maximum beta-fairness
    """
    global MAX_FAIRNESS_OT_SCORE_MEM
    if MAX_FAIRNESS_OT_SCORE_MEM is not None and MAX_FAIRNESS_OT_SCORE_MEM[
            0] == bin_size:
        return MAX_FAIRNESS_OT_SCORE_MEM[1]

    target_dist = create_ideal_fair_and_efficient_2d_target_dist(bin_size)
    M, scale_k = create_fair_and_efficient_2d_cost_matrix(
        bin_size, True, 1.0, False, 0.0)
    source_dist = create_most_unfair_2d_target_dist(bin_size)

    ot_score = calculate_emd_matrix(source_dist[2], target_dist[2], M,
                                    scale_k)[0]

    MAX_FAIRNESS_OT_SCORE_MEM = (bin_size, ot_score)

    return MAX_FAIRNESS_OT_SCORE_MEM[1]
