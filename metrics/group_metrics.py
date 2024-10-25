from collections import defaultdict

import numpy as np

from utils.housekeeping_utils import hk_init

PROJECT_ROOT, global_logger = hk_init()


def calculate_groupwise_dist(multi_run_group_info: dict,
                             node_feats: dict,
                             logger=global_logger):
    """returns the outreach distribution of each group 
    """
    uniq_groups = set(multi_run_group_info.keys())
    n_times = len(multi_run_group_info[list(uniq_groups)[0]])
    group_node_counts = defaultdict(lambda: 0)

    for node in node_feats.keys():
        group_node_counts[node_feats[node]] += 1

    group_node_counts = dict(group_node_counts)

    assert uniq_groups == set(group_node_counts.keys())

    frac_reach_per_group_per_run = {}

    for group in uniq_groups:
        frac_reach_per_group_per_run[group] = []

    for group in uniq_groups:
        for run_idx in range(n_times):
            frac_reach_per_group_per_run[group].append(
                multi_run_group_info[group][run_idx][0] /
                group_node_counts[group])

    logger.info("==================== <GROUP METRICS ====================")
    logger.info("Total realizations: %d", n_times)
    for group in uniq_groups:
        logger.info("\n")
        logger.info("For group index, %d, distribution of fractional reach,",
                    group)
        logger.info("Mean: %f, Std Dev: %f",
                    np.mean(frac_reach_per_group_per_run[group]),
                    np.std(frac_reach_per_group_per_run[group]))
    logger.info("\n")
    logger.info("==================== GROUP METRICS> ====================")

    return frac_reach_per_group_per_run


def calc_point_fairness_metric(frac_reach_per_group_per_run):
    """ given the outreach distribution for each group, it calculates its
        Mutual Fairness (beta=1). Can use 'n' instead of a '2' in the
        fairness_score formula to calculate mutual-fairness amongst n-groups.
    """
    # A (0, 0, 0, ...) is a point on the iso fair line
    # P is the expected fairness point, marginalized for each group
    # d = B (1, 1, 1,...) - A, called the line direction vector

    num_groups = len(frac_reach_per_group_per_run.keys())
    P = np.array([
        np.mean(frac_reach_per_group_per_run[key])
        for key in frac_reach_per_group_per_run.keys()
    ])

    d = np.array([1.0] * num_groups)
    AP = P

    dist = np.linalg.norm(np.cross(AP, d)) / np.linalg.norm(d)
    fairness_score = (1.0 - dist * np.sqrt(2))

    return fairness_score
