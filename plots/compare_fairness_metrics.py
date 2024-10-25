# comparing standard definitions of fairness (based on expected outreach)
#  against mutual-fairness

import matplotlib.pyplot as plt
import numpy as np

from metrics.group_metrics import (calc_point_fairness_metric,
                                   calculate_groupwise_dist)
from metrics.ot_metrics import (seedset_reach_transport_score,
                                translate_ot_scores_to_reverse_metrics)
from propagation.multiple_propagate import repeated_propagate
from propagation.propagate import propagate
from utils.data_utils import dump_vectors_to_csv
from utils.graph_utils import get_largest_graph_diameter
from utils.housekeeping_utils import hk_init
from utils.ot_utils import prepare_ot_references

PROJECT_ROOT, global_logger = hk_init()


# based on propagation.propagate.propagate
def fairness_with_conduction(G,
                             node_feats,
                             seedset,
                             realizations=1000,
                             bin_size=100,
                             logger=global_logger,
                             fig_name=None):
    """Routine to plot mutual-fairness and expected fairness across several
      conduction probabilities in a social network. This reflects the
        fundamental difference between the two metrics.

    Args:
        G (nx.Graph): Social Network
        node_feats (list): Node features list
        seedset (list): list of selected seeds as seedset
        realizations (int, optional): # realizations of random graph.
          Defaults to 1000.
        bin_size (int, optional): # of discrete units per group in the joint
          distribution digitized space. Defaults to 100.
        logger (Logger, optional): logger object used for logging. Defaults
          to global_logger.
        fig_name (string, optional): plot name specifiers. Defaults to None.
    """
    edge_probs = np.arange(0.0, 1.0001, 0.05)

    fair_only_ot_refs = prepare_ot_references(bin_size,
                                              rel_fairness=1,
                                              rel_efficiency=0)

    ot_fair_scores = translate_ot_scores_to_reverse_metrics([
        seedset_reach_transport_score(seedset,
                                      G,
                                      node_feats,
                                      edge_prob,
                                      ot_refs=fair_only_ot_refs,
                                      time_horizon_scale=1,
                                      realizations=realizations,
                                      bin_size=bin_size)
        for edge_prob in edge_probs
    ],
                                                            bin_size=bin_size)

    point_based_fair_scores = [
        calc_point_fairness_metric(
            calculate_groupwise_dist(repeated_propagate(
                propagate,
                realizations,
                G=G,
                node_feats=node_feats,
                edge_prob=edge_prob,
                seedset=seedset,
                time_horizon=get_largest_graph_diameter(G) // 1,
                logger=logger)[2],
                                     node_feats,
                                     logger=logger))
        for edge_prob in edge_probs
    ]

    if fig_name:
        dump_vectors_to_csv(
            str(fig_name) + '_fairness_in_point.csv', edge_probs,
            point_based_fair_scores)
        dump_vectors_to_csv(
            str(fig_name) + '_fairness_in_distr.csv', edge_probs,
            ot_fair_scores)

    fig, ax = plt.subplots()
    ax.set_xlabel('Conduction Probs --->')
    ax.set_ylabel('Fairness(scaled to [0, 1])--->')
    ax.plot(edge_probs,
            point_based_fair_scores,
            color='tab:red',
            label='$fair_{exp}$')
    ax.plot(edge_probs, ot_fair_scores, color='tab:blue', label='$fair_{ot}$')
    plt.xticks(np.arange(0, 1.1, 0.1))

    ax.legend()

    if fig_name:
        plt.savefig(str(fig_name) + '_fairness.png')
        plt.close()

    # plt.close()

    return
