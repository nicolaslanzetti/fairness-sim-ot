# Plots comparing seedset selections on the mutual-fairness (beta=0) and
#  efficiency landspace

import matplotlib.pyplot as plt

from metrics.node_metrics import get_propagation_efficiency
from metrics.ot_metrics import (seedset_reach_transport_score,
                                translate_ot_scores_to_reverse_metrics)
from propagation.multiple_propagate import repeated_propagate
from propagation.propagate import propagate
from utils.data_utils import dump_vectors_to_csv
from utils.graph_utils import get_largest_graph_diameter
from utils.housekeeping_utils import hk_init
from utils.ot_utils import prepare_ot_references

# import numpy as np

PROJECT_ROOT, global_logger = hk_init()


def fairness_efficiency_scatter_for_seedsets(G,
                                             node_feats,
                                             edge_prob,
                                             labelled_seedsets,
                                             realizations=1000,
                                             bin_size=100,
                                             logger=global_logger,
                                             fig_name=None):
    """ Plots seedset selection algorithms/heuristics on the E-f, the
        efficiency and mutual-fairness, space. All seedset selections from
        labelled_seedsets are evaluated for their efficiency and
        (beta=0) fairness, that is mutual-fairness, and then plotted as
        a single point on this E-f space.
    """
    # x-axis -> efficiency, y-axis -> fairness
    labelled_efficiency_for_seedsets = {
        label:
        get_propagation_efficiency(
            repeated_propagate(propagate,
                               realizations,
                               G=G,
                               node_feats=node_feats,
                               edge_prob=edge_prob,
                               seedset=seedset,
                               time_horizon=get_largest_graph_diameter(G) // 1,
                               logger=logger)[0])
        for label, seedset in labelled_seedsets.items()
    }

    fair_only_ot_refs = prepare_ot_references(bin_size,
                                              rel_fairness=1,
                                              rel_efficiency=0)

    labelled_fairness_for_seedsets = translate_ot_scores_to_reverse_metrics(
        {
            label:
            seedset_reach_transport_score(seedset,
                                          G,
                                          node_feats,
                                          edge_prob,
                                          ot_refs=fair_only_ot_refs,
                                          time_horizon_scale=1,
                                          realizations=realizations,
                                          bin_size=bin_size)
            for label, seedset in labelled_seedsets.items()
        }, bin_size)

    fig, ax = plt.subplots()
    ax.scatter([
        efficiency
        for _, efficiency in labelled_efficiency_for_seedsets.items()
    ], [
        fairness_score
        for _, fairness_score in labelled_fairness_for_seedsets.items()
    ],
               marker='x')

    all_ef_coords_x = []
    all_ef_coords_y = []
    all_ef_labels = []
    for label, efficiency in labelled_efficiency_for_seedsets.items():
        ax.annotate(label, (efficiency, labelled_fairness_for_seedsets[label]))
        all_ef_coords_x.append(efficiency)
        all_ef_coords_y.append(labelled_fairness_for_seedsets[label])
        all_ef_labels.append(label)

    if fig_name:
        dump_vectors_to_csv(
            str(fig_name) + '_EF_values.csv', all_ef_coords_x, all_ef_coords_y,
            all_ef_labels)

    ax.set_title('Efficiency-Fairness Plot for different seedsets')
    ax.set_xlabel("Efficiency ([0, 1]) --->")
    ax.set_ylabel("Fairness(scaled to [0, 1]) --->")

    # ax.set_xticks(np.arange(0, 1.1, 0.1))
    # ax.set_yticks(np.arange(0, 1.1, 0.1))

    fig.tight_layout()

    if fig_name:
        plt.savefig(str(fig_name) + '_EF_plot.png')
        plt.close()

    # plt.close()

    return labelled_efficiency_for_seedsets, labelled_fairness_for_seedsets
