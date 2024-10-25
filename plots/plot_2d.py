# Plotting joint distribution of outreach between groups

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from metrics.group_metrics import calculate_groupwise_dist
from propagation.multiple_propagate import repeated_propagate
from propagation.propagate import propagate
from utils.data_utils import (convert_2d_matrix_to_vector_features,
                              dump_vectors_to_csv)
from utils.graph_utils import get_largest_graph_diameter
from utils.housekeeping_utils import hk_init

PROJECT_ROOT, global_logger = hk_init()


def plot_2d_iso_curves(plot_obj, least_count=0.2, fig_name=None):
    """Plots constant mutual-fairness and constant efficiency lines in
      the outreach distribution space
    """
    # draw references (hard-code on [0, 1]^2 space)

    # iso-fairness
    central_iso_fair_coord = [[0, 0], [1, 1]]
    all_iso_fair_x_coords = []
    all_iso_fair_y_coords = []
    all_iso_fair_values = []

    for target_coord_idx in [0, 1]:
        for adj in np.arange(0, 1, least_count):
            iso_fair_coord = deepcopy(central_iso_fair_coord)
            iso_fair_coord[0][target_coord_idx] += adj
            iso_fair_coord[1][1 - target_coord_idx] -= adj
            x_ref = [elem[0] for elem in iso_fair_coord]
            y_ref = [elem[1] for elem in iso_fair_coord]
            plot_obj.plot(x_ref,
                          y_ref,
                          marker='.',
                          ls='--',
                          lw=(1 - adj) * 0.1,
                          color='blue')
            x_coords = np.arange(x_ref[0], x_ref[1], 100).tolist()
            y_coords = np.arange(y_ref[0], y_ref[1], 100).tolist()
            iso_fair_vals = [(1 - adj) * 0.1] * len(x_coords)
            all_iso_fair_x_coords += x_coords
            all_iso_fair_y_coords += y_coords
            all_iso_fair_values += iso_fair_vals

    if fig_name:
        dump_vectors_to_csv(
            str(fig_name) + '_iso_fairs.csv', all_iso_fair_x_coords,
            all_iso_fair_y_coords, all_iso_fair_values)

    # iso-efficiency
    centre_of_efficiency = [1, 1]
    thetas = np.linspace(np.pi, 1.5 * np.pi, 100)
    all_iso_efficient_x_coords = []
    all_iso_efficient_y_coords = []
    all_iso_efficient_values = []

    for rad in np.arange(least_count, 1.5, least_count):
        points_in_iso_curve = [[
            centre_of_efficiency[0] + rad * np.cos(theta),
            centre_of_efficiency[1] + rad * np.sin(theta)
        ] for theta in thetas]
        x_ref = [elem[0] for elem in points_in_iso_curve]
        y_ref = [elem[1] for elem in points_in_iso_curve]
        plot_obj.plot(x_ref,
                      y_ref,
                      marker='.',
                      markersize=0.1,
                      ls='--',
                      lw=(1.5 - rad) * 0.1,
                      color='brown')
        all_iso_efficient_x_coords += x_ref
        all_iso_efficient_y_coords += y_ref
        all_iso_efficient_values += [(1.5 - rad) * 0.1] * len(x_ref)

    if fig_name:
        dump_vectors_to_csv(
            str(fig_name) + '_iso_efficients.csv', all_iso_efficient_x_coords,
            all_iso_efficient_y_coords, all_iso_efficient_values)


def plot_groupwise_outreach_histogram(reach_per_group_per_run,
                                      num_bins,
                                      return_only_bins=False,
                                      fig_name=None):
    bins = np.linspace(0, 1, num_bins + 1)
    plt.rcParams["figure.figsize"] = [4.50, 4.50]
    plt.rcParams["figure.autolayout"] = True

    _, ax = plt.subplots()
    hist_in_bins = ax.hist2d(reach_per_group_per_run[0],
                             reach_per_group_per_run[1],
                             cmap="Greens",
                             bins=bins,
                             range=[[0, 1], [0, 1]])
    hist_in_bins = hist_in_bins[0:-1]

    if fig_name:
        dump_vectors_to_csv(
            str(fig_name) + '_outreach.csv',
            convert_2d_matrix_to_vector_features(hist_in_bins[0]))

    ax.set_title('2D Histogram of groupwise normalized outreach')
    ax.set_xlabel("Fraction of Outreach in Group 0")
    ax.set_ylabel("Fraction of Outreach in Group 1")

    plot_2d_iso_curves(plot_obj=ax, least_count=0.2, fig_name=fig_name)

    if fig_name:
        plt.savefig(str(fig_name) + '_outreach.png')
        plt.close()

    if not return_only_bins:
        plt.show()

    plt.close()
    return hist_in_bins


def plot_2d_hist_binnings(binnings, bin_edges, rot=True, fig_name=None):
    # whatever be the granularity, ticks are on 1/10th scale
    num_ticks_plot = 10
    np_arr_binnings = np.array(binnings)
    fig, ax = plt.subplots()
    ax.set_xlabel("Fraction of Outreach in Group 0")
    ax.set_ylabel("Fraction of Outreach in Group 1")
    # TODO(schowdhary): see why this works differently for 10 bins
    arr_labels = ['%.2f' % (elem + 0.00) for elem in bin_edges]
    arr_labels_subsampled = [
        arr_label if
        (i % (np_arr_binnings.shape[0] / num_ticks_plot)) == 0 else ""
        for i, arr_label in enumerate(arr_labels)
    ]

    if rot:
        np_arr_binnings = np.rot90(np_arr_binnings, k=1, axes=(0, 1))
    plt.xticks(ticks=list(range(np_arr_binnings.shape[0])),
               labels=arr_labels_subsampled[:-1])
    arr_labels_subsampled.reverse()
    plt.yticks(ticks=list(range(np_arr_binnings.shape[1])),
               labels=arr_labels_subsampled[1:])

    plt.imshow(np_arr_binnings, cmap='Greens', interpolation='nearest')

    if fig_name:
        plt.savefig(str(fig_name) + '_2d_hist.png')

    else:
        plt.show()


def plot_2d_cost_matrix(mat, fig_name=None):
    scale = int(np.log10(mat.shape[0]))
    fig, ax = plt.subplots()
    ax.set_xlabel(
        "Row-major Support Indices (top-left to bottom-right), Target Dist.",
        fontsize=10 * scale)
    ax.set_ylabel("Row-major Support Indices, Source Dist.",
                  fontsize=10 * scale)
    ax.figure.set_figheight(5 * scale)
    ax.figure.set_figwidth(5 * scale)
    plt.imshow(mat, cmap='Greens', interpolation='nearest')

    if fig_name:
        plt.savefig(str(fig_name) + '_2d_cost.png')

    else:
        plt.show()


# plots based on propagation.propagate
def plot_seedset_joint_dist(seedset,
                            time_horizon_factor=8,
                            bin_size=100,
                            *args,
                            **kwargs):
    """Plots discretized joint outreach distribution across groups

    Args:
        seedset (list): seedset that causes joint outreach
        time_horizon_factor (int, optional): reduction factor for the
          time-horizon until which info propagation takes place. Defaults to 8.
        bin_size (int, optional): # discretized units per group in the
          Outreach Distribution support space. Defaults to 100.
    """
    assert "G" in kwargs.keys()
    assert "node_feats" in kwargs.keys()

    kwargs_with_fig_name_opt = deepcopy(kwargs)
    if "fig_name" in kwargs.keys():
        del kwargs["fig_name"]
    else:
        kwargs_with_fig_name_opt["fig_name"] = None

    _, _, runs_group_info = repeated_propagate(
        propagate,
        seedset=seedset,
        time_horizon=get_largest_graph_diameter(kwargs["G"]) //
        time_horizon_factor,
        *args,
        **kwargs)

    reach_per_group_per_run = calculate_groupwise_dist(runs_group_info,
                                                       kwargs["node_feats"])
    _ = plot_groupwise_outreach_histogram(
        reach_per_group_per_run,
        bin_size,
        return_only_bins=False,
        fig_name=kwargs_with_fig_name_opt["fig_name"])


# plots based on propagation.propagate
def plot_several_seedsets_joint_dist(labelled_seedsets,
                                     time_horizon_factor=8,
                                     bin_size=100,
                                     *args,
                                     **kwargs):
    """same as plot_seedset_joint_dist, but repeatedly for several
      labelled_seedsets, finally to be plotted on the same support space.
    """
    assert "G" in kwargs.keys()
    assert "node_feats" in kwargs.keys()

    kwargs_with_fig_name_opt = deepcopy(kwargs)
    if "fig_name" in kwargs.keys():
        del kwargs["fig_name"]
    else:
        kwargs_with_fig_name_opt["fig_name"] = None

    labelled_runs_group_infos = {
        label:
        repeated_propagate(
            propagate,
            seedset=seedset,
            time_horizon=get_largest_graph_diameter(kwargs["G"]) //
            time_horizon_factor,
            *args,
            **kwargs)[2]
        for label, seedset in labelled_seedsets.items()
    }

    labelled_reach_per_group_per_runs = {
        label: calculate_groupwise_dist(runs_group_info, kwargs["node_feats"])
        for label, runs_group_info in labelled_runs_group_infos.items()
    }

    _ = plot_several_groupwise_outreach_histograms(
        labelled_reach_per_group_per_runs,
        bin_size,
        fig_name=kwargs_with_fig_name_opt["fig_name"])


def plot_several_groupwise_outreach_histograms(
        labelled_reach_per_group_per_runs, num_bins, fig_name=None):
    bins = np.linspace(0, 1, num_bins + 1)
    num_entries = len(labelled_reach_per_group_per_runs.keys())
    plt.rcParams["figure.figsize"] = [4.50 * num_entries, 3.00 * num_entries]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots()

    for idx, (label, reach_per_group_per_run) in enumerate(
            labelled_reach_per_group_per_runs.items()):
        visible_label, cmap = label.split("-")
        hist = ax.hist2d(reach_per_group_per_run[0],
                         reach_per_group_per_run[1],
                         cmap=cmap,
                         bins=bins,
                         range=[[0, 1], [0, 1]],
                         cmin=1,
                         label=visible_label,
                         alpha=1.0 - (idx * 1.0 / (num_entries - 1)) / 1.8)
        cbar = fig.colorbar(hist[3], ax=ax)
        cbar.ax.set_ylabel(visible_label, rotation=270)

    ax.set_title('2D Histogram of groupwise normalized outreach')
    ax.set_xlabel("Fraction of Outreach in Group 0")
    ax.set_ylabel("Fraction of Outreach in Group 1")

    plot_2d_iso_curves(plot_obj=ax, least_count=0.2)

    if fig_name:
        plt.savefig(str(fig_name) + '_outreach.png')
        plt.close()

    # plt.close()

    return
