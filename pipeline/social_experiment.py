# A ready to use interface for a 1 shot dataset, seedset and propagation algo
#  selection to get metrics, OT scores, and Outreach Distribution plots
#  resulting from the setup.

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from metrics.group_metrics import calculate_groupwise_dist
from metrics.seed_metrics import calculate_seedset_performance
from plots.plot_2d import (plot_2d_cost_matrix, plot_2d_hist_binnings,
                           plot_groupwise_outreach_histogram)
from propagation.multiple_propagate import repeated_propagate
from utils.data_utils import load_graph_from_edgelist, load_graph_node_features
from utils.graph_utils import (draw_social_graph, get_largest_graph_diameter,
                               log_social_graph_data_summary)
from utils.housekeeping_utils import (generate_random_alphanumeric_string,
                                      get_local_logger, hk_init)
from utils.ot_utils import (calculate_emd_matrix,
                            create_fair_and_efficient_2d_cost_matrix,
                            create_fair_and_efficient_2d_dist,
                            unroll_dist_from_hist_bins)

PROJECT_ROOT, global_logger = hk_init()

DEFAULT_INPUT = {
    "edgelist_file": PROJECT_ROOT + '/data/sample_data/graph_edges.csv',
    "node_feat_file":
    PROJECT_ROOT + '/data/sample_data/graph_node_features.csv',
    "EDGE_PROB": 0.2,
    "NUM_SEEDS": 2,
    "REALIZATIONS": 1000,
    "BIN_SIZE": 10
}

DEFAULT_PIPELINE_MOD = {
    "seed_selector": None,
    "propagate": None,
    "target_dist": None,
    "plot_graph": False
}

DEFAULT_OUTPUT = {"ot_score": None, "group_metrics": {}}


def validate_input(input):
    assert set(DEFAULT_INPUT.keys()) <= set(input.keys())

    for key, val in input.items():
        if val is None:
            global_logger.error("Provide a legal value for input key, %s", key)

            return False

    return True


def validate_pipeline(pipeline):
    assert set(DEFAULT_PIPELINE_MOD.keys()) <= set(pipeline.keys())

    for key, val in pipeline.items():
        if val is None:
            global_logger.error(
                "Provide a legal value for pipeline_mod key, %s", key)

            return False

    return True


def dump_configuration(config_dict, file_path):
    with open(file_path, 'w') as f:
        json.dump(config_dict, f, indent=4)


def translate_modular_config_to_description(config_dict):
    description_config = {}

    for key, val in config_dict.items():
        if hasattr(val, "__module__"):
            description_config[key] = str(val.__module__) + "." + str(val)

    return description_config


def flush_pipeline(loc_loggers):
    for loc_logger in loc_loggers:
        assert len(loc_logger.handlers) == 1
        loc_logger.handlers[0].flush()
    for gl_handler in global_logger.handlers:
        gl_handler.flush()

    # close all implicitly opened figures of the experiment
    plt.close('all')


def run_experiment(input=DEFAULT_INPUT,
                   pipeline_mod=DEFAULT_PIPELINE_MOD,
                   output_dir=PROJECT_ROOT + "/experiments/"):
    """A ready to use interface for a 1 shot dataset, seedset and propagation
      algo selection to get metrics, OT scores, and Outreach Distribution
        plots resulting from the setup.
    """
    assert validate_input(input)
    run_hash = generate_random_alphanumeric_string(8)
    global_logger.info(
        "==================== <Running Experiment Hash %s"
        "====================", run_hash)

    exp_dir = os.path.join(output_dir, "exp_" + str(run_hash)) + "/"
    os.makedirs(exp_dir, exist_ok=True)

    local_logger = get_local_logger("exp_" + str(run_hash),
                                    exp_dir + "/run.log")

    G = load_graph_from_edgelist(input["edgelist_file"])
    node_feats = load_graph_node_features(input["node_feat_file"])
    G.graph[
        "prop_max_diam"] = 13  # hack to prevent long max diam calculation, ...
    # ... manual runs already have this
    log_social_graph_data_summary(G, node_feats, logger=local_logger)
    assert set(G.nodes()) == set(node_feats.keys())

    assert validate_pipeline(pipeline_mod)

    output = DEFAULT_OUTPUT

    # optional config
    if "FRAC_SEEDS" in input.keys():
        global_logger.info("Overriding NUM_SEEDS config with FRAC_SEEDS!")
        num_seeds = int(input["FRAC_SEEDS"] * G.number_of_nodes())
    else:
        num_seeds = input["NUM_SEEDS"]

    seedset = pipeline_mod["seed_selector"](G,
                                            node_feats,
                                            num_seeds,
                                            input["EDGE_PROB"],
                                            input["REALIZATIONS"],
                                            logger=local_logger)

    global_logger.info("Seedset selection complete with %d seeds!", num_seeds)
    local_logger.info("Seedset selection complete with %d seeds!", num_seeds)

    if pipeline_mod["plot_graph"]:
        # TODO(schowdhary): This routine takes 8+hrs Insta DS!
        draw_social_graph(G,
                          input["EDGE_PROB"],
                          seedset=seedset,
                          node_feats=node_feats,
                          plot_plain=True,
                          fig_name=exp_dir + str(run_hash))
        global_logger.info("Rendering the social network, complete!")

    runs_prop_info, runs_seed_info, runs_group_info = repeated_propagate(
        pipeline_mod["propagate"],
        input["REALIZATIONS"],
        G=G,
        node_feats=node_feats,
        edge_prob=input["EDGE_PROB"],
        seedset=seedset,
        time_horizon=get_largest_graph_diameter(G) // 1,
        logger=local_logger)
    global_logger.info("Propagation across multiple realizations, complete!")

    aggr_seed_perf = calculate_seedset_performance(runs_seed_info,
                                                   logger=local_logger)
    local_logger.info("Multi run seed performance, seed: [total_reach, "
                      "{group_id: [group_reach, time_taken], ...}]")
    local_logger.info(str(aggr_seed_perf))

    reach_per_group_per_run = calculate_groupwise_dist(runs_group_info,
                                                       node_feats,
                                                       logger=local_logger)
    for key, val in reach_per_group_per_run.items():
        output["group_metrics"][key] = np.mean(val)

    plot_hist_in_bins = plot_groupwise_outreach_histogram(
        reach_per_group_per_run, input["BIN_SIZE"], return_only_bins=True)

    bin_edges = plot_hist_in_bins[1]
    plot_2d_hist_binnings(plot_hist_in_bins[0],
                          bin_edges,
                          rot=True,
                          fig_name=exp_dir + str(run_hash) + "_social_graph")

    target_dist = pipeline_mod["target_dist"]
    plot_2d_hist_binnings(target_dist[0],
                          bin_edges,
                          rot=False,
                          fig_name=exp_dir + str(run_hash) + "_target")

    plot_2d_hist_binnings(create_fair_and_efficient_2d_dist(
        input["BIN_SIZE"], True, 1.0, True, 1.0)[0],
                          bin_edges,
                          rot=False,
                          fig_name=exp_dir + str(run_hash) + "_profit_region")

    M, scale_k = create_fair_and_efficient_2d_cost_matrix(
        input["BIN_SIZE"], True, 1.0, True, 1.0)
    score, res_dict = calculate_emd_matrix(
        unroll_dist_from_hist_bins(plot_hist_in_bins[0]), target_dist[2], M,
        scale_k)
    plot_2d_cost_matrix(res_dict["G"], fig_name=exp_dir + str(run_hash))

    local_logger.info("Final Transport Score: %f", score)
    output["ot_score"] = score
    global_logger.info(
        "==================== Running Experiment Hash %s>"
        "====================", run_hash)

    dump_configuration(input, exp_dir + "/input.json")
    dump_configuration(translate_modular_config_to_description(pipeline_mod),
                       exp_dir + "/pipeline.json")

    flush_pipeline(loc_loggers=[local_logger])

    return output
