#! /usr/bin/env python

# TASK #####################################

# Generate results for,
# 1. statistical significance tests

# Notes #####################################
# 1. Python objects named `ana*` denote heuristic baselines presented by [Stoica et. al.](https://dl.acm.org/doi/10.1145/3366423.3380275), denoted as `hrt_d` and `hrt_g` in our work

# GLOBAL #####################################
RUNNER_CONFIG = {
    "RUN_ID":
    "stat_test",
    "ds_name":
    "ind_villages_ds",
    "EDGE_PROB":
    0.1,
    "NUM_SEEDS":
    2,
    "REALIZATIONS":
    1000,
    "BIN_SIZE":
    100,
    "G_prop_max_diam":
    13,
    "rel_fairness":
    1,
    "rel_efficiency":
    1,
    # EXPLOIT_TO_EXPLORE_RATIO = <some-val>,
    # SHALLOW_LOOKAHEAD_HORIZON_FACTOR = <some-val>,
    "NUM_OPT_ITRS":
    1000,
    "STAT_TEST_SAMPLES":
    100,
    "experiment_path":
    "experiments/paper_exps/ind_villages_tradeoff_shift_no_regularize/new"
}

RUN_ID = RUNNER_CONFIG["RUN_ID"]  # <<<
# global config options
LOG_LEVEL = "INFO"  # only DEBUG and INFO supported, anything else leads to CRITICAL
LOG_FILE = "./run_" + str(
    RUN_ID) + ".log"  # path relative to current working directory

# global imports
import os

PROJECT_ROOT = os.getcwd()
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
print("Project Root:", PROJECT_ROOT)

os.environ["LOG_LEVEL"] = LOG_LEVEL
print("Log level for libraries:", LOG_LEVEL)

os.environ["LOG_FILE"] = PROJECT_ROOT + "/" + LOG_FILE
print("Library logs are present at", os.environ["LOG_FILE"])

from utils.data_utils import (calculate_sample_mu_sigma, dump_config_to_json,
                              dump_obj, dump_vectors_to_csv, load_to_obj)

os.system("echo \"\" >" + "run_" + str(RUNNER_CONFIG["RUN_ID"]) + ".log; rm " +
          PROJECT_ROOT + "/experiments/runner_config.json; rm -rf " +
          PROJECT_ROOT + "/experiments/exp_*; rm -rf " + PROJECT_ROOT +
          "/experiments/*.png; rm -rf " + PROJECT_ROOT +
          "/experiments/*.pkl; rm -rf " + PROJECT_ROOT + "/experiments/*.csv;")

dump_config_to_json(RUNNER_CONFIG,
                    filename=PROJECT_ROOT + "/experiments/runner_config.json")

# DATA-LOADING #####################################

ds_name = RUNNER_CONFIG["ds_name"]  # <<<
EDGE_PROB = RUNNER_CONFIG["EDGE_PROB"]  # <<<
NUM_SEEDS = RUNNER_CONFIG["NUM_SEEDS"]  # <<<
REALIZATIONS = RUNNER_CONFIG["REALIZATIONS"]  # <<<
BIN_SIZE = RUNNER_CONFIG["BIN_SIZE"]  # <<<
EXP_ROOT = os.path.join(PROJECT_ROOT, RUNNER_CONFIG["experiment_path"])

from early_seed_selection.node_centrality import \
    greedy_node_reach as baseline_greedy
from early_seed_selection.node_centrality import node_degree as baseline_degree
from metrics.ot_metrics import seedset_reach_transport_score
from pipeline import social_experiment as baseline_exp
from plots.plot_2d import plot_seedset_joint_dist
from propagation.propagate import propagate
from utils.data_utils import load_graph_from_edgelist, load_graph_node_features
from utils.graph_utils import log_social_graph_data_summary
from utils.housekeeping_utils import concat_dicts, hk_init
from utils.ot_utils import (create_ideal_fair_and_efficient_2d_target_dist,
                            prepare_ot_references)

ds_string = ds_name
G = load_graph_from_edgelist(
    PROJECT_ROOT + "/data/" + ds_string +
    "/graph_edges.csv")  # nodes are string, edges are between strings
G.graph["prop_max_diam"] = RUNNER_CONFIG["G_prop_max_diam"]  # <<<
node_feats = load_graph_node_features(
    PROJECT_ROOT + "/data/" + ds_string +
    "/graph_node_features.csv")  # node features are ints for now
assert set(G.nodes()) == set(node_feats.keys())

# log_social_graph_data_summary(G, node_feats)

# ot_refs = prepare_ot_references(
#     BIN_SIZE,
#     rel_fairness=RUNNER_CONFIG["rel_fairness"],
#     rel_efficiency=RUNNER_CONFIG["rel_efficiency"])  # <<<

_, g_logger = hk_init()

# TASK #1 #####################################

# degree

baseline_degree_seedset = load_to_obj(
    os.path.join(EXP_ROOT, "baseline_degree_seedset.pkl"))

g_logger.info("Task #1.1 Baseline Degree seedset: %s",
              str(baseline_degree_seedset))

# greedy

baseline_greedy_seedset = load_to_obj(
    os.path.join(EXP_ROOT, "baseline_greedy_seedset.pkl"))

g_logger.info("Task #1.2 Baseline Greedy seedset: %s",
              str(baseline_greedy_seedset))

g_logger.info("Task #1 complete!")

# Task #3 #####################################

from early_seed_selection.fairness_aware import \
    greedy_node_reach as ana_fair_greedy
from early_seed_selection.fairness_aware import \
    node_degree as ana_fair_node_degree

# degree

ana_fair_degree_seedset = load_to_obj(
    os.path.join(EXP_ROOT, "heuristic_degree_seedset.pkl"))

g_logger.info("Task #3.1 Ana's Fair Degree seedset: %s",
              str(ana_fair_degree_seedset))

# greedy

ana_fair_greedy_seedset = load_to_obj(
    os.path.join(EXP_ROOT, "heuristic_greedy_seedset.pkl"))

g_logger.info("Task #3.2 Ana's Fair Greedy seedset: %s",
              str(ana_fair_greedy_seedset))

g_logger.info("Task #3 complete!")

# TASK #5 #####################################

from copy import deepcopy

import numpy as np

from optimize.seedset_opt import optimize_target_seedset

s3d_degree_seedset = load_to_obj(
    os.path.join(EXP_ROOT, "s3d_degree_seedset.pkl"))

# greedy

s3d_greedy_seedset = load_to_obj(
    os.path.join(EXP_ROOT, "s3d_greedy_seedset.pkl"))

g_logger.info("Task #5 complete!")

# TASK #7 #####################################

from itertools import combinations

from plots.plot_2d import plot_several_seedsets_joint_dist

# degree

labelled_degree_seedsets = [("s3d_degree", s3d_degree_seedset),
                            ("heuristic_nearby_fair_degree",
                             ana_fair_degree_seedset),
                            ("baseline_degree", baseline_degree_seedset)]

# greedy

labelled_greedy_seedsets = [("s3d_greedy", s3d_greedy_seedset),
                            ("heuristic_nearby_fair_greedy",
                             ana_fair_greedy_seedset),
                            ("baseline_greedy", baseline_greedy_seedset)]

# all seedsets in an EF-plot
from plots.fairness_efficiency_space import \
    fairness_efficiency_scatter_for_seedsets

short_label_translate = {
    "s3d_degree": "s3d_d",
    "s3d_greedy": "s3d_g",
    "heuristic_nearby_fair_degree": "hrt_d",
    "heuristic_nearby_fair_greedy": "hrt_g",
    "baseline_degree": "bas_d",
    "baseline_greedy": "bas_g"
}

all_short_labelled_seedsets = {
    short_label_translate[k]: v
    for k, v in dict(labelled_degree_seedsets +
                     labelled_greedy_seedsets).items()
}

labelled_efficiency_for_seedsets = {
    k: []
    for k, _ in all_short_labelled_seedsets.items()
}
labelled_fairness_for_seedsets = {
    k: []
    for k, _ in all_short_labelled_seedsets.items()
}

NUM_STAT_TEST_ITRS = RUNNER_CONFIG["STAT_TEST_SAMPLES"]

for iter in range(NUM_STAT_TEST_ITRS):
    if iter % 10 == 0:
        g_logger.info("Iterations complete: %d", iter)
    eff_dict, fair_dict = fairness_efficiency_scatter_for_seedsets(
        G,
        node_feats,
        EDGE_PROB,
        all_short_labelled_seedsets,
        REALIZATIONS,
        BIN_SIZE,
        logger=g_logger)
    labelled_efficiency_for_seedsets = concat_dicts(
        labelled_efficiency_for_seedsets, eff_dict)
    labelled_fairness_for_seedsets = concat_dicts(
        labelled_fairness_for_seedsets, fair_dict)

mu_vecs = []
std_vecs = []
label_vecs = []

for k, v in labelled_efficiency_for_seedsets.items():
    mu, sigma = calculate_sample_mu_sigma(v)
    mu_vecs.append(mu)
    std_vecs.append(sigma)
    label_vecs.append("eff_" + k)

for k, v in labelled_fairness_for_seedsets.items():
    mu, sigma = calculate_sample_mu_sigma(v)
    mu_vecs.append(mu)
    std_vecs.append(sigma)
    label_vecs.append("fair_" + k)

dump_vectors_to_csv("seedset_algos_EF_error_bars.csv", mu_vecs, std_vecs,
                    label_vecs)

g_logger.info("Task #7 complete!")
