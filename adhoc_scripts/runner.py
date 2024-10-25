#! /usr/bin/env python

# TASK #####################################

# Generate results for,
# 1. geometry + 0 fairness (degree, greedy),
# 2. geometry + rand. fariness (avg of different seedsets), << seems misfit for the problem
# 3. geometry + 1-NN hard fairness (degree, greedy)
# 4. fairness optimization from other papers as baseline? << only when it runs
# 5. geometry + fairness from OT (degree, greedy)
# 6. 3-way comparison of joint outreach

# Notes #####################################
# 1. Python objects named `ana*` denote heuristic baselines presented by
#  [Stoica et. al.](https://dl.acm.org/doi/10.1145/3366423.3380275),
#  denoted as `hrt_d` and `hrt_g` in our work

# GLOBAL #####################################
RUNNER_CONFIG = {
    "RUN_ID": 0,
    "ds_name": "instagram_ds",
    "EDGE_PROB": 0.1,
    "NUM_SEEDS": 10,
    "REALIZATIONS": 1000,
    "BIN_SIZE": 100,
    "G_prop_max_diam": 13,
    "rel_fairness": 4,
    "rel_efficiency": 1,
    # EXPLOIT_TO_EXPLORE_RATIO = <some-val>,
    # SHALLOW_LOOKAHEAD_HORIZON_FACTOR = <some-val>,
    "NUM_OPT_ITRS": 1000
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

from utils.data_utils import dump_config_to_json, dump_obj

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

from early_seed_selection.node_centrality import \
    greedy_node_reach as baseline_greedy
from early_seed_selection.node_centrality import node_degree as baseline_degree
from metrics.ot_metrics import seedset_reach_transport_score
from pipeline import social_experiment as baseline_exp
from plots.plot_2d import plot_seedset_joint_dist
from propagation.propagate import propagate
from utils.data_utils import load_graph_from_edgelist, load_graph_node_features
from utils.graph_utils import log_social_graph_data_summary
from utils.housekeeping_utils import hk_init
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
log_social_graph_data_summary(G, node_feats)

ot_refs = prepare_ot_references(
    BIN_SIZE,
    rel_fairness=RUNNER_CONFIG["rel_fairness"],
    rel_efficiency=RUNNER_CONFIG["rel_efficiency"])  # <<<

_, g_logger = hk_init()

# TASK #1 #####################################

# degree

custom_input = {
    "edgelist_file": PROJECT_ROOT + '/data/' + ds_name + '/graph_edges.csv',
    "node_feat_file":
    PROJECT_ROOT + '/data/' + ds_name + '/graph_node_features.csv',
    "EDGE_PROB": EDGE_PROB,
    "NUM_SEEDS": NUM_SEEDS,
    "REALIZATIONS": REALIZATIONS,
    "BIN_SIZE": BIN_SIZE
}

custom_pipeline = {
    "seed_selector":
    baseline_degree.propose_seedset,
    "propagate":
    propagate,
    "target_dist":
    create_ideal_fair_and_efficient_2d_target_dist(custom_input["BIN_SIZE"]),
    "plot_graph":
    False
}

out = baseline_exp.run_experiment(input=custom_input,
                                  pipeline_mod=custom_pipeline)

baseline_degree_seedset = baseline_degree.propose_seedset(
    G, node_feats, NUM_SEEDS, EDGE_PROB, num_realizations=REALIZATIONS)

dump_obj(baseline_degree_seedset,
         PROJECT_ROOT + "/experiments/baseline_degree_seedset.pkl")

plot_seedset_joint_dist(baseline_degree_seedset,
                        time_horizon_factor=1,
                        bin_size=BIN_SIZE,
                        G=G,
                        node_feats=node_feats,
                        edge_prob=EDGE_PROB,
                        realizations=REALIZATIONS,
                        fig_name=PROJECT_ROOT + "/experiments/baseline_degree")

g_logger.info("Task #1.1 Baseline Degree seedset: %s",
              str(baseline_degree_seedset))
g_logger.info("Task #1.1 Baseline Degree OT Score: %f", out["ot_score"])

# greedy

custom_pipeline["seed_selector"] = baseline_greedy.propose_seedset

out = baseline_exp.run_experiment(input=custom_input,
                                  pipeline_mod=custom_pipeline)

baseline_greedy_seedset = baseline_greedy.propose_seedset(
    G, node_feats, NUM_SEEDS, EDGE_PROB, num_realizations=REALIZATIONS)

dump_obj(baseline_greedy_seedset,
         PROJECT_ROOT + "/experiments/baseline_greedy_seedset.pkl")

plot_seedset_joint_dist(baseline_greedy_seedset,
                        time_horizon_factor=1,
                        bin_size=BIN_SIZE,
                        G=G,
                        node_feats=node_feats,
                        edge_prob=EDGE_PROB,
                        realizations=REALIZATIONS,
                        fig_name=PROJECT_ROOT + "/experiments/baseline_greedy")

g_logger.info("Task #1.2 Baseline Greedy seedset: %s",
              str(baseline_greedy_seedset))
g_logger.info("Task #1.2 Baseline Greedy OT Score: %f", out["ot_score"])

g_logger.info("Task #1 complete!")

# Task #3 #####################################

from early_seed_selection.fairness_aware import \
    greedy_node_reach as ana_fair_greedy
from early_seed_selection.fairness_aware import \
    node_degree as ana_fair_node_degree

# degree

ana_fair_degree_seedset = ana_fair_node_degree.propose_seedset(
    G,
    node_feats,
    NUM_SEEDS,
    EDGE_PROB,
    num_realizations=REALIZATIONS,
    min_fair_ratio=0.4,
    max_fair_ratio=0.6)

g_logger.info("Task #3.1 Ana's Fair Degree seedset: %s",
              str(ana_fair_degree_seedset))

dump_obj(ana_fair_degree_seedset,
         PROJECT_ROOT + "/experiments/heuristic_degree_seedset.pkl")

plot_seedset_joint_dist(ana_fair_degree_seedset,
                        time_horizon_factor=1,
                        bin_size=BIN_SIZE,
                        G=G,
                        node_feats=node_feats,
                        edge_prob=EDGE_PROB,
                        realizations=REALIZATIONS,
                        fig_name=PROJECT_ROOT + "/experiments/ana_fair_degree")

ana_fair_degree_ot_score = seedset_reach_transport_score(
    ana_fair_degree_seedset,
    G,
    node_feats,
    EDGE_PROB,
    ot_refs=ot_refs,
    time_horizon_scale=1,
    realizations=REALIZATIONS,
    bin_size=BIN_SIZE)

g_logger.info("Task #3.1 Ana's Fair Degree OT score: %f",
              ana_fair_degree_ot_score)

dump_obj(ana_fair_degree_ot_score,
         PROJECT_ROOT + "/experiments/heuristic_degree_ot_score.pkl")

# greedy

ana_fair_greedy_seedset = ana_fair_greedy.propose_seedset(
    G,
    node_feats,
    NUM_SEEDS,
    EDGE_PROB,
    num_realizations=REALIZATIONS,
    min_fair_ratio=0.4,
    max_fair_ratio=0.6)

g_logger.info("Task #3.2 Ana's Fair Greedy seedset: %s",
              str(ana_fair_greedy_seedset))

dump_obj(ana_fair_greedy_seedset,
         PROJECT_ROOT + "/experiments/heuristic_greedy_seedset.pkl")

plot_seedset_joint_dist(ana_fair_greedy_seedset,
                        time_horizon_factor=1,
                        bin_size=BIN_SIZE,
                        G=G,
                        node_feats=node_feats,
                        edge_prob=EDGE_PROB,
                        realizations=REALIZATIONS,
                        fig_name=PROJECT_ROOT + "/experiments/ana_fair_greedy")

ana_fair_greedy_ot_score = seedset_reach_transport_score(
    ana_fair_greedy_seedset,
    G,
    node_feats,
    EDGE_PROB,
    ot_refs=ot_refs,
    time_horizon_scale=1,
    realizations=REALIZATIONS,
    bin_size=BIN_SIZE)

g_logger.info("Task #3.2 Ana's Fair Greedy OT score: %f",
              ana_fair_greedy_ot_score)

dump_obj(ana_fair_greedy_ot_score,
         PROJECT_ROOT + "/experiments/heuristic_greedy_ot_score.pkl")

g_logger.info("Task #3 complete!")

# TASK #5 #####################################

from copy import deepcopy

import numpy as np

from optimize.seedset_opt import optimize_target_seedset

# EXPLOIT_TO_EXPLORE_RATIO = <some-val>  # <<<
# SHALLOW_LOOKAHEAD_HORIZON_FACTOR = <some-val>  # <<<
NUM_OPT_ITRS = RUNNER_CONFIG["NUM_OPT_ITRS"]  # <<<

# degree

itr_target_seedset = deepcopy(baseline_degree_seedset)
improvements = [(baseline_degree_seedset,
                 seedset_reach_transport_score(baseline_degree_seedset,
                                               G,
                                               node_feats,
                                               EDGE_PROB,
                                               ot_refs=ot_refs,
                                               time_horizon_scale=1,
                                               realizations=REALIZATIONS,
                                               bin_size=BIN_SIZE))]
opt_itrs = NUM_OPT_ITRS

for itr in range(opt_itrs):
    ret = optimize_target_seedset(itr_target_seedset,
                                  G=G,
                                  node_feats=node_feats,
                                  edge_prob=EDGE_PROB,
                                  ot_refs=ot_refs,
                                  realizations=REALIZATIONS,
                                  bin_size=BIN_SIZE)

    itr_target_seedset = ret[0][0]

    improvements.append(deepcopy((ret[0][0], ret[0][1])))

    if itr % 100 == 0:
        min_itr_idx = np.argmin(np.array([elem[1] for elem in improvements]))
        g_logger.info(
            "By iter %d, min OT score improvement at iter %d, and is %f", itr,
            min_itr_idx, improvements[min_itr_idx][1])

least_impr = deepcopy(improvements[0])
least_same_size_impr = deepcopy(least_impr)  # distinct seedset init

for iter_improvement in improvements:
    curr_seedset = iter_improvement[0]
    curr_ot_score = iter_improvement[1]

    if curr_ot_score < least_impr[1]:
        least_impr = deepcopy((curr_seedset, curr_ot_score))

        if len(set(curr_seedset)) == len(least_same_size_impr):
            least_same_size_impr = deepcopy((curr_seedset, curr_ot_score))

g_logger.info("Task #5.1 Degree Set %s --> %s, and %f --> %f",
              str(baseline_degree_seedset), str(least_impr[0]),
              improvements[0][1], least_impr[1])

dump_obj(least_impr[0], PROJECT_ROOT + "/experiments/s3d_degree_seedset.pkl")
dump_obj(improvements[0][1],
         PROJECT_ROOT + "/experiments/baseline_degree_ot_score.pkl")
dump_obj(least_impr[1], PROJECT_ROOT + "/experiments/s3d_degree_ot_score.pkl")

s3d_degree_seedset = least_impr[0]

plot_seedset_joint_dist(s3d_degree_seedset,
                        time_horizon_factor=1,
                        bin_size=BIN_SIZE,
                        G=G,
                        node_feats=node_feats,
                        edge_prob=EDGE_PROB,
                        realizations=REALIZATIONS,
                        fig_name=PROJECT_ROOT + "/experiments/s3d_degree")

if len(least_impr[0]) != len(least_same_size_impr[0]):
    g_logger.warn(
        "Task #5.1 Degree Set min found at smaller seedset! Same size best improvement is as follows,"
    )
    g_logger.warn("Task #5.1 Same size Degree Set %s --> %s, and %f --> %f",
                  str(baseline_degree_seedset), str(least_same_size_impr[0]),
                  improvements[0][1], least_same_size_impr[1])

    s3d_same_size_degree_seedset = least_same_size_impr[0]

    plot_seedset_joint_dist(s3d_same_size_degree_seedset,
                            time_horizon_factor=1,
                            bin_size=BIN_SIZE,
                            G=G,
                            node_feats=node_feats,
                            edge_prob=EDGE_PROB,
                            realizations=REALIZATIONS,
                            fig_name=PROJECT_ROOT +
                            "/experiments/s3d_same_size_degree")

# greedy

itr_target_seedset = deepcopy(baseline_greedy_seedset)
improvements = [(baseline_greedy_seedset,
                 seedset_reach_transport_score(baseline_greedy_seedset,
                                               G,
                                               node_feats,
                                               EDGE_PROB,
                                               ot_refs=ot_refs,
                                               time_horizon_scale=1,
                                               realizations=REALIZATIONS,
                                               bin_size=BIN_SIZE))]
opt_itrs = NUM_OPT_ITRS

for itr in range(opt_itrs):
    ret = optimize_target_seedset(itr_target_seedset,
                                  G=G,
                                  node_feats=node_feats,
                                  edge_prob=EDGE_PROB,
                                  ot_refs=ot_refs,
                                  realizations=REALIZATIONS,
                                  bin_size=BIN_SIZE)

    itr_target_seedset = ret[0][0]

    improvements.append(deepcopy((ret[0][0], ret[0][1])))

    if itr % 100 == 0:
        min_itr_idx = np.argmin(np.array([elem[1] for elem in improvements]))
        g_logger.info(
            "By iter %d, min OT score improvement at iter %d, and is %f", itr,
            min_itr_idx, improvements[min_itr_idx][1])

least_impr = deepcopy(improvements[0])
least_same_size_impr = deepcopy(least_impr)  # distinct seedset init

for iter_improvement in improvements:
    curr_seedset = iter_improvement[0]
    curr_ot_score = iter_improvement[1]

    if curr_ot_score < least_impr[1]:
        least_impr = deepcopy((curr_seedset, curr_ot_score))

        if len(set(curr_seedset)) == len(least_same_size_impr):
            least_same_size_impr = deepcopy((curr_seedset, curr_ot_score))

g_logger.info("Task #5.2 Greedy Set %s --> %s, and %f --> %f",
              str(baseline_greedy_seedset), str(least_impr[0]),
              improvements[0][1], least_impr[1])

dump_obj(least_impr[0], PROJECT_ROOT + "/experiments/s3d_greedy_seedset.pkl")
dump_obj(improvements[0][1],
         PROJECT_ROOT + "/experiments/baseline_greedy_ot_score.pkl")
dump_obj(least_impr[1], PROJECT_ROOT + "/experiments/s3d_greedy_ot_score.pkl")

s3d_greedy_seedset = least_impr[0]

plot_seedset_joint_dist(s3d_greedy_seedset,
                        time_horizon_factor=1,
                        bin_size=BIN_SIZE,
                        G=G,
                        node_feats=node_feats,
                        edge_prob=EDGE_PROB,
                        realizations=REALIZATIONS,
                        fig_name=PROJECT_ROOT + "/experiments/s3d_greedy")

if len(least_impr[0]) != len(least_same_size_impr[0]):
    g_logger.warn(
        "Task #5.2 Greedy Set min found at smaller seedset! Same size best improvement is as follows,"
    )
    g_logger.warn("Task #5.2 Same size Greedy Set %s --> %s, and %f --> %f",
                  str(baseline_greedy_seedset), str(least_same_size_impr[0]),
                  improvements[0][1], least_same_size_impr[1])

    s3d_same_size_greedy_seedset = least_same_size_impr[0]

    plot_seedset_joint_dist(s3d_same_size_greedy_seedset,
                            time_horizon_factor=1,
                            bin_size=BIN_SIZE,
                            G=G,
                            node_feats=node_feats,
                            edge_prob=EDGE_PROB,
                            realizations=REALIZATIONS,
                            fig_name=PROJECT_ROOT +
                            "/experiments/s3d_same_size_greedy")

g_logger.info("Task #5 complete!")

# TASK #6 #####################################

from itertools import combinations

from plots.plot_2d import plot_several_seedsets_joint_dist

# degree

labelled_degree_seedsets = [("s3d_degree", s3d_degree_seedset),
                            ("heuristic_nearby_fair_degree",
                             ana_fair_degree_seedset),
                            ("baseline_degree", baseline_degree_seedset)]

for seedset_info_pair in combinations(labelled_degree_seedsets, 2):
    labelled_seedsets = {
        seedset_info_pair[0][0] + "-Greens": seedset_info_pair[0][1],
        seedset_info_pair[1][0] + "-YlOrRd": seedset_info_pair[1][1]
    }
    custom_fig_name = seedset_info_pair[0][0] + "-vs-" + seedset_info_pair[1][0]
    plot_several_seedsets_joint_dist(labelled_seedsets,
                                     time_horizon_factor=1,
                                     bin_size=BIN_SIZE,
                                     G=G,
                                     node_feats=node_feats,
                                     edge_prob=EDGE_PROB,
                                     realizations=REALIZATIONS,
                                     fig_name=PROJECT_ROOT + "/experiments/" +
                                     custom_fig_name)

# greedy

labelled_greedy_seedsets = [("s3d_greedy", s3d_greedy_seedset),
                            ("heuristic_nearby_fair_greedy",
                             ana_fair_greedy_seedset),
                            ("baseline_greedy", baseline_greedy_seedset)]

for seedset_info_pair in combinations(labelled_greedy_seedsets, 2):
    labelled_seedsets = {
        seedset_info_pair[0][0] + "-Greens": seedset_info_pair[0][1],
        seedset_info_pair[1][0] + "-YlOrRd": seedset_info_pair[1][1]
    }
    custom_fig_name = seedset_info_pair[0][0] + "-vs-" + seedset_info_pair[1][0]
    plot_several_seedsets_joint_dist(labelled_seedsets,
                                     time_horizon_factor=1,
                                     bin_size=BIN_SIZE,
                                     G=G,
                                     node_feats=node_feats,
                                     edge_prob=EDGE_PROB,
                                     realizations=REALIZATIONS,
                                     fig_name=PROJECT_ROOT + "/experiments/" +
                                     custom_fig_name)

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

fairness_efficiency_scatter_for_seedsets(G,
                                         node_feats,
                                         EDGE_PROB,
                                         all_short_labelled_seedsets,
                                         REALIZATIONS,
                                         BIN_SIZE,
                                         logger=g_logger,
                                         fig_name=PROJECT_ROOT +
                                         "/experiments/seedset_algos")

g_logger.info("Task #6 complete!")
