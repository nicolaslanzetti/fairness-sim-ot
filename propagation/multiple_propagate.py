# repeated aggregation of any Information Propagation algorithm

from utils.graph_utils import get_largest_graph_diameter
from utils.housekeeping_utils import concat_dicts, hk_init

PROJECT_ROOT, global_logger = hk_init()


def repeated_propagate(prop_routine, realizations=1000, *args, **kwargs):
    """Repeat given prop_routine for realizations number of times and return
      information across the runs, grouped per 1. node, 2. seedset, 3. group

    Args:
        prop_routine (func): function inline with the interface
          from propagation.propagate
        realizations (int, optional): # realizations of the random graph on
          each of which prop_routine runs. Defaults to 1000.

    Returns:
        tuple: aggregated multi-propagation information across the runs,
          grouped per 1. node, 2. seedset, 3. group
    """
    assert realizations > 0

    runs_prop_info = {}
    runs_seed_info = {}
    runs_group_info = {}

    if "logger" in kwargs.keys():
        logger = kwargs["logger"]
    else:
        logger = global_logger

    # should be the nature of every propagation algo (requirement)
    assert "G" in kwargs.keys() and "time_horizon" in kwargs.keys()
    G = kwargs["G"]
    time_horizon = kwargs["time_horizon"]

    if time_horizon is not None:
        logger.info("Custom time_horizon set for propagation, %d",
                    time_horizon)
    else:
        num_edges = G.number_of_edges()
        logger.info("Since time_horizon isn't set, set it to num_edges, %d",
                    num_edges)
        kwargs["time_horizon"] = num_edges

    out_runs = [prop_routine(*args, **kwargs) for _ in range(realizations)]
    nodes = list(out_runs[0][0].keys())
    seeds = list(out_runs[0][1].keys())
    groups = list(out_runs[0][2].keys())

    for node in nodes:
        runs_prop_info[node] = []

    for seed in seeds:
        runs_seed_info[seed] = []

    for group in groups:
        runs_group_info[group] = []

    for out_run in out_runs:
        runs_prop_info = concat_dicts(
            runs_prop_info, out_run[0])  # node X runs x node_info_struct
        runs_seed_info = concat_dicts(
            runs_seed_info, out_run[1])  # seed X runs X seed_info_struct
        runs_group_info = concat_dicts(
            runs_group_info, out_run[2])  # group X runs X group_info_struct

    return runs_prop_info, runs_seed_info, runs_group_info


def get_nbrs_helper(seedset,
                    multi_prop_algo,
                    time_horizon_scale=1,
                    no_target=True,
                    *args,
                    **kwargs):
    if "seedset" in kwargs.keys():
        del kwargs["seedset"]
    kwargs["seedset"] = seedset

    assert "G" in kwargs.keys()
    if "time_horizon" in kwargs.keys():
        del kwargs["time_horizon"]
    kwargs["time_horizon"] = get_largest_graph_diameter(
        kwargs["G"]) // time_horizon_scale

    run_prop_info, _, _ = multi_prop_algo(*args, **kwargs)

    target_seedset_nbrs = {}

    # implicit elem[0] boolean conversion to integer sum
    target_seedset_nbrs = {
        node: sum([elem[0] for elem in hist])
        for node, hist in run_prop_info.items()
    }

    if no_target:
        for target_seed in seedset:
            assert target_seedset_nbrs[target_seed] == kwargs["realizations"]
            target_seedset_nbrs[target_seed] = 0

    return target_seedset_nbrs


def get_seed_nbrs(target_seed, *args, **kwargs):
    """get the neighbour nodes reachable from the given target_seed
    """
    lone_seedset = [target_seed]
    if "time_horizon_scale" not in kwargs.keys():
        time_horizon_scale = 8
    else:
        time_horizon_scale = kwargs["time_horizon_scale"]
        del kwargs["time_horizon_scale"]

    if "no_target" not in kwargs.keys():
        no_target = True
    else:
        no_target = kwargs["no_target"]
        del kwargs["no_target"]

    return get_nbrs_helper(lone_seedset,
                           time_horizon_scale=time_horizon_scale,
                           no_target=no_target,
                           *args,
                           **kwargs)


def get_seedset_nbrs(seedset, *args, **kwargs):
    """get the neighbour nodes reachable from the given seedset
    """
    if "time_horizon_scale" not in kwargs.keys():
        time_horizon_scale = 1
    else:
        time_horizon_scale = kwargs["time_horizon_scale"]
        del kwargs["time_horizon_scale"]

    if "no_target" not in kwargs.keys():
        no_target = False
    else:
        no_target = kwargs["no_target"]
        del kwargs["no_target"]

    return get_nbrs_helper(seedset,
                           time_horizon_scale=time_horizon_scale,
                           no_target=no_target,
                           *args,
                           **kwargs)
