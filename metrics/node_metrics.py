from utils.housekeeping_utils import hk_init

PROJECT_ROOT, global_logger = hk_init()


# based on propagation.propagate.propagate
def get_propagation_efficiency(runs_prop_info):
    """returns effective reach of each of the nodes (from seedset) in the
      social graph
    """
    # runs_prop_info is a dict per node with a value as per
    #  propagation.propagate.propagate

    nodes = list(runs_prop_info.keys())
    num_nodes = len(nodes)
    num_realizations = len(runs_prop_info[nodes[0]])

    node_reach_efficiency = sum([
        sum([elem[0] for elem in node_info])
        for _, node_info in runs_prop_info.items()
    ]) * 1.0 / (num_nodes * num_realizations)

    return node_reach_efficiency
