# Implements S3D on beta-fairness from our paper. Based on initial seedset
#  selection from baselines, bas_d/bas_g, S3D converges to
#  s3d_d/s3d_g respectively

import random
from copy import deepcopy

import numpy as np

from metrics.ot_metrics import seedset_reach_transport_score
from propagation.multiple_propagate import (get_seed_nbrs, get_seedset_nbrs,
                                            repeated_propagate)
from propagation.propagate import propagate
from utils.housekeeping_utils import hk_init

random.seed(42)
PROJECT_ROOT, global_logger = hk_init()

# default params can be changed to try different flavours
EXPLOIT_TO_EXPLORE_RATIO = 1.3
NON_ACCEPTANCE_RETENTION_PROB = 0.95
SHALLOW_LOOKAHEAD_HORIZON_FACTOR = 4


def adjust_sampling_dict(nbrs_dict,
                         seed_cand,
                         time_horizon_factor=8,
                         realizations=1000,
                         *args,
                         **kwargs):
    """Adjust the probability (count) of seeds to be selected in future given
      the current seed_cand is reached
    """
    cand_reach_dict = get_seed_nbrs(seed_cand,
                                    repeated_propagate,
                                    time_horizon_scale=time_horizon_factor,
                                    no_target=True,
                                    prop_routine=propagate,
                                    realizations=realizations,
                                    *args,
                                    **kwargs)

    adjusted_nbrs_dict = {
        key: (np.clip(val - cand_reach_dict[key], a_min=0, a_max=None)
              if key in cand_reach_dict.keys() else val)
        for key, val in nbrs_dict.items()
    }

    return adjusted_nbrs_dict


# metropolis in seedset (based on propagation.propagate) #####################


def optimize_target_seedset(target_seedset, *args, **kwargs):
    """single optimization step of S3D, moving from current target_seedset to
      new_target_seedset. Optimizes on beta-fairness configured in,
      kwargs["ot_refs"]. See utils/ot_utils.prepare_ot_references for details

    Returns:
        tuple: both the old and the new seedset selection with their OT
          scores in ordered pair
    """
    if "realizations" not in kwargs.keys():
        kwargs["realizations"] = 1000  # default values
    if "bin_size" not in kwargs.keys():
        kwargs["bin_size"] = 100  # default values

    kwargs_for_seedset_nbrs = {
        k: v
        for k, v in kwargs.items() if k not in ["ot_refs", "bin_size"]
    }

    if len(target_seedset) != len(set(target_seedset)):
        assert "G" in kwargs.keys()
        global_logger.warn(
            "Some seedsets repeated! Replacing duplicates with random nodes.")
        set_target_seedset = set(target_seedset)
        total_replacements = len(target_seedset) - len(set_target_seedset)
        underlying_G = kwargs["G"]
        uniq_nodes_set = set(underlying_G.nodes()) - set_target_seedset

        assert len(uniq_nodes_set) >= total_replacements

        target_seedset = list(set_target_seedset) + random.sample(
            list(uniq_nodes_set), total_replacements)

    num_seeds = len(target_seedset)
    target_seedset_nbrs = get_seedset_nbrs(target_seedset,
                                           repeated_propagate,
                                           time_horizon_scale=1,
                                           no_target=True,
                                           prop_routine=propagate,
                                           *args,
                                           **kwargs_for_seedset_nbrs)

    global_logger.info("Seedset neighbours calculated!")

    curr_nbrs_counter_dict = target_seedset_nbrs

    for target_seed in target_seedset:
        assert curr_nbrs_counter_dict[target_seed] >= 0

    new_cand_target_seedset = [
        random.choices(list(curr_nbrs_counter_dict.keys()),
                       weights=list(curr_nbrs_counter_dict.values()))[0]
    ]

    for _ in range(num_seeds - 1):
        curr_nbrs_counter_dict = adjust_sampling_dict(
            curr_nbrs_counter_dict,
            new_cand_target_seedset[-1],
            time_horizon_factor=SHALLOW_LOOKAHEAD_HORIZON_FACTOR,
            *args,
            **kwargs_for_seedset_nbrs)
        new_cand_target_seedset += [
            random.choices(list(curr_nbrs_counter_dict.keys()),
                           weights=list(curr_nbrs_counter_dict.values()))[0]
        ]

    assert len(new_cand_target_seedset) == num_seeds

    target_seedset_ot_score = seedset_reach_transport_score(
        target_seedset, time_horizon_scale=1, *args, **kwargs)

    cand_seedset_ot_score = seedset_reach_transport_score(
        new_cand_target_seedset, time_horizon_scale=1, *args, **kwargs)

    global_logger.info("Old/New OT scores calculated!")

    acceptance_prob = np.clip(
        np.exp(EXPLOIT_TO_EXPLORE_RATIO *
               (target_seedset_ot_score - cand_seedset_ot_score)),
        a_min=0,
        a_max=1)

    new_target_seedset = None
    new_target_seedset_ot_score = None

    if random.random() < acceptance_prob:
        new_target_seedset = deepcopy(new_cand_target_seedset)
        new_target_seedset_ot_score = cand_seedset_ot_score
    else:
        if random.random() < NON_ACCEPTANCE_RETENTION_PROB:
            new_target_seedset = deepcopy(target_seedset)
            new_target_seedset_ot_score = target_seedset_ot_score
        else:
            underlying_G = kwargs["G"]
            new_target_seedset = random.sample(list(underlying_G.nodes()),
                                               num_seeds)
            new_target_seedset_ot_score = seedset_reach_transport_score(
                new_target_seedset, time_horizon_scale=1, *args, **kwargs)

    return (new_target_seedset,
            new_target_seedset_ot_score), (target_seedset,
                                           target_seedset_ot_score)
