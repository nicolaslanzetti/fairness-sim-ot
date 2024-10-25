import math

import numpy as np
import ot

from utils import housekeeping_utils
from utils.plot_utils import rotate_hist_binnings_anticlockwise

PROJECT_ROOT, global_logger = housekeeping_utils.hk_init()


def create_ideal_fair_and_efficient_2d_target_dist(n_bins):
    # similar i, j, x, y convention as above
    distr = np.zeros((n_bins, n_bins))
    coords = np.zeros((n_bins, n_bins, 4))
    unrolled_distr = np.zeros((n_bins**2, ))

    for i in range(n_bins):
        for j in range(n_bins):
            x = j
            y = n_bins - 1 - i
            prob_measure = 1 if x + y == 2 * n_bins - 2 else 0
            distr[i][j] = prob_measure
            unrolled_distr[i * n_bins + j] = prob_measure
            coords[i][j] = (x, y, i, j)

    return distr, coords, unrolled_distr


def create_most_unfair_2d_target_dist(n_bins):
    # similar i, j, x, y convention as above
    distr = np.zeros((n_bins, n_bins))
    coords = np.zeros((n_bins, n_bins, 4))
    unrolled_distr = np.zeros((n_bins**2, ))

    for i in range(n_bins):
        for j in range(n_bins):
            x = j
            y = n_bins - 1 - i
            prob_measure = 1 if (x == n_bins - 1 and y == 0) else 0
            distr[i][j] = prob_measure
            unrolled_distr[i * n_bins + j] = prob_measure
            coords[i][j] = (x, y, i, j)

    return distr, coords, unrolled_distr


def create_fair_and_efficient_2d_dist(n_bins,
                                      is_fair=True,
                                      scale_ratio_fairness=1.0,
                                      is_efficient=True,
                                      scale_ratio_efficiency=1.0):
    # similar i, j, x, y convention as above
    distr = np.zeros((n_bins, n_bins))
    coords = np.zeros((n_bins, n_bins, 4))
    unrolled_distr = np.zeros((n_bins**2, ))

    if is_fair:
        for i in range(n_bins):
            for j in range(n_bins):
                x = j
                y = n_bins - 1 - i
                prob_measure = scale_ratio_fairness * abs(n_bins - 1 -
                                                          abs(y - x))
                distr[i][j] += prob_measure
                unrolled_distr[i * n_bins + j] += prob_measure
                coords[i][j] = (x, y, i, j)

    if is_efficient:
        for i in range(n_bins):
            for j in range(n_bins):
                x = j
                y = n_bins - 1 - i
                prob_measure = scale_ratio_efficiency * (
                    math.sqrt(2) *
                    (n_bins - 1) - math.sqrt((x - n_bins - 1)**2 +
                                             (y - n_bins - 1)**2))
                distr[i][j] += prob_measure
                unrolled_distr[i * n_bins + j] += prob_measure
                coords[i][j] = (x, y, i, j)

    return distr, coords, unrolled_distr


def create_fair_and_efficient_2d_cost_matrix(n_bins, *args, **kwargs):
    ref_distr, ref_coords, _ = create_fair_and_efficient_2d_dist(
        n_bins, *args, **kwargs)

    cost_mat = np.zeros((n_bins**2, n_bins**2))
    scale_cost = 1

    for i in range(n_bins):
        for j in range(n_bins):
            # src_x, src_y = ref_coords[i][j][0:2].astype(int)

            for k in range(n_bins):
                for el in range(n_bins):
                    # dest_x, dest_y = ref_coords[k][l][0:2].astype(int)
                    cost_mat[i * n_bins + j, k * n_bins +
                             el] = (ref_distr[k][el] - ref_distr[i][j])

    return cost_mat, scale_cost


def calculate_emd_matrix(source_mat, dest_mat, cost_mat, scale_cost=1):
    # all matrices are row-major indexed and are conveniently
    # mapped for (x, y) cartesian coordinates
    score, res_dict = ot.emd2(source_mat / source_mat.sum(),
                              dest_mat / dest_mat.sum(),
                              (scale_cost * cost_mat),
                              return_matrix=True)

    return score, res_dict


def unroll_dist_from_hist_bins(plot_hist_bins):
    dist = rotate_hist_binnings_anticlockwise(plot_hist_bins)
    unrolled_dist = np.reshape(dist, -1)

    return unrolled_dist


def prepare_ot_references(bin_size, rel_fairness=1, rel_efficiency=1):
    """beta-fairness is encoded here as,
      beta = rel_fairness/(rel_fairness + rel_efficiency)
      This param is then used to calculate beta-fairness
    """
    target_dist = create_ideal_fair_and_efficient_2d_target_dist(bin_size)
    M, scale_k = create_fair_and_efficient_2d_cost_matrix(
        bin_size, True, rel_fairness, True, rel_efficiency
    )  # 1.0, 1.0 are relative weights on fairness and efficiency

    return {"target_dist": target_dist, "cost_mat": M, "scale": scale_k}
