import numpy as np

from utils import housekeeping_utils

PROJECT_ROOT, global_logger = housekeeping_utils.hk_init()


def rotate_hist_binnings_anticlockwise(hist_mat):
    return np.rot90(hist_mat, k=1, axes=(0, 1))
