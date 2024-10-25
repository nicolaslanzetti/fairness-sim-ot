#!/usr/bin/env python

# run it from the PROJECT_ROOT while keeping the script in the PROJECT_ROOT too
# the script converts datapoints from csv format to tikz compatible ones
import csv
import os
import sys

import numpy as np

from utils.data_utils import load_vectors_from_csv
from utils.housekeeping_utils import hk_init

RUN_ID = 1
# global config options
LOG_LEVEL = "INFO"  # only DEBUG and INFO supported, anything else leads ...
# ... to CRITICAL
LOG_FILE = "./run_" + str(
    RUN_ID) + ".log"  # path relative to current working directory

PROJECT_ROOT = os.getcwd()
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
print("Project Root:", PROJECT_ROOT)

os.environ["LOG_LEVEL"] = LOG_LEVEL
print("Log level for libraries:", LOG_LEVEL)

os.environ["LOG_FILE"] = PROJECT_ROOT + "/" + LOG_FILE
print("Library logs are present at", os.environ["LOG_FILE"])

_, global_logger = hk_init()

ROOT_DIR = sys.argv[1]  # "./experiments/paper_exps/"


def convert_point_space_to_coordinated_form(vecs, bin_size=100):
    assert len(vecs) == 3
    x_coords = []
    y_coords = []
    vals = [0.0] * (bin_size**2)

    for i in range(bin_size):
        for j in range(bin_size):
            x_coords.append(i * 1.0)
            y_coords.append(j * 1.0)

    max_len = len(vecs[0])
    assert max_len == len(vecs[1]) and max_len == len(vecs[2])

    for i in range(max_len):
        x_coord = int(vecs[0][i] *
                      bin_size) if vecs[0][i] < 1 else bin_size - 1
        y_coord = int(vecs[1][i] *
                      bin_size) if vecs[1][i] < 1 else bin_size - 1

        vals[x_coord * bin_size + y_coord] += vecs[2][i]

    return (x_coords, y_coords, vals)


def dump_space_points_to_segmented_csv_format(filename, is_coordianted=True):
    vecs = load_vectors_from_csv(filename=filename)
    if not is_coordianted:
        vecs = convert_point_space_to_coordinated_form(vecs)

    vecs_lens = [len(vec) for vec in vecs]
    max_len = max(vecs_lens)
    coord_norm_factor = round(np.sqrt(max_len))
    num_vecs = len(vecs_lens)

    assert (num_vecs == 3)

    new_filename = filename[:-4] + "_seg.csv"

    with open(new_filename, 'w') as f:
        write = csv.writer(f)
        write_ctr = 0
        skip = False

        while write_ctr < max_len:
            row = []
            if (write_ctr != 0 and write_ctr % coord_norm_factor == 0
                    and not skip):
                skip = True
            else:
                skip = False
                for j in range(num_vecs):
                    if write_ctr < vecs_lens[j]:
                        val = vecs[j][write_ctr]
                        if j in [0, 1]:
                            val = 1.0 * val / coord_norm_factor
                        row.append(val)
                    else:
                        row.append("")
                write_ctr += 1
            write.writerow(row)


for root, subdirs, files in os.walk(ROOT_DIR):
    for file in files:
        if (file.find("outreach") >= 0 and file.endswith("_compat.csv")):
            file_path = os.path.join(root, file)
            print(file_path)

            dump_space_points_to_segmented_csv_format(file_path,
                                                      is_coordianted=True)

            global_logger.info("Processed %s", file_path)
        elif file.endswith("markers.csv"):
            file_path = os.path.join(root, file)
            print(file_path)

            dump_space_points_to_segmented_csv_format(file_path,
                                                      is_coordianted=False)

            global_logger.info("Processed %s", file_path)
