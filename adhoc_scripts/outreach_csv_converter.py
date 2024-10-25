#!/usr/bin/env python

# run it from the PROJECT_ROOT while keeping the script in the PROJECT_ROOT too
# the script converts datapoints from csv format to tikz compatible ones

import ast
import os
import sys

from utils.data_utils import dump_vectors_to_csv, load_vectors_from_csv
from utils.housekeeping_utils import hk_init

RUN_ID = 1
# global config options
LOG_LEVEL = "INFO"  # only DEBUG and INFO supported, anything else leads...
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

for root, subdirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.find("outreach") >= 0 and file.find(
                "compat") == -1 and file.endswith(".csv"):
            file_path = os.path.join(root, file)
            print(file_path)
            out = load_vectors_from_csv(file_path)[0]
            vecs = (ast.literal_eval(elem) for elem in out)

            new_file_path = os.path.join(root, file[:-4] + "_compat.csv")

            dump_vectors_to_csv(new_file_path, *vecs)

            global_logger.info("%s --> %s", file_path, new_file_path)
