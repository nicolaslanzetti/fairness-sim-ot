import logging
import os
import random
import string

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
LOG_LEVEL = os.environ.get("LOG_LEVEL")
LOG_FILE = os.environ.get("LOG_FILE")

assert PROJECT_ROOT is not None
assert LOG_LEVEL is not None
assert LOG_FILE is not None

REQD_LOG_LEVEL = logging.CRITICAL

if LOG_LEVEL == "DEBUG":
    REQD_LOG_LEVEL = logging.DEBUG
elif LOG_LEVEL == "INFO":
    REQD_LOG_LEVEL = logging.INFO

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
GLOBAL_LOGGER_ID = "global_logger"

logging.basicConfig(level=REQD_LOG_LEVEL,
                    filename=LOG_FILE,
                    filemode='a',
                    format=LOG_FORMAT)


def hk_init():
    global_logger = logging.getLogger(GLOBAL_LOGGER_ID)
    return PROJECT_ROOT, global_logger


def get_local_logger(name, log_file):
    # replace with a new handler for the same logger
    logger = logging.getLogger(name)

    num_curr_handlers = len(logger.handlers)
    assert num_curr_handlers <= 1

    if num_curr_handlers == 1:
        curr_handler = logger.handlers[0]
        logger.removeHandler(curr_handler)
        del curr_handler

    assert len(logger.handlers) == 0

    new_handler = logging.FileHandler(log_file, mode='w')
    # individual experiments should always get an info log level
    new_handler.setLevel(logging.INFO)
    logger.addHandler(new_handler)

    return logger


def concat_dicts(aggr_dict, elem_dict):
    assert aggr_dict.keys() == elem_dict.keys()

    for key in elem_dict.keys():
        aggr_dict[key].append(elem_dict[key])

    return aggr_dict


def generate_random_alphanumeric_string(length):
    """Generates a random alphanumeric string of the given length."""
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))


def is_float(elem):
    try:
        float(elem)
        return True
    except ValueError:
        return False
