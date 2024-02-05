import os
import sys
import time
import argparse
import logging
import logging.config

import munch
import yaml

def merge_nested_dict(d, other):
    new = dict(d)
    for k, v in other.items():
        if d.get(k, None) is not None and type(v) is dict:
            new[k] = merge_nested_dict(d[k], v)
        else:
            new[k] = v
    return new


def get_config(default_file):
    p = argparse.ArgumentParser(description='Learned Step Size Quantization')
    p.add_argument('config_file', metavar='PATH', nargs='+',
                   help='path to a configuration file')
    arg = p.parse_args()

    with open(default_file) as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    for f in arg.config_file:
        if not os.path.isfile(f):
            raise FileNotFoundError('Cannot find a configuration file at', f)
        with open(f) as yaml_file:
            c = yaml.safe_load(yaml_file)
            cfg = merge_nested_dict(cfg, c)

    return munch.munchify(cfg)


def init_logger(experiment_name, experiment_dir, output_dir, cfg_file=None):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    exp_full_name = time_str if experiment_name is None \
                             else experiment_name + '_' + time_str
    log_dir = output_dir / experiment_dir / exp_full_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / (exp_full_name + '.log')
    logging.config.fileConfig(cfg_file, defaults={'logfilename': log_file})
    logger = logging.getLogger()
    logger.info('Log file for this run: ' + str(log_file))

    return log_dir


def init_directory(experiment_name, experiment_dir, output_dir):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    exp_full_name = time_str if experiment_name is None \
                             else experiment_name + '_' + time_str
    log_dir = output_dir / experiment_dir / exp_full_name
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir


def setup_logger(save_dir, rank, filename='log.txt'):
    logger = logging.getLogger()
    # don't log results for the non-master process
    if rank > 0:
        logger.setLevel(logging.ERROR)
        return logger
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

