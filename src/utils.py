# coding=utf-8

"""Utilities for logging and serialization"""

import random
import numpy as np
import torch

def save_rank_0(args, message):
    with open(args.log_file, "a") as f:
        f.write(message + "\n")
        f.flush()


def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
