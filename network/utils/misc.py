"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import sys
import time
import logging
import argparse
import functools
from shutil import get_terminal_size

import torch
import torchvision

logger = logging.getLogger()
INF = float('inf')
AVOID_ZERO_DIV = torch.tensor(1e-12)


def mask_loss_fn(mask_val):
    # only computes loss on examples where target == mask_val
    def wrapper(loss_fn):
        @functools.wraps(loss_fn)
        def wrapped(input, target):
            idx = target == mask_val
            return loss_fn(input[idx], target[idx])
        return wrapped
    return wrapper


def one_hot_encode(target, n_classes):
    """
    One-hot encode. Uses the values in target as the positions of
    the 1s in the resulting one-hot vectors.

    Args:
      target (tensor): Categorical labels with values in
        {0, 1, ..., n_classes-1}.
      n_classes (int)

    Shape:
      - Input: (n_examples, 1) or (n_examples,)
      - Output: (n_examples, n_classes)
    """
    if len(target.size()) > 1:
        target.squeeze_()
    target_onehot = target.new_zeros(target.size(0), n_classes)
    target_onehot[range(target.size(0)), target] = 1
    return target_onehot


def to_unit_vector(input):
    """
    Make each vector in input have Euclidean norm 1.

    Shape:
      input: (n_examples, d)
    """
    return input / torch.max(torch.norm(input, dim=1, keepdim=True), AVOID_ZERO_DIV.to(input.device))


def sample(tensor, n):
    """
    Returns a random sample of size n from tensor.

    tensor may have shape (N, ...). And the first dimension is assumed
    to be the batch dimension
    """
    logger.debug('Sampling {} from a tensor of size {}...'.format(
        n, list(tensor.size())))
    perm = torch.randperm(len(tensor))
    idx = perm[:n]
    return tensor[idx]


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('true', 't'):
        return True
    elif s.lower() in ('false', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            'Cannot interpret {} as bool'.format(s))


def make_deterministic(seed):
    import os
    import random
    import numpy as np
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def upper_tri(mtrx):
    """
    Returns elements on the upper triangle minus the main diagonal of mtrx.

    Args:
      mtrx (2D matrix): The matrix to be processed.

    Returns a 1D vector of elements from mtrx.
    """
    upper_tri_indices = torch.triu_indices(mtrx.size(0), mtrx.size(1), offset=1)
    return mtrx[upper_tri_indices[0], upper_tri_indices[1]]
