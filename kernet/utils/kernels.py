"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn.functional as F

from .misc import *


def poly2_phi_fn_dir(input, **kwargs):
    """
    Polynomial kernel of order 2 (direct)
    """
    # TODO is unbounded so no k_min or k_max. Implement?
    raise NotImplementedError()


def nn_tanh_phi_fn_dir(input, **kwargs):
    """
    Shape:
      input: (n_examples, d)
    """
    output = torch.tanh(input)
    # make sure any nonzero \phi(x) is of unit norm
    return to_unit_vector(output)


def nn_sigmoid_phi_fn_dir(input, **kwargs):
    """
    Shape:
      input: (n_examples, d)
    """
    output = torch.sigmoid(input)
    # make sure any nonzero \phi(x) is of unit norm
    return to_unit_vector(output)


def nn_relu_phi_fn_dir(input, **kwargs):
    """
    Shape:
      input: (n_examples, d)
    """
    output = F.relu(input)
    # make sure any nonzero \phi(x) is of unit norm
    return to_unit_vector(output)


def nn_reapen_phi_fn_dir(input, **kwargs):
    """
    ReLU + average pool 2D + flatten.

    Shape:
      input: (n_examples, d1, d2, d3)
    """
    output = F.relu(input)
    output = F.avg_pool2d(output, 4)
    output = output.view(output.size(0), -1)
    # make sure any nonzero \phi(x) is of unit norm
    return to_unit_vector(output)
