"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Modified from: 
  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/__init__.py
"""
import logging
import importlib

import torch

import network.utils as utils


logger = logging.getLogger()
AVOID_ZERO_DIV = torch.tensor(1e-12)


class Normalize(torch.nn.Module):
    """
    Make each vector in input have Euclidean norm 1.

    This class is a wrap on utils.to_unit_vector so that it
    can be registered as a module when used in a model.
    """

    def forward(self, input):
        """
        Shape:
          input: (n_examples, d)
        """
        return utils.to_unit_vector(input)


class Flatten(torch.nn.Module):
    """
    A convenient wrap on torch.flatten.
    """

    def forward(self, input):
        """
        Shape:
          input: (n_examples, d1, ...)

        Returns a tensor with shape (n_examples, d1 * ...).
        """
        return torch.flatten(input, start_dim=1)

class UnitVectorize(torch.nn.Module):
    """
    Make each vector in input have Euclidean norm 1.

    Shape:
      input: (n_examples, d)
    """

    def forward(self, input):
        return input / torch.max(
            torch.norm(input, dim=1, keepdim=True),
            AVOID_ZERO_DIV.to(input.device)
        )