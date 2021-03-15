"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch

import kernet.utils as utils


class _DynamicLayer(torch.nn.Module):
    """
    Base class for all dynamic layers.
    """

    def __init__(self, *args, **kwargs):
        super(_DynamicLayer, self).__init__(*args, **kwargs)

    @staticmethod
    def modify_commandline_options(parser, **kwargs):
        return parser

class AlignmentLinear(_DynamicLayer):
    def __init__(
        self,
        out_features,
        in_features=None,
        activation='relu',
        bias=True,
        *args,
        **kwargs):
        super(AlignmentLinear, self).__init__(*args, **kwargs)

        self.out_features = out_features
        self.in_features = in_features
        self.phi = SimplePhi(activation)
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        return self.linear(self.phi(input))


class SimplePhi(torch.nn.Module):
    def __init__(self, activation, *args, **kwargs):
        super(SimplePhi, self).__init__(*args, **kwargs)
        
        if activation == 'tanh':
            self.k_min, self.k_max = -1., 1.
            self.phi_fn = utils.nn_tanh_phi_fn_dir
        elif activation == 'sigmoid':
            self.k_min, self.k_max = 0., 1.
            self.phi_fn = utils.nn_sigmoid_phi_fn_dir
        elif activation == 'relu':
            self.k_min, self.k_max = 0., 1.
            self.phi_fn = utils.nn_relu_phi_fn_dir
        elif activation == 'reapen':
            self.k_min, self.k_max = 0., 1.
            self.phi_fn = utils.nn_reapen_phi_fn_dir

    def forward(self, input):
        return self.phi_fn(input)

    def get_k_mtrx(self, input1, input2):
        return self(input1).mm(self(input2).t())

    def get_ideal_k_mtrx(self, target1, target2, n_classes):
        """
        Returns the "ideal" kernel matrix G* defined as
          (G*)_{ij} = k_min if y_i == y_j;
          (G*)_{ij} = k_max if y_i != y_j.

        Args:
          target1 (tensor): Categorical labels with values in 
            {0, 1, ..., n_classes-1}.
          target2 (tensor): Categorical labels with values in 
            {0, 1, ..., n_classes-1}.
          n_classes (int)

        Shape:
          - Input:
            target1: (n_examples1, 1) or (1,) (singleton set)
            target2: (n_examples2, 1) or (1,) (singleton set)
          - Output: (n_examples1, n_examples2)
        """
        if n_classes < 2:
            raise ValueError('You need at least 2 classes')

        if len(target1.size()) == 1:
            target1.unsqueeze_(1)
        elif len(target1.size()) > 2:
            raise ValueError('target1 has too many dimensions')
        if len(target2.size()) == 1:
            target2.unsqueeze_(1)
        elif len(target2.size()) > 2:
            raise ValueError('target2 has too many dimensions')

        if torch.max(target1) + 1 > n_classes:
            raise ValueError('target1 has at least one invalid entry')
        if torch.max(target2) + 1 > n_classes:
            raise ValueError('target2 has at least one invalid entry')

        target1_onehot, target2_onehot = \
            utils.one_hot_encode(target1, n_classes).to(torch.float), \
            utils.one_hot_encode(target2, n_classes).to(torch.float)

        ideal = target1_onehot.mm(target2_onehot.t())

        if self.k_min != 0:
            min_mask = torch.full_like(ideal, self.k_min)
            ideal = torch.where(ideal == 0, min_mask, ideal)
        if self.k_max != 1:
            max_mask = torch.full_like(ideal, self.k_max)
            ideal = torch.where(ideal == 1, max_mask, ideal)

        return ideal
