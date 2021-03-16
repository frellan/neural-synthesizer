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
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        
        if activation == 'tanh':
            self.act = utils.nn_tanh_phi_fn_dir
        elif activation == 'sigmoid':
            self.act = utils.nn_sigmoid_phi_fn_dir
        elif activation == 'relu':
            self.act = utils.nn_relu_phi_fn_dir
        else:
            raise NotImplementedError()

    def forward(self, input):
        output = self.act(input)
        output = self.linear(output)
        return output
