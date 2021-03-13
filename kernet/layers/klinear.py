"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch

import kernet.utils as utils
from kernet.layers.kcore import Phi


class _kLayer(torch.nn.Module):
    """
    Base class for all kernelized layers.
    """

    def __init__(self, *args, **kwargs):
        super(_kLayer, self).__init__(*args, **kwargs)

    @staticmethod
    def modify_commandline_options(parser, **kwargs):
        parser.add_argument('--expert_size', type=int, default=300,
                            help='The expert_size param for the kLinear committees.')
        return parser

    @staticmethod
    def update(layer, update_fn):
        for i in range(layer.n_experts):
            expert = getattr(layer, 'expert' + str(i))
            expert.centers = update_fn(expert.centers_init)


class kLinear(_kLayer):
    def __init__(
        self,
        out_features,
        in_features=None,
        kernel='gaussian',
        bias=True,
        evaluation='direct',
        centers=None,
        trainable_centers=False,
        sigma=1.,
        *args,
        **kwargs):
        """
        A kernelized linear layer. With input x, this layer computes:
        w.T @ \phi(x) + bias, where 
          1) when evaluation is 'indirect',
          \phi(x).T = (k(c_1, x), ..., k(c_m, x)), k is the kernel function, 
          and the c_i's are the centers;

          2) when evaluation is 'direct',
          \phi(x).T is the actual image of x under the corresponding feature map.
          In this case, there is no need to specify the centers.

        Direct evaluation only works on kernels whose feature maps \phi's are 
        implementable. 

        Args:
          out_features (int): The output dimension. 
          in_features (int): The input dimension. Not needed for certain
          kernels. Default: None. 
          kernel (str): Name of the kernel. Default: 'gaussian'.
          bias (bool): If True, add a bias term to the output. Default: True. 
          evaluation (str): Whether to evaluate the kernel machines directly 
          as the inner products between weight vectors and \phi(input), which
          requires explicitly writing out \phi, or indirectly 
          via an approximation using the reproducing property, which works 
          for all kernels but can be less accurate and less efficient. 
          Default: 'direct'. Choices: 'direct' | 'indirect'.
          centers (tensor): A set of torch tensors. Needed only when evaluation
          is set to 'indirect'. Default: None.
          trainable_centers (bool): Whether to treat the centers as trainable
            parameters. Default: False.
          sigma (float): The sigma hyperparameter for the kernel. See the
            kernel definitions for details. Default: 1..
        """
        super(kLinear, self).__init__(*args, **kwargs)
        self.evaluation = evaluation.lower()
        if self.evaluation not in ['direct', 'indirect']:
            raise ValueError('Evaluation method can be "direct" or "indirect", got {}'.format(evaluation))

        # prepare the centers for indirect evaluation
        if self.evaluation == 'indirect':
            if trainable_centers:
                self.trainable_centers = True
                self.centers = torch.nn.Parameter(
                    centers.clone().detach().requires_grad_(True))
            else:
                self.centers = centers
                # centers_init will be saved together w/ the model but centers won't
                # see https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/11
                self.register_buffer('centers_init', centers.clone().detach())
        else:
            del centers

        self.kernel = kernel
        self.phi = Phi(
            kernel=kernel,
            in_features=in_features,
            evaluation=self.evaluation,
            sigma=sigma)
        self.out_features = out_features
        self.in_features = in_features
        if self.evaluation == 'indirect':
            self.linear = torch.nn.Linear(
                len(centers),
                out_features,
                bias=bias)
        else:
            self.linear = torch.nn.Linear(
                self.phi.out_features,
                out_features,
                bias=bias)

    def forward(self, input):
        if self.evaluation == 'indirect':
            return self.linear(self.phi(input, centers=self.centers))
        else:
            return self.linear(self.phi(input))
