"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import logging

import torch

import kernet.utils as utils
from kernet.models.klenet5 import kLeNet5
from kernet.layers.klinear import _kLayer, kLinear


logger = logging.getLogger()


class k2LeNet5(kLeNet5):
    def __init__(self, opt, centers=None, *args, **kwargs):
        super(k2LeNet5, self).__init__(opt, *args, **kwargs)
        self.fc1 = torch.nn.Linear(self.feat_len, 120)

        if centers is not None:
            # centers is a tuple of (input, target)
            centers2 = utils.supervised_sample(
                centers[0], centers[1], opt.n_centers2).clone().detach()
            centers3 = utils.supervised_sample(
                centers[0], centers[1], opt.n_centers3).clone().detach()
        else:
            centers2, centers3 = None, None

        self.fc2 = kLinear(in_features=120, out_features=84,
                           kernel=self.kernel, evaluation=self.evaluation, centers=centers2, sigma=opt.sigma2)
        self.fc3 = kLinear(in_features=84, out_features=10,
                           kernel=self.kernel, evaluation=self.evaluation, centers=centers3, sigma=opt.sigma3)

        if opt.memory_efficient:
            self.fc2 = utils.to_committee(self.fc2, opt.expert_size)
            self.fc3 = utils.to_committee(self.fc3, opt.expert_size)
        self.print_network(self)

    def update_centers(self):
        # update centers
        if self.fc2.evaluation == 'indirect':
            # not using actual class name here for potential inheritance
            logger.debug('Updating centers for {} fc2...'.format(
                self.__class__.__name__))

            def update_fn(x): return self.fc1(self.conv2(self.conv1(x)))
            _kLayer.update(self.fc2, update_fn)
        if self.fc3.evaluation == 'indirect':
            logger.debug('Updating centers for {} fc3...'.format(
                self.__class__.__name__))

            def update_fn(x): return self.fc2(
                self.fc1(self.conv2(self.conv1(x))))
            _kLayer.update(self.fc3, update_fn)

    @staticmethod
    def modify_commandline_options(parser, **kwargs):
        parser = kLeNet5.modify_commandline_options(parser, **kwargs)

        parser.add_argument('--n_centers2', type=int, default=1000,
                            help='The number of centers for the kernelized fc layer 2. Note that kernels evaluated directly do not need centers. For them, this param has no effect.')
        parser.add_argument('--n_centers3', type=int, default=1000,
                            help='The number of centers for the kernelized fc layer 3. Note that kernels evaluated directly do not need centers. For them, this param has no effect.')
        parser.add_argument('--sigma2', type=float, default=5.,
                            help='The optional sigma hyperparameter for layer fc2.')
        parser.add_argument('--sigma3', type=float, default=9.,
                            help='The optional sigma hyperparameter for layer fc3.')
        return parser
