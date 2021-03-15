"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import logging
import functools

import torch

import kernet.utils as utils
from kernet.models import Flatten
# from kernet.layers.klinear import _kLayer, kLinear
from kernet.layers.alignment_linear import _DynamicLayer, AlignmentLinear
from kernet.models.base_model import BaseModel


logger = logging.getLogger()

# TODO trainable centers?


class Simple(BaseModel):
    def __init__(self, opt, centers=None, *args, **kwargs):
        super(Simple, self).__init__(*args, **kwargs)
        if opt.dataset in ['mnist', 'fashionmnist']:
            self.feat_len = 400
            in_channels = 1
        elif opt.dataset in ['cifar10', 'cifar100', 'svhn']:
            self.feat_len = 576
            in_channels = 3
        else:
            raise NotImplementedError()

        if opt.activation == 'tanh':
            self.act = torch.nn.Tanh
        elif opt.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid
        elif opt.activation == 'relu':
            self.act = torch.nn.ReLU
        else:
            raise NotImplementedError()

        conv1 = torch.nn.Conv2d(in_channels, 6, 5, padding=2)
        act1 = self.act()
        pool1 = torch.nn.MaxPool2d(2)
        self.conv1 = torch.nn.Sequential(*[conv1, act1, pool1])
        conv2 = torch.nn.Conv2d(6, 16, 5)
        act2 = self.act()
        pool2 = torch.nn.MaxPool2d(2)
        flatten = Flatten()
        self.conv2 = torch.nn.Sequential(*[conv2, act2, pool2, flatten])

        self.fc1 = torch.nn.Sequential(*[torch.nn.Linear(self.feat_len, 120), self.act()])
        self.fc2 = torch.nn.Linear(120, 84)

        if centers is not None:
            # centers is a tuple of (input, target)
            centers3 = utils.supervised_sample(
                centers[0],
                centers[1],
                opt.n_centers3).clone().detach()
        else:
            centers3 = None

        # self.fc3 = kLinear(
        #     in_features=84,
        #     out_features=10,
        #     kernel=self.kernel,
        #     evaluation=self.evaluation,
        #     centers=centers3,
        #     sigma=opt.sigma3)
        self.fc3 = AlignmentLinear(in_features=84, out_features=10, activation=opt.activation)

        self.opt = opt

        self.print_network(self)

    def forward(self, input, update_centers=True):
        # if update_centers:
        #     self.update_centers()
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output

    def split(self, n_parts, **kwargs):
        """
        Split the model into n_parts parts for modular training.
        The last part is always the (kernelized) output layer.

        Args:
          n_parts (int): The number of parts to split.

        Returns:
          models (tuple): A tuple of n_parts models.
          trainables (tuple): A tuple of n_parts sets of trainable params.
        """
        if n_parts < 1:
            raise ValueError(
                'n_parts should be at least 1, got {} instead'.format(n_parts))
        if n_parts > 5:
            logger.warning('{} can be split into at most 5 parts. Splitting it into ' +
                           '5 parts instead of the requested {} parts...'.format(self.__class__.__name__, n_parts))
            n_parts = 5

        # in modular training (assumed training mode given split has been called),
        # update_centers do not need to be performed at each forward call. Define
        # new forward to bypass the forward function defined for the entire model
        self.forward = functools.partial(self.forward, update_centers=False)

        output_layer = list(self.children())[-1]
        if n_parts == 1:
            models, trainables = (self,), (self,)
        elif n_parts == 2:
            hidden_layers = utils.attach_head(
                torch.nn.Sequential(*list(self.children())[:-1]), self.opt)
            t1 = hidden_layers
            models, trainables = (hidden_layers, self), (t1, output_layer)
        elif n_parts == 3:
            hidden_layers1 = utils.attach_head(
                torch.nn.Sequential(*list(self.children())[:-2]), self.opt)
            hidden_layers2 = utils.attach_head(
                torch.nn.Sequential(*list(self.children())[:-1]), self.opt)
            t1 = hidden_layers1
            t2 = hidden_layers2[-1]
            models, trainables = (hidden_layers1, hidden_layers2, self), \
                                 (t1, t2, output_layer)
        elif n_parts == 4:
            hidden_layers1 = utils.attach_head(
                torch.nn.Sequential(*list(self.children())[:-3]), self.opt)
            hidden_layers2 = utils.attach_head(
                torch.nn.Sequential(*list(self.children())[:-2]), self.opt)
            hidden_layers3 = utils.attach_head(
                torch.nn.Sequential(*list(self.children())[:-1]), self.opt)
            t1 = hidden_layers1
            t2 = hidden_layers2[-1]
            t3 = hidden_layers3[-1]
            models, trainables = (hidden_layers1, hidden_layers2, hidden_layers3, self), \
                                 (t1, t2, t3, output_layer)
        elif n_parts == 5:
            hidden_layers1 = utils.attach_head(
                torch.nn.Sequential(*list(self.children())[:-4]), self.opt)
            hidden_layers2 = utils.attach_head(
                torch.nn.Sequential(*list(self.children())[:-3]), self.opt)
            hidden_layers3 = utils.attach_head(
                torch.nn.Sequential(*list(self.children())[:-2]), self.opt)
            hidden_layers4 = utils.attach_head(
                torch.nn.Sequential(*list(self.children())[:-1]), self.opt)
            t1 = hidden_layers1
            t2 = hidden_layers2[-1]
            t3 = hidden_layers3[-1]
            t4 = hidden_layers4[-1]
            models, trainables = (hidden_layers1, hidden_layers2, hidden_layers3, hidden_layers4, self), \
                                 (t1, t2, t3, t4, output_layer)
        else:
            raise ValueError('Invalid n_parts: {}'.format(n_parts))

        logger.debug('Splitting {} into:'.format(self.__class__.__name__))
        for i, t in enumerate(trainables):
            logger.debug('part {}:\n'.format(i + 1) + str(t))

        return models, [_.parameters() for _ in trainables]

    @staticmethod
    def modify_commandline_options(parser, **kwargs):
        parser = _DynamicLayer.modify_commandline_options(parser, **kwargs)

        parser.add_argument('--n_centers3', type=int, default=1000,
                            help='The number of centers for the kernelized fc layer 3. Note that kernels evaluated directly do not need centers. For them, this param has no effect.')
        parser.add_argument('--sigma3', type=float, default=9.,
                            help='The optional sigma hyperparameter for layer fc3.')

        return parser