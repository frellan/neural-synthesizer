"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import logging

import torch.nn as nn
import torch.nn.functional as F

import kernet.utils as utils
from kernet.layers.klinear import _kLayer, kLinear
from network.models.resnet import BasicBlock, Bottleneck, ResNet


logger = logging.getLogger()


class BasicBlockNoOutputReLU(BasicBlock):
    """
    The BasicBlock with the output ReLU nonlinearity stripped off.
    """

    def __init__(self, *args, **kwargs):
        super(BasicBlockNoOutputReLU, self).__init__(*args, **kwargs)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out


class BottleneckNoOutputReLU(Bottleneck):
    """
    The Bottleneck with the output ReLU nonlinearity stripped off.
    """

    def __init__(self, *args, **kwargs):
        super(BottleneckNoOutputReLU, self).__init__(*args, **kwargs)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return out


class kResNet(ResNet):
    def __init__(self, opt, centers, block, num_blocks, num_classes=10, in_channels=3):
        super(kResNet, self).__init__(block, num_blocks,
                                      num_classes=num_classes, skip_layer=['layer5', 'fc'], in_channels=in_channels)
        self.opt = opt
        if opt.activation == 'tanh':
            self.kernel = 'nn_tanh'
            self.evaluation = 'direct'
        elif opt.activation == 'sigmoid':
            self.kernel = 'nn_sigmoid'
            self.evaluation = 'direct'
        elif opt.activation == 'relu':
            self.kernel = 'nn_relu'
            self.evaluation = 'direct'
        elif opt.activation == 'reapen':
            self.kernel = 'nn_reapen'
            self.evaluation = 'direct'
        elif opt.activation == 'gaussian':
            self.kernel = 'gaussian'
            self.evaluation = 'indirect'
        else:
            raise NotImplementedError()

        self.layer5 = self._make_layer_no_output_relu(
            block, 512, num_blocks[3], stride=2)

        if centers is not None:
            # centers is a tuple of (input, target)
            centers = utils.supervised_sample(
                centers[0], centers[1], opt.n_centers).clone().detach()
        else:
            centers = None
        fc = kLinear(in_features=512*block.expansion, out_features=num_classes,
                     kernel=self.kernel, evaluation=self.evaluation, centers=centers, sigma=opt.sigma)
        if opt.memory_efficient:
            fc = utils.to_committee(fc, opt.expert_size)
        self.fc = fc
        self.print_network(self)

    def _make_layer_no_output_relu(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides[:-1]:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        # last block does not have the output relu
        stride = strides[-1]
        last_block = BasicBlockNoOutputReLU if block is BasicBlock else BottleneckNoOutputReLU
        layers.append(last_block(self.in_planes, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, input):
        return super().forward(input)
