"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy
import logging

import torch
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from .misc import INF
from kernet import datasets


logger = logging.getLogger()


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.
    Reference: https://github.com/xinntao/BasicSR

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, torch.nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def exclude_during_backward(model):
    logger.debug('Exclude {} during backward...'.format(
        model.__class__.__name__))
    for p in model.parameters():
        p.requires_grad_(False)


def include_during_backward(model):
    logger.debug('Include {} during backward...'.format(
        model.__class__.__name__))
    for p in model.parameters():
        p.requires_grad_()
