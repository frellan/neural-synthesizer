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
from kernet.layers.alignment_linear import AlignmentLinear


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
            elif isinstance(m, AlignmentLinear):
                init.kaiming_normal_(m.linear.weight, **kwargs)
                m.linear.weight.data *= scale
                if m.linear.bias is not None:
                    m.linear.bias.data.fill_(bias_fill)
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

def attach_head(model, opt):
    """
    Attach a trainable, two-layer MLP projection head to model.
    The size of the head is determined dynamically.

    Args:
      model (a torch.nn.Sequential object): The network to be modified. It is important
      that model is a torch.nn.Sequential object since if otherwise model may not be
      subscriptable.

    Returns a new model with a projection head attached to the last module in
    the model.
    """
    if not getattr(opt, 'use_proj_head', None):
        return model

    from kernet.models import Flatten

    device = next(model.parameters()).device
    dummy_input = torch.randn((1,) + eval(opt.data_shape)).to(device)
    dummy_output = model(dummy_input)
    output_size = len(dummy_output.flatten())
    if output_size == opt.head_size:
        return model

    mid = (output_size + opt.head_size) // 2

    proj_head = torch.nn.Sequential(*[
        Flatten(),
        torch.nn.Linear(output_size, mid),
        torch.nn.ReLU(),
        torch.nn.Linear(mid, opt.head_size)
    ]).to(device)

    # only modify the last module of the model
    logger.debug('Before adding projection head:\n')
    logger.debug(str(model))
    logger.debug('Adding projection head...')
    model[-1] = torch.nn.Sequential(*[
        model[-1],
        proj_head
    ])
    logger.debug('After adding projection head:\n')
    logger.debug(str(model))
    return model
