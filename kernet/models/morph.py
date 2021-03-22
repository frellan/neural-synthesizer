"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import logging
import functools

import torch

import kernet.utils as utils
from kernet.models import Flatten
from kernet.models.base_model import BaseModel


logger = logging.getLogger()


class Morph(torch.nn.Module):
    def __init__(self, opt, device, *args, **kwargs):
        super(Morph, self).__init__(*args, **kwargs)

        self.opt = opt
        self.device = device
        self.n_modules = 0
        self.permanent_components = []
        self.pending_components = []

    def forward(self, input):
        permanent = torch.nn.Sequential(*self.permanent_components).to(self.device)
        pending = torch.nn.Sequential(*self.pending_components).to(self.device)
        output = permanent(input)
        output = pending(output)
        return output

    def add_to_pending_module(self, *components):
        if (len(components) > 1):
            self.pending_components.append(torch.nn.Sequential(*components))
        else:
            self.pending_components.append(*components)

    def clear_pending_module(self):
        self.pending_components = []

    def solidify_pending_module(self):
        self.permanent_components.extend(self.pending_components)
        self.clear_pending_module()
        self.n_modules += 1

    def get_trainable_params(self):
        return torch.nn.Sequential(*self.pending_components).to(self.device).parameters()

    @staticmethod
    def modify_commandline_options(parser, **kwargs):
        return parser
