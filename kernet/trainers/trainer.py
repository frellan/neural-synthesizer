"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import logging
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger()

class Trainer(torch.nn.Module):
    def __init__(
        self,
        opt,
        model=None,
        set_eval=None,
        optimizer=None,
        val_metric_name='acc',
        val_metric_obj='max'):
        super(Trainer, self).__init__()
        if val_metric_obj not in ['min', 'max']:
            raise ValueError()
        self.opt = opt
        self.steps_taken = 0  # the total number of train steps taken
        self.start_epoch = 0
        self.val_metric_name = val_metric_name
        self.val_metric_obj = val_metric_obj
        self.best_val_metric = float('inf') if val_metric_obj == 'min' else -float('inf')
        self.model = model
        self.set_eval = set_eval

        if opt.is_train:
            self.optimizer = optimizer
            if opt.schedule_lr:
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    val_metric_obj,
                    factor=opt.lr_schedule_factor,
                    patience=opt.lr_schedule_patience,
                    verbose=True)

    def step(self, input, target, criterion, minimize=True):
        self.model.train()
        if self.set_eval:
            self.set_eval.eval()

        self.optimizer.zero_grad()
        output = self.model(input)

        cri_val = criterion(output, target)
        if minimize:
            loss = cri_val
        else:
            loss = -cri_val

        loss.backward()
        self.optimizer.step()
        self.steps_taken += 1

        if minimize:
            return output.detach(), loss.item()
        else:
            return output.detach(), -loss.item()

    def get_eval_output(self, input):
        with torch.no_grad():
            self.model.eval()
            output = self.model(input)
            return output
    
    def scheduler_step(self, val_loss_value):
        self.scheduler.step(val_loss_value)

    def log_loss_values(self, loss_dict):
        for k, v in loss_dict.items():
            logger.add_scalar(k, v, self.steps_taken)
