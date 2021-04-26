"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import logging

import torch


logger = logging.getLogger()


def train_output(opt, n_epochs, trainer, loader, val_loader, criterion, part_id, device):
    logger.info(f'Starting training part {part_id}...')

    total_epoch = trainer.start_epoch + n_epochs

    for epoch in range(trainer.start_epoch, total_epoch):
        for input, target in loader:
            # train step
            input, target = \
                input.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output, loss = trainer.step(input, target, criterion)

            # get some batch statistics
            _, predicted = torch.max(output, 1)
            acc = 100 * (predicted == target).sum().to(torch.float).item() / target.size(0)

        # validate
        if epoch % opt.val_freq == opt.val_freq - 1:
            if val_loader is not None:
                correct, total = 0, 0
                total_batch_val = len(val_loader)
                for i, (input, target) in enumerate(val_loader):
                    input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
                    output = trainer.get_eval_output(input)

                    _, predicted = torch.max(output, 1)
                    batch_correct = (predicted == target).sum().item()
                    correct += batch_correct
                    total += target.size(0)
                    batch_acc = 100.0 * batch_correct / target.size(0)
                    message = f'batch: {i + 1}/{total_batch_val}, batch val acc (%): {batch_acc:.3f}'
                    logger.debug(message)

                acc = 100.0 * correct / total

                message = f'[Output layer {part_id}, epoch: {epoch+1}] val acc (%): {acc:.3f}'
                logger.info(message)
                if opt.schedule_lr:
                    trainer.scheduler_step(acc)

    logger.info(f'Output layer {part_id} training finished!')
