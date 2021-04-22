"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import logging

import torch

logger = logging.getLogger()
best_path = "./checkpoint/best.pth"

def train_hidden(opt, n_epochs, trainer, loader, val_loader, criterion, part_id, device, save_model=False):
    logger.info(f'Starting training layer {part_id}...')

    best_val_acc = 0
    unimproving_epochs = 0

    total_epoch = trainer.start_epoch + n_epochs

    for epoch in range(trainer.start_epoch, total_epoch):
        for input, target in loader:
            # train step
            input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output, loss = trainer.step(input, target, criterion, minimize=False)

        # validate
        if epoch % opt.val_freq == opt.val_freq - 1:
            if val_loader is not None:
                hidden_obj, total = 0, 0
                for i, (input, target) in enumerate(val_loader):
                    input, target = \
                        input.to(device, non_blocking=True), target.to(
                            device, non_blocking=True)
                    output = trainer.get_eval_output(input)
                    batch_obj = criterion(output, target).item()
                    hidden_obj += batch_obj
                    total += 1

                hidden_obj /= total
                message = f'[layer {part_id}, epoch: {epoch+1}] val {opt.hidden_objective}: {hidden_obj:.3f}'
                logger.info(message)
                if opt.schedule_lr:
                    trainer.scheduler_step(hidden_obj)

        # early stopping
        if hidden_obj > best_val_acc:
            best_val_acc = hidden_obj
            unimproving_epochs = 0
            if save_model:
                torch.save(trainer.model, best_path)
                print(f'New best val acc {best_val_acc}, saving model')
        else:
            unimproving_epochs += 1
            if unimproving_epochs > 1:
                print(f'                   {unimproving_epochs} epochs')
            else:
                print(f'No improvement for {unimproving_epochs} epochs')

        if unimproving_epochs >= 20:
            print('No improvement for 20 epochs, STOPPING EARLY')
            break

    logger.info(f'Layer {part_id} training finished!')
    return best_val_acc