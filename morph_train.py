"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Modular training example.
"""

import torch
from ax.service.managed_loop import optimize

import network.utils as utils
import network.models.srs_loss as losses
import network.datasets as datasets
from network.models import Flatten
from network.models.resnet import ResnetBlock, ResnetOutput
from network.models.morph import Block, Cell, Morph
from network.parsers.argument_parser import ArgumentParser
from network.trainers import train_hidden, train_output, Trainer


parameters = {
    'lr': .15,
    'weight_decay': .0000625,
    'momentum': .9
}
checkpoint_path = "./checkpoint/bayesian_start.pth"

opt = None
model = None
hidden_criterion = None
output_criterion = None
loader = None
val_loader = None
device = None


def modify_commandline_options(parser, **kwargs):
    loss_names = [
        'srs_raw',
        'srs_nmse',
        'srs_alignment',
        'srs_upper_tri_alignment',
        'srs_contrastive',
        'srs_log_contrastive'
    ]
    parser.add_argument('--hidden_objective',
        choices=loss_names + [_ + '_neo' for _ in loss_names],
        default='srs_alignment',
        help='Proxy hidden objective.')
    return parser

def get_hidden_criterion(opt):
    selected_loss_fn = getattr(losses, opt.hidden_objective)
    return selected_loss_fn(opt.activation, opt.n_classes)

def train_evaluate(parameterization):
    global model
    parameters = {
        'lr': parameterization.get("lr", 0.001),
        'weight_decay': parameterization.get("weight_decay", .0000625),
        'momentum': parameterization.get("momentum", .9)
    }
    optimizer = utils.get_optimizer(
        opt,
        params=model.get_trainable_params(),
        lr=parameters['lr'],
        weight_decay=parameters['weight_decay'],
        momentum=parameters['momentum'])
    trainer = Trainer(opt=opt, model=model, optimizer=optimizer)
    for epoch in range(10):
        for input, target in loader:
            input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
            trainer.step(input, target, hidden_criterion, minimize=False)
    
    with torch.no_grad():
        hidden_obj, total = 0, 0
        for input, target in val_loader:
            input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = trainer.get_eval_output(input)
            batch_obj = hidden_criterion(output, target).item()
            hidden_obj += batch_obj
            total += 1

    model = torch.load(checkpoint_path)

    return hidden_obj / total

def train_pending(parameters, epochs):
    optimizer = utils.get_optimizer(
        opt,
        params=model.get_trainable_params(),
        lr=parameters['lr'],
        weight_decay=parameters['weight_decay'],
        momentum=parameters['momentum'])
    trainer = Trainer(
        opt=opt,
        model=model,
        set_eval=None,
        optimizer=optimizer,
        val_metric_name=opt.hidden_objective,
        val_metric_obj='max')
    train_hidden(
        opt,
        n_epochs=epochs,
        trainer=trainer,
        loader=loader,
        val_loader=val_loader,
        criterion=hidden_criterion,
        part_id=model.n_modules + 1,
        device=device)


def main():
    global opt
    global loader
    global val_loader
    global model
    global hidden_criterion
    global output_criterion

    opt = ArgumentParser().parse()
    utils.set_logger(opt=opt, filename='train.log', filemode='w')
    if opt.seed:
        utils.make_deterministic(opt.seed)
    loader, val_loader = datasets.get_dataloaders(opt)

    epochs = 100
    output_epochs = 10
    model = Morph(opt, device).to(device)

    hidden_criterion = get_hidden_criterion(opt)
    output_criterion = torch.nn.CrossEntropyLoss() if opt.loss == 'xe' else torch.nn.MultiMarginLoss()

    # Add inital modules
    model.add_pending(torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False).to(device))
    model.add_pending(torch.nn.BatchNorm2d(16).to(device))

    for i in range(3):
        # Add components
        cell = Cell(in_channels=16, use_residual=i != 2, blocks=[
            Block(16, 16, 3, device),
            Block(16, 16, 3, device),
            Block(16, 16, 3, device),
            Block(16, 16, 3, device),
            Block(16, 16, 3, device),
            Block(16, 16, 3, device),
        ], device=device)
        model.add_pending(cell)
        model.add_pending(cell)

        # Save model for resetting on bayesian trials and after
        torch.save(model, checkpoint_path)

        # Search for parameters
        best_parameters, values, experiment, model = optimize(
            parameters=[
                {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
                {"name": "weight_decay", "type": "range", "bounds": [1e-5, 1e-3]},
                {"name": "momentum", "type": "range", "bounds": [0.5, 1.0]},     
            ],
            evaluation_function=train_evaluate,
            objective_name=opt.hidden_objective,
        )

        # Print and set the best parameters
        print("BEST PARAMETERS: ", end='')
        print(best_parameters)
        parameters = {
            'lr': best_parameters["lr"],
            'weight_decay': best_parameters["weight_decay"],
            'momentum': best_parameters["momentum"]
        }

        # Train model with found parameters
        model = torch.load(checkpoint_path)
        train_pending(parameters, epochs)
        model.freeze_pending()

    # Train output layer
    model.add_pending(ResnetOutput(16, 10))
    optimizer = utils.get_optimizer(
        opt,
        params=model.get_trainable_params(),
        lr=parameters['lr'],
        weight_decay=parameters['weight_decay'],
        momentum=parameters['momentum'])
    trainer = Trainer(
        opt=opt,
        model=model,
        set_eval=None,
        optimizer=optimizer,
        val_metric_name=opt.hidden_objective,
        val_metric_obj='max')
    train_output(
        opt,
        n_epochs=output_epochs,
        trainer=trainer,
        loader=loader,
        val_loader=val_loader,
        criterion=output_criterion,
        part_id=model.n_modules,
        device=device)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    main()
