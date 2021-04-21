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
from network.models.morph import Block, Cell, OutputCell, Morph
from network.parsers.argument_parser import ArgumentParser
from network.trainers import train_hidden, train_output, Trainer


data_channels = 3
in_channels = 16
epochs = 150
output_epochs = 20
max_layers = 5
checkpoint_path = "./checkpoint/bayesian_start.pth"
best_path = "./checkpoint/best.pth"
useful_layer = True

opt = None
model = None
hidden_criterion = None
output_criterion = None
loader = None
val_loader = None
device = None
parameters = {
    'lr': 0.1
}

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


def get_module(parameterization):
    net = torch.load(checkpoint_path)
    if (net.n_modules == 0):
        net.add_pending(torch.nn.Conv2d(data_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False).to(device))
        net.add_pending(torch.nn.BatchNorm2d(in_channels).to(device))
    cell = Cell(in_channels=in_channels, use_residual=True, blocks=[
        Block(in_channels, parameterization.get('out1', 16), 3, device),
        Block(parameterization.get('out1', 16), parameterization.get('out2', 16), 3, device),
        Block(in_channels, parameterization.get('out3', 16), 5, device),
        Block(parameterization.get('out3', 16), parameterization.get('out4', 16), 5, device),
        Block(in_channels, parameterization.get('out5', 16), 7, device),
        Block(parameterization.get('out5', 16), parameterization.get('out6', 16), 7, device),
    ], device=device)
    net.add_pending(cell)
    return net


def train_evaluate(parameterization):
    print(f'Optimizing - Params {parameterization}')

    net = get_module(parameterization)
    parameters = {
        'lr': parameterization.get('lr', 0.1)
    }
    val_result = train_pending(net, parameters, 10)

    resource_constraint = 0
    for i in range(6):
        if (i < 2):
            kernel_size = 9
        elif (i < 4):
            kernel_size = 25
        else:
            kernel_size = 49
        resource_constraint += (parameterization['out' + str(i + 1)] * kernel_size)
    resource_constraint *= 3e-6

    print(f'Optimizing - Alignment: {val_result}, ResConst: {resource_constraint}', end='')
    val_result -= resource_constraint
    print(f', Metric: {val_result}')

    return val_result


def train_pending(net, parameters, epochs, save_model=False):
    optimizer = torch.optim.Adam(net.get_trainable_params(), lr=parameters['lr'])
    trainer = Trainer(
        opt=opt,
        model=net,
        set_eval=None,
        optimizer=optimizer,
        val_metric_name=opt.hidden_objective)
    return train_hidden(
        opt,
        n_epochs=epochs,
        trainer=trainer,
        loader=loader,
        val_loader=val_loader,
        criterion=hidden_criterion,
        part_id=net.n_modules + 1,
        device=device,
        save_model=save_model)


def main():
    global opt
    global loader
    global val_loader
    global model
    global hidden_criterion
    global output_criterion
    global parameters

    opt = ArgumentParser().parse()
    utils.set_logger(opt=opt, filename='train.log', filemode='w')
    if opt.seed:
        utils.make_deterministic(opt.seed)
    loader, val_loader = datasets.get_dataloaders(opt)
    best_val_accuracy = 0
    model = Morph(device).to(device)

    hidden_criterion = get_hidden_criterion(opt)
    output_criterion = torch.nn.CrossEntropyLoss() if opt.loss == 'xe' else torch.nn.MultiMarginLoss()

    for i in range(max_layers):
        # Save current model for resetting on bayesian trials and after
        torch.save(model, checkpoint_path)

        # Search for parameters
        best_parameters, _, _, _ = optimize(
            parameters=[
                {"name": "lr", "type": "range", "bounds": [0.001, 0.3], "log_scale": True},
                {"name": "out1", "type": "range", "bounds": [8, 64]},
                {"name": "out2", "type": "range", "bounds": [8, 64]},
                {"name": "out3", "type": "range", "bounds": [8, 64]},
                {"name": "out4", "type": "range", "bounds": [8, 64]},
                {"name": "out5", "type": "range", "bounds": [8, 64]},
                {"name": "out6", "type": "range", "bounds": [8, 64]},
            ],
            total_trials=30,
            evaluation_function=train_evaluate,
            objective_name=opt.hidden_objective,
        )

        # Print and set the best parameters
        print("BEST PARAMETERS: ", end='')
        print(best_parameters)
        parameters = {
            'lr': best_parameters["lr"]
        }

        # Reset model
        model = torch.load(checkpoint_path)

        # Add components with best parameters
        if (model.n_modules == 0):
            model.add_pending(torch.nn.Conv2d(data_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False).to(device))
            model.add_pending(torch.nn.BatchNorm2d(in_channels).to(device))
        cell = Cell(in_channels=in_channels, use_residual=True, blocks=[
            Block(in_channels, best_parameters['out1'], 3, device),
            Block(best_parameters['out1'], best_parameters['out2'], 3, device),
            Block(in_channels, best_parameters['out3'], 5, device),
            Block(best_parameters['out3'], best_parameters['out4'], 5, device),
            Block(in_channels, best_parameters['out5'], 7, device),
            Block(best_parameters['out5'], best_parameters['out6'], 7, device),
        ], device=device)
        model.add_pending(cell)

        # Train for given epochs and then freeze
        new_accuracy = train_pending(model, parameters, epochs, True)

        print(f'new_acc: {new_accuracy} best_acc: {best_val_accuracy}')
        if new_accuracy > best_val_accuracy:
            print(f'Better acc from new layer:')
            print(f'{new_accuracy} > {best_val_accuracy}')
            best_val_accuracy = new_accuracy
            model = torch.load(best_path)
            model.freeze_pending()
        else:
            print('New layer did not improve upon previous, STOPPING HIDDEN TRAINING')
            model.clear_pending()
            break

    # Train output layer
    model.add_pending(OutputCell(in_channels, opt.n_classes))
    optimizer = torch.optim.Adam(model.get_trainable_params(), lr=parameters['lr'])
    trainer = Trainer(
        opt=opt,
        model=model,
        set_eval=None,
        optimizer=optimizer,
        val_metric_name=opt.hidden_objective,
        val_metric_obj='max')
    train_output(
        opt=opt,
        n_epochs=output_epochs,
        trainer=trainer,
        loader=loader,
        val_loader=val_loader,
        criterion=output_criterion,
        part_id=model.n_modules,
        device=device)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("USING DEVICE " + device)
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    main()
