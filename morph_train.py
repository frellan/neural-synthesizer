import os
import functools
import torch
import numpy as np
from ax.service.managed_loop import optimize
from ax.service.ax_client import AxClient

import network.utils as utils
import network.models.srs_loss as losses
import network.datasets as datasets
from network.models.morph import Block, DepthwiseBlock, Cell, OutputCell, Morph
from network.parsers.argument_parser import ArgumentParser
from network.trainers import train_hidden, train_output, Trainer

data_channels = 3
in_channels = 16
epochs = 200
output_epochs = 100
opt_trials = 30
opt_epochs_per_trial = 10
max_layers = 5
data_dir = os.path.abspath("./data")
checkpoint_dir = os.path.abspath("./checkpoint")
opt_start_path = "./checkpoint/opt_start.pth"
opt_checkpoint_path = "./checkpoint/opt_checkpoint.pth"
best_path = "./checkpoint/best.pth"
useful_layer = True

opt = None
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


def get_module(config):
    net = torch.load(os.path.join(checkpoint_dir, "opt_start.pth"))
    if (net.module.n_modules == 0):
        net.module.add_pending(torch.nn.Conv2d(data_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False).to(device))
        net.module.add_pending(torch.nn.BatchNorm2d(in_channels).to(device))
    cell = Cell(in_channels=in_channels, use_residual=True, blocks=[
        Block(in_channels, config['out1'], 3),
        Block(config['out1'], config['out2'], 3),
        Block(in_channels, config['out3'], 5),
        Block(config['out3'], config['out4'], 5),
        Block(in_channels, config['out5'], 7),
        Block(config['out5'], config['out6'], 7),
    ]).to(device)
    net.module.add_pending(cell)
    print(f"Trying a model with {net.module.n_params()} params, of which {net.module.n_trainable_params()} are trainable")
    return net


def train_evaluate(config):
    print(f'Optimizing - Config: {config}')

    net = get_module(config)
    val_result = train_pending(net, { 'lr': config['lr'] }, opt_epochs_per_trial)

    res_constraint = 0
    for i in range(6):
        if (i < 2):
            kernel_size = 9
        elif (i < 4):
            kernel_size = 25
        else:
            kernel_size = 49
        res_constraint += (config['out' + str(i + 1)] * kernel_size)
    res_constraint *= 3e-6

    print(f'Optimizing - Alignment: {val_result}, ResConst: {res_constraint}', end='')
    val_result -= res_constraint
    print(f', Metric: {val_result}')

    return val_result


def train_pending(net, parameters, epochs, save_model=False):
    optimizer = torch.optim.Adam(net.module.get_trainable_params(), lr=parameters['lr'])
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
        part_id=net.module.n_modules + 1,
        device=device,
        save_model=save_model)


def main():
    global opt
    global loader
    global val_loader
    global hidden_criterion
    global output_criterion

    opt = ArgumentParser().parse()
    utils.set_logger(opt=opt, filename='train.log', filemode='w')
    if opt.seed:
        utils.make_deterministic(opt.seed)
    loader, val_loader = datasets.get_dataloaders(opt)
    best_val_accuracy = 0
    net = Morph()
    Morph().to(device)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    hidden_criterion = get_hidden_criterion(opt)
    output_criterion = torch.nn.CrossEntropyLoss()

    for i in range(max_layers):
        # Save current net for resetting on bayesian trials and after
        torch.save(net, opt_start_path)

        # Search for parameters
        params, _, _, _ = optimize(
            parameters=[
                {'name': 'lr', 'type': 'range', 'value_type': 'float', 'bounds': [0.001, 0.2], 'log_scale': True},
                {'name': 'out1', 'type': 'choice', 'value_type': 'int', 'values': [4, 8, 16, 32, 64, 128]},
                {'name': 'out2', 'type': 'choice', 'value_type': 'int', 'values': [4, 8, 16, 32, 64, 128]},
                {'name': 'out3', 'type': 'choice', 'value_type': 'int', 'values': [4, 8, 16, 32, 64, 128]},
                {'name': 'out4', 'type': 'choice', 'value_type': 'int', 'values': [4, 8, 16, 32, 64, 128]},
                {'name': 'out5', 'type': 'choice', 'value_type': 'int', 'values': [4, 8, 16, 32, 64, 128]},
                {'name': 'out6', 'type': 'choice', 'value_type': 'int', 'values': [4, 8, 16, 32, 64, 128]},
            ],
            total_trials=opt_trials,
            evaluation_function=train_evaluate,
            objective_name=opt.hidden_objective,
        )
        print("BEST PARAMETERS: ", params)

        # Reset net
        net = torch.load(opt_start_path)

        # Add components with best parameters
        if (net.module.n_modules == 0):
            net.module.add_pending(torch.nn.Conv2d(data_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False).to(device))
            net.module.add_pending(torch.nn.BatchNorm2d(in_channels).to(device))
        cell = Cell(in_channels=in_channels, use_residual=True, blocks=[
            Block(in_channels, params['out1'], 3),
            Block(params['out1'], params['out2'], 3),
            Block(in_channels, params['out3'], 5),
            Block(params['out3'], params['out4'], 5),
            Block(in_channels, params['out5'], 7),
            Block(params['out5'], params['out6'], 7),
        ]).to(device)
        net.module.add_pending(cell)

        # Train for given epochs and then freeze
        new_accuracy = train_pending(net, { 'lr': params['lr'] }, epochs, True)

        print(f'new_acc: {new_accuracy}, best_acc: {best_val_accuracy}')
        if new_accuracy > 1.01 * best_val_accuracy: # Be better with at least 1%
            print(f'Better acc from new layer:')
            print(f'{new_accuracy} > {best_val_accuracy}')
            best_val_accuracy = new_accuracy
            net = torch.load(best_path)
            net.module.freeze_pending()
        else:
            print('New layer did not improve upon previous, STOPPING HIDDEN TRAINING')
            net.module.clear_pending()
            net.module.frozen[-1].use_residual = False
            break

    print(f"TOTAL PARAMS FOR HIDDEN LAYERS: {net.module.n_params()}")

    # Unfreeze entire network and train output layer
    net.module.add_pending(OutputCell(in_channels, opt.n_classes).to(device))
    net.module.unfreeze_network()
    optimizer = torch.optim.Adam(net.module.get_all_trainable_params(), lr=0.01)
    trainer = Trainer(
        opt=opt,
        model=net,
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
        part_id=net.module.n_modules + 1,
        device=device)

    print(f"TOTAL PARAMS FOR ENTIRE NETWORK: {net.module.n_params()}")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using Device:", device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("Using GPUs:", torch.cuda.device_count())
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    main()
