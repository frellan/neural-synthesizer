"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Modular training example.
"""

import torch

import network.utils as utils
import network.models as models
import network.models.srs_loss as losses
import network.datasets as datasets
from network.models import Flatten, UnitVectorize
from network.models.morph import Morph
from network.parsers.argument_parser import ArgumentParser
from network.trainers import train_hidden, train_output, Trainer


loss_names = [
    'srs_raw',
    'srs_nmse',
    'srs_alignment',
    'srs_upper_tri_alignment',
    'srs_contrastive',
    'srs_log_contrastive'
]


def modify_commandline_options(parser, **kwargs):
    parser.add_argument('--hidden_objective',
        choices=loss_names + [_ + '_neo' for _ in loss_names],
        default='srs_alignment',
        help='Proxy hidden objective.')
    return parser


def main():
    opt = ArgumentParser().parse()

    # set up logger
    utils.set_logger(opt=opt, filename='train.log', filemode='w')

    if opt.seed:
        utils.make_deterministic(opt.seed)
    loader, val_loader = datasets.get_dataloaders(opt)

    model = Morph(opt, device).to(device)

    lr = .1
    weight_decay = .0002
    momentum = .9
    epochs = 30

    hidden_criterion = get_hidden_criterion(opt)
    output_criterion = torch.nn.CrossEntropyLoss() if opt.loss == 'xe' else torch.nn.MultiMarginLoss()

    model.add_pending(
        torch.nn.Conv2d(1, 6, 5, padding=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2))
    model.add_pending(
        torch.nn.Conv2d(6, 16, 5),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        Flatten())
    model.add_pending(torch.nn.Linear(400, 120))
    model.add_pending(torch.nn.ReLU())
    model.add_pending(torch.nn.Linear(120, 84))

    utils.default_init_weights(model)

    train_pending(opt, model, lr, weight_decay, momentum, epochs,
        loader, val_loader, hidden_criterion, device)
    model.freeze_pending()

    model.add_pending(torch.nn.ReLU())
    model.add_pending(UnitVectorize())
    model.add_pending(torch.nn.Linear(84, 10))

    # train output layer
    optimizer = utils.get_optimizer(opt, params=model.get_trainable_params(), lr=lr,
        weight_decay=weight_decay, momentum=momentum)
    trainer = Trainer(opt=opt, model=model, set_eval=None, optimizer=optimizer,
        val_metric_name=opt.hidden_objective, val_metric_obj='max')
    train_output(opt, n_epochs=epochs, trainer=trainer, loader=loader, val_loader=val_loader,
        criterion=output_criterion, part_id=model.n_modules, device=device)

def get_hidden_criterion(opt):
    selected_loss_fn = getattr(losses, opt.hidden_objective)
    return selected_loss_fn(opt.activation, opt.n_classes)

def train_pending(opt, model, lr, weight_decay, momentum, epochs, loader, val_loader, hidden_criterion, device):
    optimizer = utils.get_optimizer( opt, params=model.get_trainable_params(), lr=lr,
        weight_decay=weight_decay, momentum=momentum)
    trainer = Trainer(opt=opt, model=model, set_eval=None, optimizer=optimizer,
        val_metric_name=opt.hidden_objective, val_metric_obj='max')
    train_hidden(opt, n_epochs=epochs, trainer=trainer, loader=loader, val_loader=val_loader,
        criterion=hidden_criterion, part_id=model.n_modules + 1, device=device)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    main()
