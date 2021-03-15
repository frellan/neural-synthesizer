"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Modular training example.
"""

import torch

import kernet.utils as utils
import kernet.models as models
import kernet.layers.loss as losses
import kernet.datasets as datasets
from kernet.parsers import TrainParser
from kernet.trainers.trainer import Trainer
from kernet.engines import train_hidden, train_output


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
    parser.add_argument('--use_proj_head', type=utils.str2bool,
                        nargs='?', const=True, default=False,
                        help='Whether to attach a trainable two-layer MLP projection head to the ' +
                             'output of the hidden modules during training. If added, the heads project ' +
                        'all activations to the same Euclidean space with dimension determined by head_size.')
    parser.add_argument('--split_mode', type=int, default=1,
                        help='The mode to perform the split. Effective only for certain networks.')
    parser.add_argument('--head_size', type=int, default=512,
                        help='Output size of the projection head.')
    n_parts = kwargs["n_parts"]
    for i in range(1, n_parts + 1):
        parser.add_argument('--lr{}'.format(i), type=float, default=1e-3,
                            help='Initial learning rate for part {}.'.format(i))
        parser.add_argument('--momentum{}'.format(i), type=float, default=.9,
                            help='Momentum for the SGD optimizer for part {}.'.format(i))
        parser.add_argument('--weight_decay{}'.format(i), type=float, default=5e-4,
                            help='L2 regularization on the model weights for part {}.'.format(i))
        parser.add_argument('--n_epochs{}'.format(i), type=int, default=200,
                            help='The number of training epochs for part {}.'.format(i))
    return parser


def main():
    opt = TrainParser().parse()

    # set up logger
    utils.set_logger(opt=opt, filename='train.log', filemode='w')

    if opt.seed:
        utils.make_deterministic(opt.seed)
    loader, val_loader = datasets.get_dataloaders(opt)

    model = models.get_model(opt)
    model = model.to(device)
    modules, params = model.split(n_parts=opt.n_parts, mode=opt.split_mode)

    trainer_cls = Trainer

    output_layer = list(model.children())[-1]
    selected_loss_fn = getattr(losses, opt.hidden_objective)
    hidden_criterion = selected_loss_fn(output_layer.phi, opt.n_classes)
    if opt.loss == 'xe':
        output_criterion = torch.nn.CrossEntropyLoss()
    elif opt.loss == 'hinge':
        output_criterion = torch.nn.MultiMarginLoss()

    optimizers, trainers = [], []
    for i in range(1, opt.n_parts + 1):
        optimizers.append(utils.get_optimizer(
            opt,
            params=params[i - 1],
            lr=getattr(opt, 'lr{}'.format(i)),
            weight_decay=getattr(opt, 'weight_decay{}'.format(i)),
            momentum=getattr(opt, 'momentum{}'.format(i))
        ))
        trainer = trainer_cls(
            opt=opt,
            model=modules[i - 1],
            set_eval=modules[i - 2] if i > 1 else None,
            optimizer=optimizers[i - 1],
            val_metric_name=opt.hidden_objective if i < opt.n_parts else 'acc',
            val_metric_obj='max'
        )
        trainers.append(trainer)

    # train the first hidden module
    train_hidden(
        opt,
        n_epochs=opt.n_epochs1,
        trainer=trainers[0],
        loader=loader,
        val_loader=val_loader,
        criterion=hidden_criterion,
        part_id=1,
        device=device)

    # train other hidden modules
    for i in range(2, opt.n_parts):
        # exclude certain network part(s) from the graph to make things faster
        utils.exclude_during_backward(modules[i - 2])
        train_hidden(
            opt,
            n_epochs=getattr(opt, 'n_epochs{}'.format(i)),
            trainer=trainers[i - 1],
            loader=loader,
            val_loader=val_loader,
            criterion=hidden_criterion,
            part_id=i,
            device=device
        )

    # train output layer
    utils.exclude_during_backward(modules[-2])
    train_output(
        opt,
        n_epochs=getattr(opt, 'n_epochs{}'.format(opt.n_parts)),
        trainer=trainers[-1],
        loader=loader,
        val_loader=val_loader,
        criterion=output_criterion,
        part_id=opt.n_parts,
        device=device
    )
    utils.include_during_backward(modules[-2])


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    main()
