"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
# import pickle
import logging

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset

from network import utils

logger = logging.getLogger()

CIFAR10_2 = {
    'cifar10deau': ['deer', 'automobile'],
    'cifar10hotr': ['horse', 'truck'],
    'cifar10detr': ['deer', 'truck'],
    'cifar10hoau': ['horse', 'automobile'],
    'cifar10deca': ['deer', 'cat'],
    'cifar10hodo': ['horse', 'dog'],
    'cifar10dedo': ['deer', 'dog'],
    'cifar10hoca': ['horse', 'cat'],
    'cifar10auca': ['automobile', 'cat'],
    'cifar10trdo': ['truck', 'dog'],
    'cifar10audo': ['automobile', 'dog'],
    'cifar10trca': ['truck', 'cat'],
    'cifar10deho': ['deer', 'horse'],
    'cifar10autr': ['automobile', 'truck'],
    'cifar10cado': ['cat', 'dog']
}
NAMES = locals()


def get_option_setter(dataset_name: str):
    fn_name = dataset_name + '_modify_commandline_options'
    if fn_name in NAMES:
        return NAMES[fn_name]
    else:
        raise NotImplementedError()


def get_dataloaders(opt):
    if opt.is_train:
        tr = [
            transforms.ToTensor(),
            transforms.Normalize(opt.normalize_mean, opt.normalize_std)
        ]

        if getattr(opt, 'augment_data', None):
            # opt will have augment_data as an attr during training. During testing,
            # the only case in which training data will be loaded will be for the
            # kernelized modules to get centers. These modules will load saved centers
            # and overwrite the centers loaded here anyway so this expression is safe
            tr_extra = [
                transforms.RandomCrop(32 if (opt.dataset.startswith(
                    'cifar10') or opt.dataset in ['svhn']) else 28, padding=4),
                transforms.RandomHorizontalFlip()
            ]
            if opt.dataset == 'cifar100':
                tr_extra.append(transforms.RandomRotation(15))
            tr = tr_extra + tr

        train_transform = transforms.Compose(tr)

        if opt.dataset == 'cifar100':
            trainset = torchvision.datasets.CIFAR100(
                root='./data', train=True,
                download=True, transform=train_transform)
        elif opt.dataset.startswith('cifar10'):
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True,
                download=True, transform=train_transform)
            if opt.dataset in CIFAR10_2:
                trainset = utils.get_cifar10_subset(
                    trainset, CIFAR10_2[opt.dataset])
        elif opt.dataset == 'mnist':
            trainset = torchvision.datasets.MNIST(
                root='./data', train=True,
                download=True, transform=train_transform)
        elif opt.dataset == 'fashionmnist':
            trainset = torchvision.datasets.FashionMNIST(
                root='./data', train=True,
                download=True, transform=train_transform)
        elif opt.dataset == 'svhn':
            trainset = torchvision.datasets.SVHN(
                root='./data', split='train',
                download=True, transform=train_transform)
        else:
            raise NotImplementedError()

        logger.info("dataset [train: %s] was created" % (
            type(trainset).__name__,
        ))
        logger.debug("transformations:\n" + str(train_transform))

        if opt.n_val > 0:
            # for regular datasets, the validation set is simply the last opt.n_val elements in a randomly
            # permuted training set. Random permutation is needed to break the original order of the
            # dataset. This is critical for datasets that are ordered by, e.g., class labels
            if opt.n_val > len(trainset):
                raise ValueError(
                    'Validation set size cannot exceed that of training set.')
            rand_idx = torch.randperm(len(trainset)).tolist()
            # save rand_idx for reproducibility
            # file_name = os.path.join(opt.save_dir, 'dataset_rand_idx')
            # with open(file_name + '.txt', 'wt') as f:
            #     f.write(str(rand_idx))
            # with open(file_name + '.pkl', 'wb') as f:
            #     pickle.dump(rand_idx, f)
            trainset, valset = Subset(
                trainset, rand_idx[:-opt.n_val]), Subset(trainset, rand_idx[-opt.n_val:])
        
        def _init_fn(worker_id):
            import numpy as np
            np.random.seed(int(opt.seed))
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=opt.batch_size,
            shuffle=opt.shuffle, num_workers=opt.n_workers,
            pin_memory=True, worker_init_fn=_init_fn)
        if opt.n_val > 0:
            val_loader = torch.utils.data.DataLoader(
                valset, batch_size=opt.batch_size,
                shuffle=opt.shuffle, num_workers=opt.n_workers,
                pin_memory=True, worker_init_fn=_init_fn)
        else:
            val_loader = None

        return train_loader, val_loader
    else:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(opt.normalize_mean, opt.normalize_std)
        ])

        if opt.dataset == 'cifar100':
            testset = torchvision.datasets.CIFAR100(
                root='./data', train=False,
                download=True, transform=test_transform)
        elif opt.dataset.startswith('cifar10'):
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False,
                download=True, transform=test_transform)
            if opt.dataset in CIFAR10_2:
                testset = utils.get_cifar10_subset(
                    testset, CIFAR10_2[opt.dataset])
        elif opt.dataset == 'mnist':
            testset = torchvision.datasets.MNIST(
                root='./data', train=False,
                download=True, transform=test_transform)
        elif opt.dataset == 'fashionmnist':
            testset = torchvision.datasets.FashionMNIST(
                root='./data', train=False,
                download=True, transform=test_transform)
        elif opt.dataset == 'svhn':
            testset = torchvision.datasets.SVHN(
                root='./data', split='test',
                download=True, transform=test_transform)
        else:
            raise NotImplementedError()

        logger.info("dataset [test: %s] was created" % (
            type(testset).__name__
        ))
        logger.debug("transformations:\n" + str(test_transform))
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=opt.batch_size,
            shuffle=False, num_workers=opt.n_workers,
            pin_memory=True)

        return test_loader


#########
# option modifiers
#########


def cifar10_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=10)
    parser.set_defaults(normalize_mean='0.49139968,0.48215841,0.44653091')
    parser.set_defaults(normalize_std='0.24703223,0.24348513,0.26158784')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


# TODO stds of these binary classification variants of CIFAR-10 are incorrect (they were
# computed with an older version of utils.data.get_mean_and_std)
def cifar10deau_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=2)
    parser.set_defaults(normalize_mean='0.4714,0.4599,0.4127')
    parser.set_defaults(normalize_std='0.2030,0.1974,0.1928')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


def cifar10hotr_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=2)
    parser.set_defaults(normalize_mean='0.5003,0.4826,0.4475')
    parser.set_defaults(normalize_std='0.2258,0.2288,0.2337')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


def cifar10detr_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=2)
    parser.set_defaults(normalize_mean='0.4851,0.4753,0.4281')
    parser.set_defaults(normalize_std='0.2064,0.2044,0.2028')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


def cifar10hoau_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=2)
    parser.set_defaults(normalize_mean='0.4866,0.4672,0.4320')
    parser.set_defaults(normalize_std='0.2223,0.2218,0.2237')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


def cifar10deca_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=2)
    parser.set_defaults(normalize_mean='0.4835,0.4608,0.3969')
    parser.set_defaults(normalize_std='0.1911,0.1862,0.1803')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


def cifar10hodo_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=2)
    parser.set_defaults(normalize_mean='0.5009,0.4723,0.4167')
    parser.set_defaults(normalize_std='0.2110,0.2108,0.2115')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


def cifar10dedo_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=2)
    parser.set_defaults(normalize_mean='0.4858,0.4649,0.3974')
    parser.set_defaults(normalize_std='0.1916,0.1863,0.1805')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


def cifar10hoca_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=2)
    parser.set_defaults(normalize_mean='0.4987,0.4681,0.4162')
    parser.set_defaults(normalize_std='0.2104,0.2107,0.2113')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


def cifar10auca_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=2)
    parser.set_defaults(normalize_mean='0.4833,0.4555,0.4314')
    parser.set_defaults(normalize_std='0.2223,0.2186,0.2198')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


def cifar10trdo_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=2)
    parser.set_defaults(normalize_mean='0.4993,0.4750,0.4473')
    parser.set_defaults(normalize_std='0.2263,0.2256,0.2300')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


def cifar10audo_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=2)
    parser.set_defaults(normalize_mean='0.4856,0.4596,0.4319')
    parser.set_defaults(normalize_std='0.2228,0.2187,0.2199')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


def cifar10trca_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=2)
    parser.set_defaults(normalize_mean='0.4971,0.4709,0.4468')
    parser.set_defaults(normalize_std='0.2257,0.2255,0.2298')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


def cifar10deho_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=2)
    parser.set_defaults(normalize_mean='0.4868,0.4725,0.3975')
    parser.set_defaults(normalize_std='0.1911,0.1895,0.1843')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


def cifar10autr_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=2)
    parser.set_defaults(normalize_mean='0.4849,0.4699,0.4626')
    parser.set_defaults(normalize_std='0.2376,0.2367,0.2422')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


def cifar10cado_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=2)
    parser.set_defaults(normalize_mean='0.4977,0.4605,0.4160')
    parser.set_defaults(normalize_std='0.2109,0.2075,0.2075')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


def mnist_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=10)
    parser.set_defaults(normalize_mean='0.1307')
    parser.set_defaults(normalize_std='0.3081')
    parser.set_defaults(normalize_mean='0.1307')
    parser.set_defaults(normalize_std='0.3081')
    parser.set_defaults(data_shape='(1, 28, 28)')
    return parser


def fashionmnist_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=10)
    parser.set_defaults(normalize_mean='0.2861')
    parser.set_defaults(normalize_std='0.3530')
    parser.set_defaults(data_shape='(1, 28, 28)')
    return parser


def cifar100_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=100)
    parser.set_defaults(normalize_mean='0.50707516,0.48654887,0.44091784')
    parser.set_defaults(normalize_std='0.26733429,0.25643846,0.27615047')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser


def svhn_modify_commandline_options(parser, **kwargs):
    parser.set_defaults(n_classes=10)
    parser.set_defaults(normalize_mean='0.4376821,0.4437697,0.47280442')
    parser.set_defaults(normalize_std='0.19803012,0.20101562,0.19703614')
    parser.set_defaults(data_shape='(3, 32, 32)')
    return parser
