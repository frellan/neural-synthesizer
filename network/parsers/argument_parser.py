"""""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Modified from:
  https://github.com/NVlabs/SPADE/blob/master/options/base_options.py
"""

import os
import sys
import logging
import argparse

import network.utils as utils
import network.datasets as datasets

logger = logging.getLogger()


class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


class ArgumentParser:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataset', choices=[
            'mnist',
            'cifar10',
            'cifar100',
            'fashionmnist',
            'svhn'
        ] + list(datasets.CIFAR10_2.keys()), default='mnist', help='Dataset name.')
        parser.add_argument('--activation', choices=['tanh', 'sigmoid', 'relu'], default='tanh',
            help='Activation function. Not used by certain models such as the ResNets.')
        parser.add_argument('--in_channels', type=int, default=3,
            help='The number of input channels of the network.')
        parser.add_argument('--batch_size', type=int, default=128,
            help='Batch size for training and testing.')
        parser.add_argument('--n_workers', type=int, default=2,
            help='The number of workers for data loading during training and testing.')
        parser.add_argument('--normalize_mean', type=str,
            help='Comma separated channel means for data normalization')
        parser.add_argument('--normalize_std', type=str,
            help='Comma separated channel standard deviations for data normalization')
        parser.add_argument('--max_testset_size', type=int, default=int(1e12),
            help='Max size for the test set.')
        parser.add_argument('--balanced', type=utils.str2bool,
            nargs='?', const=True, default=False,
            help='If set to True, will sample with balanced classes when either ' +
            'the train set or the test set is constrained to be a random subset of ' +
            'the actual train/test set.')
        parser.add_argument('--multi_gpu', type=utils.str2bool,
            nargs='?', const=True, default=False,
            help='Whether to use multiple (all) available GPUs.')
        parser.add_argument('--loglevel', type=str, default='INFO',
            help='Logging level above which the logs will be displayed.')
        parser.add_argument('--n_parts', type=int, default=2,
            help='The number of parts to split the network into when performing modular training/testing.')
        parser.add_argument('--optimizer', choices=['sgd', 'adam','samsgd'], default='adam',
            help='The optimizer to be used.')
        parser.add_argument('--loss', choices=['xe', 'hinge', 'nll'], default='xe',
            help='The overall loss function to be used. xe is for CrossEntropyLoss, hinge for ' +
            'MultiMarginLoss (multi-class hinge loss), nll is for NLLLoss (CrossEntropyLoss w/o logsoftmax).')
        parser.add_argument('--shuffle', type=utils.str2bool,
            nargs='?', const=True, default=True,
            help='Whether to shuffle data across training epochs. "True" or "t" will be parsed as True (bool); "False" or "f" as False. Same works for all params of bool type.')
        parser.add_argument('--augment_data', type=utils.str2bool,
            nargs='?', const=True, default=False,
            help='If True, augment training data. See datasets/__init__.py for the specific augmentations used.')
        parser.add_argument('--train_subset_indices', type=str,
            help='Path to saved training subset indices, if available.')
        parser.add_argument('--print_freq', type=int, default=100,
            help='Print training statistics once every this many mini-batches.')
        parser.add_argument('--n_classes', type=int,
            help='The number of classes in the data.')
        parser.add_argument('--seed', type=int,
            help='Random seed. If specified, training will be (mostly) deterministic.')
        parser.add_argument('--tf_log', type=utils.str2bool,
            nargs='?', const=True, default=True,
            help='Whether to log training statistics with tensorboard.')
        parser.add_argument('--schedule_lr', type=utils.str2bool,
            nargs='?', const=True, default=False,
            help='Whether to schedule learning rate with the ReduceLROnPlateau scheduler.')
        parser.add_argument('--lr_schedule_factor', type=float, default=.1,
            help='The factor argument passed to the scheduler.')
        parser.add_argument('--lr_schedule_patience', type=int, default=10,
            help='The patience argument passed to the scheduler.')
        parser.add_argument('--val_freq', type=int, default=1,
            help='Validate once every this many epochs.')
        parser.add_argument('--max_trainset_size', type=int, default=int(1e12),
            help='Max size for the training set.')
        parser.add_argument('--n_val', type=int, default=100,
            help='Validation set size. The validation set will be taken from a randomly permuted training set. Can set to 0 if do not need a validation set.')
        parser.add_argument('--dataset_rand_idx', type=str,
            help='Path to saved permuted dataset indices (used for selecting validation set), if available.')
        parser.add_argument('--max_ori_trainset_size', type=int, default=int(1e12),
            help='Sample size in the first (optional) random sampling on training set. ' +
            'This sampling is done before any other processing on the training set such ' +
            'as train/val split.')
        parser.add_argument('--ori_train_subset_indices', type=str,
            help='Path to saved subset indices for the first random sampling, if available.')
        parser.add_argument('--ori_balanced', type=utils.str2bool,
            nargs='?', const=True, default=False,
            help='If True, the first random sampling will try to sample an equal number of examples ' +
            'from each class.')

        self.initialized = True
        self.is_train = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            # some options are not added after the modifiers below but have names that are prefixes of some other options
            # added before them. Letting allow_abbrev be True would result in errors in these cases
            # because the parser would incorrectly parse the former as the latter
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                allow_abbrev=False
            )
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify dataset-related parser options
        dataset = opt.dataset
        dataset_option_setter = datasets.get_option_setter(dataset)
        parser = dataset_option_setter(parser, is_train=self.is_train)

        # modify module-specific parser options
        full_script_name = sys.argv[0]
        path, script_name = os.path.split(full_script_name)
        module_name, _ = os.path.splitext(script_name)
        with add_path(path):
            s = __import__(module_name)
            if hasattr(s, 'modify_commandline_options'):
                parser = s.modify_commandline_options(
                    parser, is_train=self.is_train, n_parts=opt.n_parts)
            del sys.modules[module_name]  # clean up

        opt, unknown = parser.parse_known_args()

        opt = parser.parse_args()

        self.parser = parser
        return opt

    def traverse_options(self, opt, message=''):
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        return message

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        message += self.traverse_options(opt, message)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        opt.is_train = self.is_train

        self.print_options(opt)

        # get lists from strings
        opt.normalize_mean = [float(_) for _ in opt.normalize_mean.split(',')]
        opt.normalize_std = [float(_) for _ in opt.normalize_std.split(',')]

        if hasattr(opt, 'adversary_norm'):
            if opt.adversary_norm == 'inf':
                import numpy as np
                opt.adversary_norm = np.inf
            elif opt.adversary_norm == '2':
                opt.adversary_norm = 2
        if hasattr(opt, 'pgd_norm'):
            if opt.pgd_norm == 'inf':
                import numpy as np
                opt.pgd_norm = np.inf
            elif opt.pgd_norm == '2':
                opt.pgd_norm = 2

        self.opt = opt
        return self.opt
