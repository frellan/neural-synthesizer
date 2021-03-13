"""""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Modified from:
  https://github.com/NVlabs/SPADE/blob/master/options/base_options.py
"""

import os
import sys
import pickle
import logging
import datetime
import argparse
import importlib

import kernet.utils as utils
import kernet.models as models
import kernet.datasets as datasets

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


class BaseParser:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100', 'fashionmnist', 'svhn'] + list(datasets.CIFAR10_2.keys()),
                            default='mnist', help='Dataset name.')
        parser.add_argument('--model', choices=[
            'kmlp',
            'lenet5',
            'k1lenet5', 'k2lenet5', 'k3lenet5'
        ], default='lenet5',
            help='Model name. The (k)ResNets are for 3-channel images only.')
        parser.add_argument('--activation', choices=['tanh', 'sigmoid', 'relu', 'gaussian', 'reapen'], default='tanh',
                            help='Model activation/kernel function. Not used by certain models such as the ResNets.')
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
        parser.add_argument('--load_opt', action='store_true',
                            help='If specified, load options from a saved file.')
        parser.add_argument('--opt_file', type=str,
                            help='A saved .pkl file to load options from.')
        parser.add_argument('--load_model', action='store_true',
                            help='If specified, load a saved model. Testing always loads model so there is no need for this flag.')
        parser.add_argument('--checkpoint_dir', type=str,
                            help='A folder where the saved model lives.')
        parser.add_argument('--save_dir', type=str, default='./checkpoint/',
                            help='A folder to save things.')
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

        self.initialized = True
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

        # if there is opt_file, load it.
        # The previous default options will be overwritten.
        if opt.load_opt:
            parser = self.update_options_from_file(parser, opt)
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        if model_option_setter:
            parser = model_option_setter(parser, is_train=self.is_train)

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

        # this second loading is for updating the params
        # modified by model/dataset/script-specific parser modifiers
        if opt.load_opt:
            parser = self.update_options_from_file(parser, opt)
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

    def option_file_path(self, opt, makedir=False):
        if makedir:
            if os.path.exists(opt.save_dir):
                sd = opt.save_dir[:-
                                  1] if opt.save_dir.endswith('/') else opt.save_dir
                new_name = sd + '_archived_' + str(datetime.datetime.now())
                logger.warning(
                    f'save_dir {opt.save_dir} already exists. Renaming the existing one to {new_name}...')
                os.rename(opt.save_dir, new_name)
            os.makedirs(opt.save_dir)

        file_name = os.path.join(opt.save_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            opt_file.write(self.traverse_options(opt))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        return pickle.load(open(opt.opt_file, 'rb'))

    def parse(self):
        opt = self.gather_options()
        opt.is_train = self.is_train
        if not opt.is_train:
            # in testing, save test log into checkpoint_dir
            opt.save_dir = opt.checkpoint_dir

        self.print_options(opt)

        if opt.is_train:
            # create save_dir and write options
            self.save_options(opt)

        # TODO multi-gpu WIP
        """
    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
      id = int(str_id)
      if id >= 0:
        opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
      torch.cuda.set_device(opt.gpu_ids[0])

    assert len(opt.gpu_ids) == 0 or opt.batch_size % len(opt.gpu_ids) == 0, \
      "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
      % (opt.batch_size, len(opt.gpu_ids))
    """

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
