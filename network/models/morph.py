import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


logger = logging.getLogger()


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, device):
        super(Block, self).__init__()

        self.device = device

        padding = kernel_size // 2
        groups = in_channels if kernel_size == 1 else 1

        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=groups,
            bias=False).to(self.device)
        self.bn = nn.BatchNorm2d(out_channels).to(self.device)
        self.relu = nn.ReLU().to(self.device)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Cell(nn.Module):
    def __init__(self, in_channels, use_residual, blocks, device):
        super(Cell, self).__init__()

        self.device = device

        self.seq1 = nn.Sequential(*blocks[0:2]).to(self.device)
        self.seq2 = nn.Sequential(*blocks[2:4]).to(self.device)
        self.seq3 = nn.Sequential(*blocks[4:6]).to(self.device)

        stack_channels = blocks[1].out_channels + blocks[3].out_channels + blocks[5].out_channels

        self.use_residual = use_residual

        self.shrink = nn.Conv2d(
            in_channels=stack_channels,
            out_channels=in_channels,
            kernel_size=1,
            groups=1, # supposed to be in_channels, stack_channels needs to be a multiple of in_channels
            bias=False).to(self.device)
        self.bn = nn.BatchNorm2d(in_channels).to(self.device)
        self.relu = nn.ReLU().to(self.device)

    def forward(self, x):
        seq1 = self.seq1(x)
        seq2 = self.seq2(x)
        seq3 = self.seq3(x)

        stack = torch.cat([seq1, seq2, seq3], dim=1)
        output = self.relu(self.bn(self.shrink(stack)))

        return output + x if self.use_residual else output


class OutputCell(nn.Module):
    def __init__(self, input_size, output_size, *args, **kwargs):
        super(OutputCell, self).__init__(*args, **kwargs)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = F.adaptive_avg_pool2d(input, 1)
        output = output.view(output.size(0), -1)
        return self.linear(output)


class Morph(nn.Module):
    def __init__(self, device, *args, **kwargs):
        super(Morph, self).__init__(*args, **kwargs)

        self.device = device
        self.n_modules = 0
        self.frozen = []
        self.pending = []

    def forward(self, input):
        pending = nn.Sequential(*self.pending).to(self.device)
        if len(self.frozen) > 0:
            frozen = nn.Sequential(*self.frozen).to(self.device)
            output = frozen(input)
            output = pending(output)
        else:
            output = pending(input)
        return output

    def add_pending(self, *components):
        _init_weights([*components])
        if (len(components) > 1):
            self.pending.extend([*components])
        else:
            self.pending.append(*components)

    def clear_pending(self):
        self.pending = []

    def freeze_pending(self):
        self.frozen.extend(self.pending)
        self.clear_pending()
        frozen_module = nn.Sequential(*self.frozen)

        for p in frozen_module.parameters():
            p.requires_grad_(False)

        self.n_modules += 1

    def unfreeze_network(self):
        frozen_module = nn.Sequential(*self.frozen)
        for p in frozen_module.parameters():
            p.requires_grad_(True)

    def get_all_trainable_params(self):
        return nn.Sequential(*self.frozen, *self.pending).to(self.device).parameters()

    def get_trainable_params(self):
        return nn.Sequential(*self.pending).to(self.device).parameters()

    
@torch.no_grad()
def _init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.
    Reference: https://github.com/xinntao/BasicSR

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
