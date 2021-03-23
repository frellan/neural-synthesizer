import logging

import torch
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


logger = logging.getLogger()


class Morph(torch.nn.Module):
    def __init__(self, opt, device, *args, **kwargs):
        super(Morph, self).__init__(*args, **kwargs)

        self.opt = opt
        self.device = device
        self.n_modules = 0
        self.frozen = []
        self.pending = []

    def forward(self, input):
        frozen = torch.nn.Sequential(*self.frozen).to(self.device)
        pending = torch.nn.Sequential(*self.pending).to(self.device)
        output = frozen(input)
        output = pending(output)
        return output

    def add_pending(self, *components):
        _init_weights([*components])
        if (len(components) > 1):
            self.pending.append(torch.nn.Sequential(*components))
        else:
            self.pending.append(*components)

    def clear_pending(self):
        self.pending = []

    def freeze_pending(self):
        self.frozen.extend(self.pending)
        self.clear_pending()
        frozen_module = torch.nn.Sequential(*self.frozen)

        for p in frozen_module.parameters():
            p.requires_grad_(False)

        self.n_modules += 1

    def get_trainable_params(self):
        return torch.nn.Sequential(*self.pending).to(self.device).parameters()

    
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
            if isinstance(m, torch.nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, torch.nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
