from torch import nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    def __init__(self, cnin, cnout, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cnout)
        self.conv1 = nn.Conv2d(cnin, cnout, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cnout)
        self.conv2 = nn.Conv2d(cnout, cnout, kernel_size=3, stride=1, padding=1, bias=False)

        self.downsample = downsample
        
        if stride != 1 or cnin != cnout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cnin, cnout, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(cnout))
            
    def forward(self, x):
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)

        out += shortcut
        return F.relu(out)


class ResnetOutput(nn.Module):
    def __init__(self, input_size, output_size, *args, **kwargs):
        super(ResnetOutput, self).__init__(*args, **kwargs)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = F.adaptive_avg_pool2d(input, 1)
        output = output.view(output.size(0), -1)
        return self.linear(output)
