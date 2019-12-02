import torch
import torch.nn as nn
import torch.nn.functional as F

from network.base_model import BaseModel
from network.masked_modules import MaskedLinear, MaskedConv2d


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class MaskedVGG_64(BaseModel):
    def __init__(self, vgg_name, use_bn=True):
        super(MaskedVGG_64, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], use_bn)
        self.classifier = MaskedLinear(512, 200)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, use_bn):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if i == 0:
                    layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=1, stride=2)]
                else:
                    layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=1)]
                if use_bn:
                    layers += [nn.BatchNorm2d(x)]
                layers += [nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def MaskedVGG11_64(use_bn=True):
    return MaskedVGG_64('VGG11', use_bn)

def MaskedVGG13_64(use_bn=True):
    return MaskedVGG_64('VGG13', use_bn)

def MaskedVGG16_64(use_bn=True):
    return MaskedVGG_64('VGG16', use_bn)

def MaskedVGG19_64(use_bn=True):
    return MaskedVGG_64('VGG19', use_bn)

