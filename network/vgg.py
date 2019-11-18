import torch
import torch.nn as nn
from backpack.core.layers import Flatten


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def get_vgg(cfg, use_bn):
    layers = []
    in_channels = 3
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]
            if use_bn:
                layers += [nn.Conv2d(x, x, kernel_size=1)]
                # layers += [nn.BatchNorm2d(x)]
            layers += [nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

    layers += [Flatten(), nn.Linear(512, 10)]
    return nn.Sequential(*layers)


def VGG11(use_bn=True):
    return get_vgg(cfg['VGG11'], use_bn)

def VGG13(use_bn=True):
    return get_vgg(cfg['VGG13'], use_bn)

def VGG16(use_bn=True):
    return get_vgg(cfg['VGG16'], use_bn)

def VGG19(use_bn=True):
    return get_vgg(cfg['VGG19'], use_bn)

