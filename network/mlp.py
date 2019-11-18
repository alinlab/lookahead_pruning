import torch.nn as nn
import torch.nn.functional as F

from network.base_model import BaseModel
from network.masked_modules import MaskedLinear


def get_mlp(widths):
	depth = len(widths) - 1
	layers = []
	for layer in range(depth):
		layers += [nn.Linear(widths[layer], widths[layer + 1])]
		if layer < depth - 1:
			layers += [nn.ReLU(inplace=True)]

	return nn.Sequential(*layers)