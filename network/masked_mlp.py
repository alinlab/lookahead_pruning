import torch
import torch.nn as nn
import torch.nn.functional as F

from network.base_model import BaseModel
from network.masked_modules import MaskedLinear, MaskedConv2d


class MaskedMLP(BaseModel):
	def __init__(self, widths, init_scheme=nn.init.xavier_normal_):
		super(MaskedMLP, self).__init__(init_scheme)
		layers = []
		for layer in range(len(widths) - 1):
			layers += [MaskedLinear(widths[layer], widths[layer + 1], init_scheme=init_scheme),
			           nn.ReLU(inplace=True)]
		self.fc = nn.Sequential(*layers)

	def forward(self, x):
		x = x.reshape(x.size()[0], -1)  # flatten
		x = self.fc(x)
		return x
