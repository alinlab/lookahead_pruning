import torch.nn as nn
import torch.nn.functional as F

from network.base_model import BaseModel
from network.masked_modules import MaskedLinear

class MaskedMLP(BaseModel):
	def __init__(self, widths, init_scheme=nn.init.xavier_normal_):
		super(MaskedMLP, self).__init__(init_scheme)
		depth = len(widths) - 1
		layers = []
		for layer in range(depth):
			layers += [MaskedLinear(widths[layer], widths[layer + 1], init_scheme=init_scheme)]
			if layer < depth - 1:
				layers += [nn.ReLU(inplace=True)]
		self.fc = nn.Sequential(*layers)

		self.layers = layers

	def forward(self, x):
		x = x.view(len(x), -1)
		x = self.fc(x)
		return x

	def forward_activation(self, x):
		activation = []

		x = x.view(len(x), -1)
		for layer in self.layers:
			x = layer(x)

			if isinstance(layer, nn.ReLU):
				activation.append((x.data > 0).sum(dim=0))

		return activation
