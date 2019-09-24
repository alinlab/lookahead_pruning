import torch
import torch.nn as nn
import torch.nn.functional as F

from network.base_model import BaseModel
from network.masked_modules import MaskedLinear, MaskedConv2d


class MaskedLeNet(BaseModel):
	def __init__(self, init_scheme=nn.init.xavier_normal_):
		super(MaskedLeNet, self).__init__(init_scheme=init_scheme)
		self.conv1 = MaskedConv2d(3, 6, kernel_size=5)
		self.conv2 = MaskedConv2d(6, 16, kernel_size=5)
		self.fc1 = MaskedLinear(16 * 5 * 5, 120)
		self.fc2 = MaskedLinear(120, 84)
		self.fc3 = MaskedLinear(84, 10)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2)
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
