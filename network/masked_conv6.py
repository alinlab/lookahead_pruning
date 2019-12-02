import torch
import torch.nn as nn
import torch.nn.functional as F

from network.base_model import BaseModel
from network.masked_modules import MaskedLinear, MaskedConv2d


class MaskedConv6(BaseModel):
	def __init__(self, init_scheme=nn.init.xavier_normal_):
		super(MaskedConv6, self).__init__(init_scheme=init_scheme)
		self.init_scheme = init_scheme

		self.conv1 = MaskedConv2d(3, 64, 3, padding=1, init_scheme=init_scheme)
		self.conv2 = MaskedConv2d(64, 64, 3, padding=1, init_scheme=init_scheme)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv3 = MaskedConv2d(64, 128, 3, padding=1, init_scheme=init_scheme)
		self.conv4 = MaskedConv2d(128, 128, 3, padding=1, init_scheme=init_scheme)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv5 = MaskedConv2d(128, 256, 3, padding=1, init_scheme=init_scheme)
		self.conv6 = MaskedConv2d(256, 256, 3, padding=1, init_scheme=init_scheme)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.fc1 = MaskedLinear(4 * 4 * 256, 256, init_scheme=init_scheme)
		self.fc2 = MaskedLinear(256, 256, init_scheme=init_scheme)
		self.fc3 = MaskedLinear(256, 10, init_scheme=init_scheme)

	def forward(self, x):
		x = F.relu(self.conv2(F.relu(self.conv1(x))))
		x = self.pool1(x)
		x = F.relu(self.conv4(F.relu(self.conv3(x))))
		x = self.pool2(x)
		x = F.relu(self.conv6(F.relu(self.conv5(x))))
		x = self.pool3(x)
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def forward_activation(self, x):
		act = []
		x = F.relu(self.conv1(x))
		act.append((x.data > 0).sum(dim=0))
		x = F.relu(self.conv2(x))
		act.append((x.data > 0).sum(dim=0))
		x = self.pool1(x)

		x = F.relu(self.conv3(x))
		act.append((x.data > 0).sum(dim=0))
		x = F.relu(self.conv4(x))
		act.append((x.data > 0).sum(dim=0))
		x = self.pool2(x)

		x = F.relu(self.conv5(x))
		act.append((x.data > 0).sum(dim=0))
		x = F.relu(self.conv6(x))
		act.append((x.data > 0).sum(dim=0))
		x = self.pool3(x)

		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		act.append((x.data > 0).sum(dim=0))
		x = F.relu(self.fc2(x))
		act.append((x.data > 0).sum(dim=0))
		x = self.fc3(x)
		return act
