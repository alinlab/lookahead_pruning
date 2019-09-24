import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class MaskedLinear(nn.Module):
	def __init__(self, in_d, out_d, init_scheme=nn.init.xavier_normal_):
		super(MaskedLinear, self).__init__()
		self.weight = nn.Parameter(init_scheme(torch.Tensor(out_d, in_d)))
		self.bias = nn.Parameter(torch.zeros(out_d))
		self.register_buffer('mask', torch.ones(out_d, in_d))

	def forward(self, x):
		return F.linear(x, self.weight * self.mask, self.bias)


class MaskedConv2d(nn.modules.conv._ConvNd):
	def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', init_scheme=nn.init.xavier_normal_):
		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(MaskedConv2d, self).__init__(in_c, out_c, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode)
		self.weight.data = init_scheme(self.weight.data)
		if bias == True:
			nn.init.zeros_(self.bias.data)
		self.register_buffer('mask', torch.ones(self.weight.data.size()))

	def forward(self, x):
		if self.padding_mode == 'circular':
			expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2, (self.padding[0] + 1) // 2, self.padding[0] // 2)
			return F.conv2d(F.pad(x, expanded_padding, mode='circular'), self.weight * self.mask, self.bias, self.stride, _pair(0), self.dilation, self.groups)
		return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)

