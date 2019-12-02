from functools import partial
from network import *

import torch.optim as optim


def get_hyperparameters(network_type):
	if network_type == 'mlp':
		network = MaskedMLP([784, 500, 500, 500, 500, 10]).cuda()
		prune_ratios = [.5, .5, .5, .5, .25]
		optimizer = partial(optim.Adam, lr=0.0012)
		pretrain_iteration = 50000
		finetune_iteration = 50000
		batch_size = 60

	elif network_type == 'lenet':
		network = MaskedLeNet().cuda()
		prune_ratios = [.2, .2, .3, .3, .15]
		optimizer = partial(optim.Adam, lr=0.0012)
		pretrain_iteration = 50000
		finetune_iteration = 50000
		batch_size = 60

	elif network_type == 'conv6':
		network = MaskedConv6().cuda()
		prune_ratios = [.15, .15, .15, .15, .15, .15, .20, .20, .10]
		optimizer = partial(optim.Adam, lr=0.0003)
		pretrain_iteration = 30000
		finetune_iteration = 20000
		batch_size = 60

	elif network_type == 'vgg11':
		network = MaskedVGG11(use_bn=True).cuda()
		prune_ratios = [.15] * 8 + [.10]
		optimizer = partial(optim.Adam, lr=0.0003)
		pretrain_iteration = 35000
		finetune_iteration = 25000
		batch_size = 60

	elif network_type == 'vgg16':
		network = MaskedVGG16(use_bn=True).cuda()
		prune_ratios = [.15] * 13 + [.10]
		optimizer = partial(optim.Adam, lr=0.0003)
		pretrain_iteration = 50000
		finetune_iteration = 35000
		batch_size = 60

	elif network_type == 'vgg19':
		network = MaskedVGG19(use_bn=True).cuda()
		prune_ratios = [.15] * 16 + [.10]
		optimizer = partial(optim.Adam, lr=0.0003)
		pretrain_iteration = 60000
		finetune_iteration = 40000
		batch_size = 60

	elif network_type == 'resnet18':
		network = MaskedResNet18().cuda()
		prune_ratios = [0] + [.15] * 16 + [.10]
		optimizer = partial(optim.Adam, lr=0.0003)
		pretrain_iteration = 35000
		finetune_iteration = 25000
		batch_size = 60

	elif network_type == 'vgg19_64':
		network = MaskedVGG19_64(use_bn=True).cuda()
		prune_ratios = [.15] * 16 + [.10]
		optimizer = partial(optim.Adam, lr=0.0003)
		pretrain_iteration = 60000
		finetune_iteration = 40000
		batch_size = 60

	elif network_type == 'resnet50_64':
		network = MaskedResNet50_64().cuda()
		prune_ratios = [0] + [.15] * 48 + [.10]
		optimizer = partial(optim.Adam, lr=0.0003)
		pretrain_iteration = 60000
		finetune_iteration = 40000
		batch_size = 60

	elif network_type == 'wrn-16-8_64':
		network = MaskedWideResNet_64(16, 8).cuda()
		prune_ratios = [0] + [.15] * 12 + [.10]
		optimizer = partial(optim.Adam, lr=0.0003)
		pretrain_iteration = 60000
		finetune_iteration = 40000
		batch_size = 60

	elif network_type == 'mlp_global':
		network = MaskedMLP([784, 500, 500, 500, 500, 10]).cuda()
		prune_ratios = [0.9364, 0.9679, 0.9837, 0.9916, 0.9957, 0.9977, 0.9988]
		optimizer = partial(optim.Adam, lr=0.0012)
		pretrain_iteration = 50000
		finetune_iteration = 50000
		batch_size = 60

	elif network_type == 'conv6_global':
		network = MaskedConv6().cuda()
		prune_ratios = [0.8938, 0.9114, 0.9261, 0.9382, 0.9483, 0.9568, 0.9638]
		optimizer = partial(optim.Adam, lr=0.0003)
		pretrain_iteration = 30000
		finetune_iteration = 20000
		batch_size = 60

	else:
		raise ValueError('Unknown network')

	return network, prune_ratios, optimizer, pretrain_iteration, finetune_iteration, batch_size