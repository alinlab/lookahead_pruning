import argparse
import os
import random
import numpy as np
from copy import deepcopy
from functools import partial

import torch
import torch.optim as optim

from dataset import get_dataset
from network import *
from train import train, test
from method import get_method
from utils import get_sparsity


def main():
	# get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=718, help='random seed')
	parser.add_argument('--dataset', type=str, default='mnist', help='dataset (mnist|fmnist|cifar10)')
	parser.add_argument('--network', type=str, default='mlp', help='network (mlp|lenet|conv6|vgg19|resnet18)')
	parser.add_argument('--method', type=str, default='mp', help='method (mp|rp|labp)')
	parser.add_argument('--pruning_iteration_start', type=int, default=1, help='start iteration for pruning')
	parser.add_argument('--pruning_iteration_end', type=int, default=30, help='end iteration for pruning')
	args = parser.parse_args()

	# fix randomness
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	# define dataset and network
	train_dataset, test_dataset = get_dataset(args.dataset)

	if args.network == 'mlp':
		network = MaskedMLP([784, 500, 500, 500, 500, 10]).cuda()
		prune_ratios = [.5, .5, .5, .5, .25]
		optimizer = partial(optim.Adam, lr=0.0012)
		pretrain_iteration = 50000
		finetune_iteration = 50000
		batch_size = 60

	elif args.network == 'lenet':
		network = MaskedLeNet().cuda()
		prune_ratios = [.2, .2, .3, .3, .15]
		optimizer = partial(optim.Adam, lr=0.0012)
		pretrain_iteration = 50000
		finetune_iteration = 50000
		batch_size = 60

	elif args.network == 'conv6':
		network = MaskedConv6().cuda()
		prune_ratios = [.15, .15, .15, .15, .15, .15, .20, .20, .10]
		optimizer = partial(optim.Adam, lr=0.0003)
		pretrain_iteration = 30000
		finetune_iteration = 20000
		batch_size = 60

	elif args.network == 'vgg11':
		network = MaskedVGG11(use_bn=True).cuda()
		prune_ratios = [.15] * 8 + [.10]
		optimizer = partial(optim.Adam, lr=0.0003)
		pretrain_iteration = 35000
		finetune_iteration = 25000
		batch_size = 60

	elif args.network == 'vgg16':
		network = MaskedVGG16(use_bn=True).cuda()
		prune_ratios = [.15] * 13 + [.10]
		optimizer = partial(optim.Adam, lr=0.0003)
		pretrain_iteration = 50000
		finetune_iteration = 35000
		batch_size = 60

	elif args.network == 'vgg19':
		network = MaskedVGG19(use_bn=True).cuda()
		prune_ratios = [.15] * 16 + [.10]
		optimizer = partial(optim.Adam, lr=0.0003)
		pretrain_iteration = 60000
		finetune_iteration = 40000
		batch_size = 60

	elif args.network == 'resnet18':
		network = MaskedResNet18().cuda()
		prune_ratios = [0] + [.15] * 16 + [.10]
		optimizer = partial(optim.Adam, lr=0.0003)
		pretrain_iteration = 35000
		finetune_iteration = 25000
		batch_size = 60

	else:
		raise ValueError('Unknown network')

	# load pre-trained network
	base_path = f'./checkpoint/{args.dataset}_{args.network}_{args.seed}'
	if not os.path.exists(base_path):
		os.makedirs(base_path)

	if not os.path.exists(os.path.join(base_path, 'base_model.pth')):
		print('Pre-train network')
		# pre-train network if not exits
		pre_train_acc, pre_train_loss = test(network, train_dataset)
		pre_test_acc, pre_test_loss = test(network, test_dataset)
		train_acc, train_loss, test_acc, test_loss = train(train_dataset, test_dataset, network, optimizer, pretrain_iteration, batch_size)

		# save network and logs
		torch.save(network.state_dict(), os.path.join(base_path, 'base_model.pth'))
		with open(os.path.join(base_path, 'logs.txt'), 'w') as f:
			f.write(f'{pre_train_loss:.3f}\t{pre_test_loss:.3f}\t{train_loss:.3f}\t{test_loss:.3f}\t'
			        f'{pre_train_acc:.2f}\t{pre_test_acc:.2f}\t{train_acc:.2f}\t{test_acc:.2f}\n')
	else:
		print('Load pre-trained network')
		state_dict = torch.load(os.path.join(base_path, 'base_model.pth'))
		network.load_state_dict(state_dict)

	# prune and fine-tune network
	exp_path = os.path.join(base_path, args.method)
	if not os.path.exists(exp_path):
		os.makedirs(exp_path)

	original_network = network  # keep the original network
	original_prune_ratio = prune_ratios  # keep the original prune ratio
	pruning_method = get_method(args.method)
	for it in range(args.pruning_iteration_start, args.pruning_iteration_end + 1):
		print(f'Pruning iter. {it}')

		network = deepcopy(original_network).cuda()  # copy original network
		prune_ratios = []  # prune ratio for current iteration
		for idx in range(len(original_prune_ratio)):
			prune_ratios.append(1.0 - ((1.0 - original_prune_ratio[idx]) ** it))

		if 'bn' in args.method:
			masks = pruning_method(network.get_weights(), network.get_masks(), prune_ratios, network.get_bn_weights())
		else:
			masks = pruning_method(network.get_weights(), network.get_masks(), prune_ratios)
		network.set_masks(masks)

		sparsity = get_sparsity(network)
		pre_train_acc, pre_train_loss = test(network, train_dataset)
		pre_test_acc, pre_test_loss = test(network, test_dataset)
		train_acc, train_loss, test_acc, test_loss = train(train_dataset, test_dataset, network, optimizer, finetune_iteration, batch_size)

		# save network and logs
		with open(os.path.join(exp_path, 'logs.txt'), 'a') as f:
			f.write(f'{it}\t{sparsity:.6f}\t'
			        f'{pre_train_loss:.3f}\t{pre_test_loss:.3f}\t{train_loss:.3f}\t{test_loss:.3f}\t'
			        f'{pre_train_acc:.2f}\t{pre_test_acc:.2f}\t{train_acc:.2f}\t{test_acc:.2f}\n')


if __name__ == '__main__':
	main()

