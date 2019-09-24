import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler


class BatchSampler(Sampler):
	def __init__(self, dataset, num_iterations, batch_size):
		self.dataset = dataset
		self.num_iterations = num_iterations
		self.batch_size = batch_size

	def __iter__(self):
		for _ in range(self.num_iterations):
			indices = random.sample(range(len(self.dataset)), self.batch_size)
			yield indices

	def __len__(self):
		return self.num_iterations


def train(train_dataset, test_dataset, network, optimizer, num_iterations, batch_size, print_step=10000):
	network.train()
	batch_sampler = BatchSampler(train_dataset, num_iterations, batch_size)  # train by iteration, not epoch
	train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4)
	optimizer = optimizer(network.parameters())  # create optimizer (argument: function)

	for i, (x, y) in enumerate(train_loader):
		x = x.cuda()
		y = y.cuda()

		optimizer.zero_grad()
		out = network(x)
		loss = F.cross_entropy(out, y)
		loss.backward()
		optimizer.step()

		if (i + 1) % print_step == 0:
			test_acc, test_loss = test(network, test_dataset)
			print(f'Steps: {i + 1}/{num_iterations}\tTest loss: {test_loss:.3f}\tTest acc: {test_acc:.2f}', end='\r')
			network.train()  # train mode

	train_acc, train_loss = test(network, train_dataset)
	test_acc, test_loss = test(network, test_dataset)
	print(f'Train loss: {train_loss:.3f}\tTrain acc: {train_acc:.2f}\tTest loss: {test_loss:.3f}\tTest acc: {test_acc:.2f}')

	return train_acc, train_loss, test_acc, test_loss


def test(network, dataset, batch_size=64):
	network.eval()
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

	correct = 0
	loss = 0
	for i, (x, y) in enumerate(loader):
		x = x.cuda()
		y = y.cuda()

		with torch.no_grad():
			out = network(x)
			_, pred = out.max(1)

		correct += pred.eq(y).sum().item()
		loss += F.cross_entropy(out, y) * len(x)

	acc = correct / len(dataset) * 100.0
	loss = loss / len(dataset)

	return acc, loss
