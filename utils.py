import torch
import torch.nn as nn
from network.masked_modules import MaskedLinear, MaskedConv2d


def is_base_module(m):
	if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
		return True
	else:
		return False


def is_masked_module(m):
	if isinstance(m, MaskedLinear) or isinstance(m, MaskedConv2d):
		return True
	else:
		return False


def is_batch_norm(m):
	if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
		return True
	else:
		return False


def get_sparsity(model):
	nonzero = 0
	total = 0
	for name, m in model.named_modules():
		if is_masked_module(m):
			p = m.mask
			nz_count = (p != 0).type(torch.float).sum()
			total_count = p.numel()
			nonzero += nz_count
			total += total_count

			print(f'{name:20} | nonzeros = {nz_count:7}/{total_count} ({100 * nz_count / total_count:6.2f}%) | total_pruned = {total_count - nz_count:7} | shape= {list(p.data.shape)}')
	print(f'surv: {nonzero}, pruned: {total - nonzero}, total: {total}, Comp. rate: {total / nonzero:10.2f}x ({100 * (total - nonzero) / total:6.2f}% pruned)')

	return nonzero / total

