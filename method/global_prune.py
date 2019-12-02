import torch
import torch.nn as nn
from copy import deepcopy
from .laprune import _look_prev_score_multiple, _look_next_score_multiple
from .obd import compute_hessians


def MP_Global(weights, masks, prune_ratios, normalize=False):

	def score_func(weights, layer):
		score = torch.abs(weights[layer])
		return score

	new_masks = _score_based_global_pruning(weights, masks, prune_ratios, score_func, normalize)
	return new_masks


def RP_Global(weights, masks, prune_ratios):

	def score_func(weights, layer):
		score = torch.abs(torch.randn(weights[layer].size()))
		return score

	new_masks = _score_based_global_pruning(weights, masks, prune_ratios, score_func)
	return new_masks


def LAP_Global(weights, masks, prune_ratios, bn_factors=None, normalize=False):
	depth = len(weights)

	layers_prev_list = [None] + list(range(depth - 1))
	layers_next_list = list(range(1, depth)) + [None]

	def score_func(weights, layer):
		prev_score = _look_prev_score_multiple(weights, layer, layers_prev_list[layer], bn_factors)
		next_score = _look_next_score_multiple(weights, layer, layers_next_list[layer], bn_factors)
		score = (weights[layer] ** 2) * prev_score * next_score

		return score

	new_masks = _score_based_global_pruning(weights, masks, prune_ratios, score_func, normalize)
	return new_masks


def OBD_Global(network, dataset, prune_ratios, network_type, data_type, normalize=False):
	hessians = compute_hessians(network, dataset, data_type, network_type)

	def score_func(weights, layer):
		score = hessians[layer] * weights[layer] * weights[layer] / 2
		return score

	new_masks = _score_based_global_pruning(network.get_weights(), network.get_masks(), prune_ratios, score_func, normalize)
	return new_masks


def OBD_LAP_Global(network, dataset, prune_ratios, network_type, data_type, normalize=False):
	hessians = compute_hessians(network, dataset, data_type, network_type)

	weights = network.get_weights()
	assert len(hessians) == len(weights)

	new_weights = []
	for (w, h) in zip(weights, hessians):
		new_weights.append((w * w * h).sqrt())

	new_masks = LAP_Global(new_weights, network.get_masks(), prune_ratios, normalize=normalize)
	return new_masks


def _score_based_global_pruning(weights, masks, prune_ratios, score_func, normalize=False):
	""" Abstract function for score-based pruning (global version) """

	# compute scores for all layers
	scores = []
	for layer in range(len(weights)):
		score = score_func(weights, layer)
		if normalize:
			score /= score.norm()  # normalize to norm 1
		scores.append(score.view(-1))
	scores = torch.cat(scores, dim=0)

	# get linearized pruned index
	_, idx = torch.sort(scores, descending=True)
	surv_ratio = 1 - prune_ratios[0]  # only one scalar
	cutoff_index = round(surv_ratio * len(scores))
	pruned_idx = idx[cutoff_index:].tolist()

	# get new masks for each layer
	new_masks = []
	for layer in range(len(weights)):
		# print(f'Global pruning... layer: {layer}')
		new_mask = torch.ones(masks[layer].shape)
		new_mask_linearized = new_mask.view(-1)

		total = len(new_mask_linearized)
		pruned_idx_layer = [idx for idx in pruned_idx if idx < total]
		pruned_idx = [idx - total for idx in pruned_idx if idx >= total]

		new_mask_linearized[pruned_idx_layer] = 0
		new_masks.append(new_mask)

	return new_masks
