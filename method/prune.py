import torch
import torch.nn as nn


def MP(weights, masks, prune_ratios):
	""" Magnitude pruning """

	def score_func(weights, layer):
		score = torch.abs(weights[layer])
		return score

	new_masks = _score_based_pruning(weights, masks, prune_ratios, score_func)
	return new_masks


def RP(weights, masks, prune_ratios):
	""" Random pruning """

	def score_func(weights, layer):
		score = torch.abs(torch.randn(weights[layer].size()))
		return score

	new_masks = _score_based_pruning(weights, masks, prune_ratios, score_func)
	return new_masks


def _score_based_pruning(weights, masks, prune_ratios, score_func, mode='base', split=1):
	""" Abstract function for score-based pruning """
	if mode == 'base':
		layers = range(len(prune_ratios))
	elif mode == 'forward':
		layers = range(len(prune_ratios))
	elif mode == 'backward':
		layers = reversed(range(len(prune_ratios)))
	else:
		raise ValueError('Unknown pruning method')

	new_masks = []
	for s in range(split):
		new_masks = []
		for layer in layers:
			score = score_func(weights, layer)  # score for current layer
			prune_rate = prune_ratios[layer] / split / (1 - prune_ratios[layer] * s / split)
			new_mask = _score_based_mask(score, masks[layer], prune_rate)
			new_masks.append(new_mask)

			if mode == 'forward' or mode == 'backward':  # update weights/masks (current layer)
				weights[layer] *= new_mask
				masks[layer] *= new_mask

		if mode == 'base':  # update weights/masks (all layers)
			for layer in layers:
				weights[layer] *= new_masks[layer]
				masks[layer] *= new_masks[layer]

	if mode == 'backward':
		new_masks.reverse()

	return new_masks


def _score_based_mask(score, mask, prune_ratio):
	""" Get mask for current layer """
	assert (prune_ratio >= 0) and (prune_ratio <= 1)
	score[mask <= 0] = float('-inf')  # necessary?

	surv_ratio = 1 - prune_ratio
	num_surv_weights = torch.sum(mask).item()
	cutoff_index = round(surv_ratio * num_surv_weights)

	_, idx = torch.sort(score.view(-1), descending=True)
	new_mask = torch.ones(mask.shape) * mask
	new_mask_linearized = new_mask.view(-1)
	new_mask_linearized[idx[cutoff_index:]] = 0

	return new_mask

