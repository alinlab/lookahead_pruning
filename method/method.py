from functools import partial
from .prune import MP, RP
from .laprune import LAP


def get_method(method):
	if method == 'mp':
		method = MP

	elif method == 'rp':
		method = RP

	elif method in ['lap', 'lap_bn']:
		method = partial(LAP, mode='base')

	elif method in ['lap_forward', 'lap_forward_bn']:
		method = partial(LAP, mode='forward')

	else:
		raise ValueError('Unknown method')

	return method

