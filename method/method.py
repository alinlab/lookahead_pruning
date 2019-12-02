from functools import partial
from .prune import MP, RP
from .laprune import LAP
from .obd import OBD
from .global_prune import MP_Global, LAP_Global, OBD_Global


def get_method(method):
	if method == 'mp':
		method = MP

	elif method == 'rp':
		method = RP

	elif method in ['lap', 'lap_bn', 'lap_act']:
		method = partial(LAP, mode='base')

	elif method in ['lap_forward', 'lap_forward_bn']:
		method = partial(LAP, mode='forward')

	elif method == 'obd':
		method = OBD

	elif method == 'mp_global_normalize':
		method = partial(MP_Global, normalize=True)

	elif method in ['lap_global_normalize', 'lap_act_global_normalize']:
		method = partial(LAP_Global, normalize=True)

	elif method == 'obd_global':
		method = OBD_Global

	elif method == 'obd_global_normalize':
		method = partial(OBD_Global, normalize=True)

	else:
		raise ValueError('Unknown method')

	return method

