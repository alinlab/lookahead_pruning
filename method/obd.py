import torch
import torch.nn as nn
import time
from .prune import _score_based_pruning
from .laprune import LAP, _look_prev_score_multiple, _look_next_score_multiple
from torch.utils.data import DataLoader, Sampler
from train import BatchSampler
from network.mlp import get_mlp
from network.conv6 import get_conv6
from network.vgg import VGG11, VGG16, VGG19
from utils import is_masked_module, is_base_module, is_batch_norm

from backpack import extend, backpack
from backpack.extensions import DiagHessian

def OBD(network, dataset, prune_ratios, network_type, data_type):
    t1 = time.time()
    hessians = compute_hessians(network, dataset, data_type, network_type)
    t2 = time.time()
    print(t2-t1)
    def score_func(weights, layer):
        score = hessians[layer] * weights[layer] * weights[layer] / 2
        return score

    new_masks = _score_based_pruning(network.get_weights(), network.get_masks(), prune_ratios, score_func)
    t3 = time.time()
    print(t3-t2)
    return new_masks

def OBD_LAP(network, dataset, prune_ratios, network_type, data_type, bn_factors=None, mode='base', split=1):
    hessians = compute_hessians(network, dataset, data_type, network_type)

    weights = network.get_weights()
    assert len(hessians) == len(weights)

    score = []
    for (w, h) in zip(weights, hessians):
        score.append((w * w * h).sqrt())

    new_masks = LAP(score, network.get_masks(), prune_ratios, bn_factors=bn_factors, mode=mode, split=split)

    return new_masks

def compute_hessians(network, dataset, data_type, network_type):
    network.train()

    batch_size, num_iterations = get_batch_type(data_type)
    batch_sampler = BatchSampler(dataset, num_iterations, batch_size)  # train by iteration, not epoch
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

    network_seq = get_seq_network(network_type)
    network_seq = copy_network(network, network_seq)

    criterion = nn.CrossEntropyLoss()
    criterion = extend(criterion)
    network_seq = extend(network_seq).cuda()

    hessians = None
    for (x, y) in data_loader:
        x = x.cuda()
        y = y.cuda()
        if data_type == 'mnist':
            x = x.view(len(x), -1)

        out = network_seq(x)
        loss = criterion(out, y)

        with backpack(DiagHessian()):
            loss.backward()

        hessians = get_hessians(network_seq, hessians)

    return hessians

def test_compute_hessians(network, dataset, data_type, network_type, x, y):
    network.train()

    batch_size, num_iterations = get_batch_type(data_type)
    batch_sampler = BatchSampler(dataset, num_iterations, batch_size)  # train by iteration, not epoch
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

    network_seq = get_seq_network(network_type)
    network_seq = copy_network(network, network_seq)

    criterion = nn.CrossEntropyLoss()
    criterion = extend(criterion)
    network_seq = extend(network_seq).cuda()

    hessians = None
    x = x.cuda()
    y = y.cuda()
    if data_type == 'mnist':
        x = x.view(len(x), -1)

    out = network_seq(x)
    loss = criterion(out, y)

    with backpack(DiagHessian()):
        loss.backward()

    hessians = get_hessians(network_seq, hessians)

    return hessians, x, y

def get_batch_type(data_type):
    if data_type == 'mnist':
        batch_size = 60
        num_iterations = 1000
    elif data_type == 'cifar10':
        batch_size = 25
        num_iterations = 2000
    else:
        raise ValueError('Unknown dataset for OBD')

    return batch_size, num_iterations

def get_seq_network(network_type):
    if network_type == 'mlp':
        return get_mlp([784, 500, 500, 500, 500, 10]).cuda()
    elif network_type == 'conv6':
        return get_conv6()
    elif network_type == 'vgg11':
        return VGG11()
    elif network_type == 'vgg16':
        return VGG16()
    elif network_type == 'vgg19':
        return VGG19()

def copy_network(network, network_seq):
    modules = extract_param_modules(network)
    modules_seq = extract_param_modules(network_seq)

    assert len(modules) == len(modules_seq)

    for i, (m, m_seq) in enumerate(zip(modules, modules_seq)):
        state_dict = m.state_dict()
        if is_masked_module(m):
            del state_dict['mask']

        if isinstance(m, nn.BatchNorm2d):
            assert isinstance(m_seq, nn.Conv2d)
            m_seq.bias.data = m.bias.data - m.running_mean.data / m.running_var.data.sqrt() * m.weight.data
            m_seq.weight.data = (m.weight.data / m.running_var.data.sqrt()).diag()
            m_seq.weight.data = m_seq.weight.data.view(m_seq.weight.shape[0], m_seq.weight.shape[1], 1, 1)
        else:
            m_seq.load_state_dict(state_dict)

    return network_seq

def extract_param_modules(network):
    modules = []

    for m in network.modules():
        if is_base_module(m) or is_masked_module(m) or is_batch_norm(m):
            modules.append(m)

    return modules


def get_hessians(network, prev_hessians=None):
    if prev_hessians is None:
        flag = True
        prev_hessians = []
    else:
        flag = False

    cnt = 0

    prev_m = None
    for m in network.modules():
        if is_base_module(m):
            if not (isinstance(prev_m, nn.Conv2d) and isinstance(m, nn.Conv2d)):
                if flag:
                    prev_hessians.append(m.weight.diag_h.data.cpu().detach())
                else:
                    prev_hessians[cnt] += m.weight.diag_h.data.cpu().detach()
                    cnt += 1

                m.weight.diag_h[:] = 0

        prev_m = m

    return prev_hessians