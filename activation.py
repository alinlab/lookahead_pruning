from train import BatchSampler
from torch.utils.data import DataLoader, Sampler


def add_list(list, list2):
    if len(list) == 0:
        return list2
    assert len(list) == len(list2)
    new_list = []
    for l, l2 in zip(list, list2):
        new_list.append(l + l2)
    return new_list


def get_activation(network, train_dataset):
    network.eval()
    batch_sampler = BatchSampler(train_dataset, 2000, 30)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4)
    act_accumulation = []
    total_num = 0
    for x, y in train_loader:
        x = x.cuda()
        temp_act = network.forward_activation(x)
        act_accumulation = add_list(act_accumulation, temp_act)
        total_num += len(x)
    act_rate = []
    for i in range(len(act_accumulation)):
        a = act_accumulation[i].float().cpu()
        a /= total_num
        act_rate.append(a)
    return act_rate