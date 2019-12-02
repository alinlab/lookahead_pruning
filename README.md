# Lookahead Pruning (LAP)

PyTorch implementation of [Lookahead: A Far-sighted Alternative of Magnitude-based Pruning](https://openreview.net/forum?id=ryl3ygHYDB).

The code was written by [Sejun Park](https://github.com/sejunpark-repository), [Jaeho Lee](https://github.com/jaeho-lee), and [Sangwoo Mo](https://github.com/sangwoomo).

## Installation

Download [BackPack](https://toiaydcdyywlhzvlob.github.io/backpack/) package to run OBD experiments.
If you only want MP/LAP experiments, simply comment the OBD parts.


## Run experiments

Run MNIST experiments (MLP)
```
python main.py --dataset mnist --network mlp --method mp
python main.py --dataset mnist --network mlp --method lap
```

Run CIFAR-10 experiments (VGG19)
```
python main.py --dataset cifar10 --network vgg19 --method mp
python main.py --dataset cifar10 --network vgg19 --method lap_bn
```

Run Tiny-ImageNet experiments (ResNet50)
```
python main.py --dataset tiny-imagenet --network resnet50_64 --method mp
python main.py --dataset tiny-imagenet --network resnet50_64 --method lap_bn
```

Run data-dependent pruning experiments
```
python main.py --dataset mnist --network mlp --method obd
python main.py --dataset mnist --network mlp --method lap_act
```

Run global pruning experiments
```
python main.py --dataset mnist --network mlp_global --method mp_global_normalize
python main.py --dataset mnist --network mlp_global --method lap_global_normalize
```

Results are saved in
```
./checkpoint/{dataset}_{network}_{pruning_type}_{seed}/{method}/logs.txt
```
