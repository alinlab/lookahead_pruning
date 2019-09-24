# Lookahead Pruning (LAP)

## Run experiments

Run on MLP network (MNIST)
```
python main.py --dataset mnist --network mlp --method mp
python main.py --dataset mnist --network mlp --method lap
```

Run on VGG11 network (CIFAR-10)
```
python main.py --dataset cifar10 --network vgg11 --method mp
python main.py --dataset cifar10 --network vgg11 --method lap_bn
```

Results are saved in
```
./checkpoint/{dataset}_{network}_{seed}/{method}/logs.txt
```
