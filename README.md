# Lookahead Pruning (LAP)


Compare magnitude pruning (MP) and lookahead pruning (LAP)
```
python main.py --dataset mnist --network mlp --method mp
python main.py --dataset mnist --network mlp --method lap
```

Results are saved in
```
./checkpoint/{dataset}_{network}_{seed}/{method}/logs.txt
```
