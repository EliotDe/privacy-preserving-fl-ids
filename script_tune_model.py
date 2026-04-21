import toml 
import json
import re
import subprocess
import random

## Initial search space

#lrs = [0.1, 0.03, 0.01, 0.001, 0.0001]
#local_epochs = [1, 3, 5, 10]
#batch_sizes = [16, 32, 64]
#window_sizes = [20, 30, 40, 50]

## Next search space
lrs = [0.1,0.03,0.01,0.0001]
local_epochs = [1,3,5]
batch_sizes = [32, 64]
window_sizes = [30,40,50]

runs = 29 

seen = set()
for _ in range(runs):
    while True:
        config = (
            random.choice(lrs),
            random.choice(local_epochs),
            random.choice(batch_sizes),
            random.choice(window_sizes)
        )

        if config not in seen:
            seen.add(config)
            break
    lr, num_local_epochs, batch_size, window_size = config

    with open("pyproject.toml") as f:
        cfg = toml.load(f)

    cfg["tool"]["flwr"]["app"]["config"]["run-inversion"] = False 
    cfg["tool"]["flwr"]["app"]["config"]["shuffle"] = True 
    cfg["tool"]["flwr"]["app"]["config"]["prox-mu"] = 0 
    cfg["tool"]["flwr"]["app"]["config"]["learning-rate"] = lr 
    cfg["tool"]["flwr"]["app"]["config"]["batch-size"] = batch_size
    cfg["tool"]["flwr"]["app"]["config"]["local-epochs"] = num_local_epochs 
    cfg["tool"]["flwr"]["app"]["config"]["window-size"] = window_size 
    
    with open("pyproject.toml", "w") as f:
        toml.dump(cfg, f)

    proc = subprocess.run(["flwr", "run"])
