import toml 
import json
import re
import subprocess
import random

rounds = [10,25,50]
max_iters = [5, 10, 20]
history_sizes = [10, 50, 100]

runs = 20 


# Initialize config
with open("pyproject.toml") as f:
    cfg = toml.load(f)

cfg["tool"]["flwr"]["app"]["config"]["run-inversion"] = True 

with open("pyproject.toml", "w") as f:
    toml.dump(cfg, f)

all_settings = []

seen = set()
for _ in range(runs):
    while True:
        config = (
            random.choice(max_iters),
            random.choice(history_sizes),
            random.choice(rounds)
        )

        if config not in seen:
            seen.add(config)
            break

    max_iter, history_size, num_rounds = config

    with open("pyproject.toml") as f:
        cfg = toml.load(f)


        cfg["tool"]["flwr"]["app"]["config"]["experiment-name"] = "tune-attack-lbfgs" 
        cfg["tool"]["flwr"]["app"]["config"]["run-inversion"] = True 
        cfg["tool"]["flwr"]["app"]["config"]["shuffle"] = False 
        cfg["tool"]["flwr"]["app"]["config"]["prox-mu"] = 0 
        cfg["tool"]["flwr"]["app"]["config"]["learning-rate"] = 0.001   # From Tuning
        cfg["tool"]["flwr"]["app"]["config"]["batch-size"] = 1          # Ideal Inversion Conditions
        cfg["tool"]["flwr"]["app"]["config"]["local-epochs"] = 1        # Ideal Inversion Conditions
        cfg["tool"]["flwr"]["app"]["config"]["window-size"] = 20        # Ideal Inversion Conditions
        cfg["tool"]["flwr"]["app"]["config"]["attack-lr"] = 1.0 
        cfg["tool"]["flwr"]["app"]["config"]["attack-reg"] = 0 
        cfg["tool"]["flwr"]["app"]["config"]["attack-max-iter"] = max_iter 
        cfg["tool"]["flwr"]["app"]["config"]["attack-history-size"] = history_size 
        cfg["tool"]["flwr"]["app"]["config"]["attack-rounds"] = num_rounds 
        
        with open("pyproject.toml", "w") as f:
            toml.dump(cfg, f)

        proc = subprocess.run(["flwr", "run"])
