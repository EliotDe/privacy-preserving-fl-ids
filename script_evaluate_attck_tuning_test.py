import toml 
import json
import re
import subprocess
import random


attack_lr = [0.1, 0.03, 0.01, 0.001, 0.0001]
attack_rounds = [300, 1000]
attack_reg = [0, 0.00001, 0.0001, 0.001]
num_seeds = 5 

runs = 4 


# Initialize config
with open("pyproject.toml") as f:
    cfg = toml.load(f)

cfg["tool"]["flwr"]["app"]["config"]["run-inversion"] = True 

with open("pyproject.toml", "w") as f:
    toml.dump(cfg, f)

all_settings = []

seen = set()
for _ in range(runs):
    for seed in range(num_seeds):
        while True:
            config = (
                random.choice(attack_lr),
                random.choice(attack_rounds),
                random.choice(attack_reg),
                seed
            )

            if config not in seen:
                seen.add(config)
                break

        lr, rounds, reg, seed = config

        with open("pyproject.toml") as f:
            cfg = toml.load(f)


        cfg["tool"]["flwr"]["app"]["config"]["experiment-name"] = "tune-attack-test" 
        cfg["tool"]["flwr"]["app"]["config"]["run-inversion"] = True 
        cfg["tool"]["flwr"]["app"]["config"]["shuffle"] = False 
        cfg["tool"]["flwr"]["app"]["config"]["prox-mu"] = 0 
        cfg["tool"]["flwr"]["app"]["config"]["learning-rate"] = 0.001   # From Tuning
        cfg["tool"]["flwr"]["app"]["config"]["batch-size"] = 32         # Ideal Inversion Conditions
        cfg["tool"]["flwr"]["app"]["config"]["local-epochs"] = 3        # Ideal Inversion Conditions
        cfg["tool"]["flwr"]["app"]["config"]["window-size"] = 20        # Ideal Inversion Conditions
        cfg["tool"]["flwr"]["app"]["config"]["attack-lr"] = lr
        cfg["tool"]["flwr"]["app"]["config"]["attack-rounds"] = rounds 
        cfg["tool"]["flwr"]["app"]["config"]["attack-reg"] = reg 
        cfg["tool"]["flwr"]["app"]["config"]["seed"] = seed + 1
        
        with open("pyproject.toml", "w") as f:
            toml.dump(cfg, f)

        proc = subprocess.run(["flwr", "run"])
