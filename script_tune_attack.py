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

for _ in range(runs):
    lr = random.choice(attack_lr)
    rounds= random.choice(attack_rounds)
    reg = random.choice(attack_reg)
    for seed in range(num_seeds):
        with open("pyproject.toml") as f:
            cfg = toml.load(f)

        cfg["tool"]["flwr"]["app"]["config"]["run-inversion"] = True 
        cfg["tool"]["flwr"]["app"]["config"]["attack-lr"] = lr
        cfg["tool"]["flwr"]["app"]["config"]["attack-rounds"] = rounds 
        cfg["tool"]["flwr"]["app"]["config"]["attack-reg"] = reg 
        cfg["tool"]["flwr"]["app"]["config"]["seed"] = seed + 1
        
        with open("pyproject.toml", "w") as f:
            toml.dump(cfg, f)

        proc = subprocess.run(["flwr", "run"])
