"""
BEFORE RUNNING THIS SCRIPT ENSURE YOU HAVE DONE THE FOLLOWING:
    - In pytorchexample/task.py batch norm layer ARE in the NN class and the method NN.forward()
    - In pytorchexample/client_app.py you have removed the noise modifier from the training decorator 

EXPERIMENT #4: TUNE ATTACK -- NO BATCH NORM -- NO NOISE
    - Fixed variables:
        - FL optimizer = SGD
        - Attack optimizer = LBFGS
"""
import toml 
import json
import re
import subprocess
import random
from pathlib import Path

attack_rounds = [10, 20, 30]
attack_max_iters = [5, 10, 20]
attack_history_size = [10, 50, 100]
attack_reg = [0, 0.00001, 0.0001, 0.001]

runs = 30 


# Initialize config
with open("pyproject.toml") as f:
    cfg = toml.load(f)

cfg["tool"]["flwr"]["app"]["config"]["run-inversion"] = True 

with open("pyproject.toml", "w") as f:
    toml.dump(cfg, f)

all_settings = []

seen = set()
config_path = Path.home() / ".flwr" / "config.toml"
for _ in range(runs):
    while True:
        config = (
            random.choice(attack_rounds),
            random.choice(attack_max_iters),
            random.choice(attack_history_size),
            random.choice(attack_reg)
        )

        if config not in seen:
            seen.add(config)
            break

    rounds, max_iter, history_size, reg = config

    # Configure Client Number
    with open(config_path) as f:
        cfg = toml.load(f)
    
    cfg["superlink"]["default"] = "local-simulation"
    cfg["superlink"]["local-simulation"]["options"]["num-supernodes"] = 10

    with open(config_path,"w") as f:
        toml.dump(cfg, f)

    with open("pyproject.toml") as f:
        cfg = toml.load(f)


    cfg["tool"]["flwr"]["app"]["config"]["experiment-name"] = "4-tune-attack-sgd-fedavg" 
    cfg["tool"]["flwr"]["app"]["config"]["num-clients"] = 10
    cfg["tool"]["flwr"]["app"]["config"]["optimizer"] = "sgd" 
    cfg["tool"]["flwr"]["app"]["config"]["run-inversion"] = True 
    cfg["tool"]["flwr"]["app"]["config"]["shuffle"] = False 

    cfg["tool"]["flwr"]["app"]["config"]["prox-mu"] = 0 

    cfg["tool"]["flwr"]["app"]["config"]["learning-rate"] = 0.001   # From Tuning
    cfg["tool"]["flwr"]["app"]["config"]["batch-size"] = 1          # Ideal Inversion Conditions
    cfg["tool"]["flwr"]["app"]["config"]["local-epochs"] = 1        # Ideal Inversion Conditions
    cfg["tool"]["flwr"]["app"]["config"]["window-size"] = 20        # Ideal Inversion Conditions

    cfg["tool"]["flwr"]["app"]["config"]["epsilon"] = 0 
    cfg["tool"]["flwr"]["app"]["config"]["delta"] = 0 
    cfg["tool"]["flwr"]["app"]["config"]["sensitivity"] = 0 
    cfg["tool"]["flwr"]["app"]["config"]["clipping_norm"] = 0

    cfg["tool"]["flwr"]["app"]["config"]["attack-lr"] = 1.0 
    cfg["tool"]["flwr"]["app"]["config"]["attack-rounds"] = rounds 
    cfg["tool"]["flwr"]["app"]["config"]["attack-max-iters"] = max_iter 
    cfg["tool"]["flwr"]["app"]["config"]["attack-history-size"] = history_size 
    cfg["tool"]["flwr"]["app"]["config"]["attack-reg"] = reg 
    cfg["tool"]["flwr"]["app"]["config"]["seed"] = 2 
    
    with open("pyproject.toml", "w") as f:
        toml.dump(cfg, f)

    proc = subprocess.run(["flwr", "run"])
