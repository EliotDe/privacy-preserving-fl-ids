"""
BEFORE RUNNING THIS SCRIPT ENSURE YOU HAVE DONE THE FOLLOWING:
    - In pytorchexample/task.py the model should be using batch norm layers
    - In pytorchexample/client_app.py the train method should NOT be decorated with a noise modifier

TUNING EXPERIMENT #1: TUNE MODEL WITH ADAM, FEDAVG
TUNING EXPERIMENT #2: TUNE MODEL WITH SGD, FEDAVG
"""
import toml 
import json
import re
import subprocess
import random
from pathlib import Path

## Initial search space

lrs = [0.1, 0.03, 0.01, 0.001, 0.0001]
local_epochs = [1, 3, 5, 10]
batch_sizes = [16, 32, 64]
window_sizes = [20, 30, 40, 50]


config_path = Path.home() / ".flwr" / "config.toml"
def tune_model(optimizer: str):
    runs = 30 
    if optimizer == "adam":
        experiment_num="1"
    else:
        experiment_num="2"
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

        # Configure Client Number
        with open(config_path) as f:
            cfg = toml.load(f)
        
        cfg["superlink"]["default"] = "local-simulation"
        cfg["superlink"]["local-simulation"]["options"]["num-supernodes"] = 10 

        with open(config_path,"w") as f:
            toml.dump(cfg, f)



        with open("pyproject.toml") as f:
            cfg = toml.load(f)

        cfg["tool"]["flwr"]["app"]["config"]["experiment-name"] = f"{experiment_num}-tune-model-{optimizer}" 
        cfg["tool"]["flwr"]["app"]["config"]["num-clients"] = 10 
        cfg["tool"]["flwr"]["app"]["config"]["shuffle"] = True 
        cfg["tool"]["flwr"]["app"]["config"]["num-server-rounds"] = 35 
        # No Attack
        cfg["tool"]["flwr"]["app"]["config"]["run-inversion"] = False 
        cfg["tool"]["flwr"]["app"]["config"]["attack-lr"] = 0 
        cfg["tool"]["flwr"]["app"]["config"]["attack-rounds"] = 0 
        cfg["tool"]["flwr"]["app"]["config"]["attack-max-iter"] = 0 
        cfg["tool"]["flwr"]["app"]["config"]["attack-history-size"] = 0 
        cfg["tool"]["flwr"]["app"]["config"]["attack-reg"] = 0 
        # Run FedAvg
        cfg["tool"]["flwr"]["app"]["config"]["prox-mu"] = 0 
        # No Noise
        cfg["tool"]["flwr"]["app"]["config"]["epsilon"] = 0 
        cfg["tool"]["flwr"]["app"]["config"]["delta"] = 0 
        cfg["tool"]["flwr"]["app"]["config"]["sensitivity"] = 0 
        cfg["tool"]["flwr"]["app"]["config"]["clipping_norm"] = 0 
        # Set variables
        cfg["tool"]["flwr"]["app"]["config"]["optimizer"] = optimizer 
        cfg["tool"]["flwr"]["app"]["config"]["learning-rate"] = lr 
        cfg["tool"]["flwr"]["app"]["config"]["batch-size"] = batch_size
        cfg["tool"]["flwr"]["app"]["config"]["local-epochs"] = num_local_epochs 
        cfg["tool"]["flwr"]["app"]["config"]["window-size"] = window_size 
        
        with open("pyproject.toml", "w") as f:
            toml.dump(cfg, f)

        proc = subprocess.run(["flwr", "run"])

tune_model(optimizer="adam")
tune_model(optimizer="sgd")
