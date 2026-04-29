"""
BEFORE RUNNING THIS SCRIPT ENSURE YOU HAVE DONE THE FOLLOWING:
    - In flids/task.py the model should be using batch norm layers
    - In flids/client_app.py the train method should NOT be decorated with a noise modifier

EXPERIMENT #5: BASELINE FOR VARYING NOISE ACROSS CLIENTS: ADAM -- BATCH NORM -- NO ATTACK
    - fixed variables: Adam Optimizer (all other variables except for num_clients are fixed as well)
"""

import toml 
import json
import re
import subprocess
import random
from pathlib import Path


clients = [5, 10, 25, 50, 75, 100,150]

config_path = Path.home() / ".flwr" / "config.toml"
for c in clients:
    # Configure Client Number
    with open(config_path) as f:
        cfg = toml.load(f)
    
    cfg["superlink"]["default"] = "local-simulation"
    cfg["superlink"]["local-simulation"]["options"]["num-supernodes"] = c 

    with open(config_path,"w") as f:
        toml.dump(cfg, f)


    # Configure Experiment
    with open("pyproject.toml") as f:
        cfg = toml.load(f)

    cfg["tool"]["flwr"]["app"]["config"]["experiment-name"] = "5-baseline-for-noise-across-clients-adam" 
    cfg["tool"]["flwr"]["app"]["config"]["num-clients"] = c 
    cfg["tool"]["flwr"]["app"]["config"]["shuffle"] = True 
    cfg["tool"]["flwr"]["app"]["config"]["num-server-rounds"] = 50 
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
    cfg["tool"]["flwr"]["app"]["config"]["delta"] =  0
    cfg["tool"]["flwr"]["app"]["config"]["sensitivity"] = 0
    cfg["tool"]["flwr"]["app"]["config"]["clipping_norm"] = 0 
    # Set variables
    cfg["tool"]["flwr"]["app"]["config"]["optimizer"] = "adam" 
    cfg["tool"]["flwr"]["app"]["config"]["learning-rate"] = 0.001 
    cfg["tool"]["flwr"]["app"]["config"]["batch-size"] = 32
    cfg["tool"]["flwr"]["app"]["config"]["local-epochs"] = 10 
    cfg["tool"]["flwr"]["app"]["config"]["window-size"] = 20 
    
    with open("pyproject.toml", "w") as f:
        toml.dump(cfg, f)

    proc = subprocess.run(["flwr", "run"])
