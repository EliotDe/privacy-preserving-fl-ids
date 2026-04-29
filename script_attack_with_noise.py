"""
BEFORE RUNNING THIS SCRIPT ENSURE YOU HAVE DONE THE FOLLOWING:
    - In flids/task.py the model should NOT be using batch norm layers
    - In flids/client_app.py the train method SHOULD be decorated with a noise modifier

EXPERIMENT #8: RECOVERABLE INFORMATION FOR DIFFERENT LEVELS OF NOISE 
    - Fixed variables: Fl optimizer - SGD (all variables apart from batch size, local_batches and local_epochs are fixed)
"""

import toml 
import json
import re
import subprocess
import random
from pathlib import Path


epsilons = [10.0, 7.5, 5.0, 2.5]

config_path = Path.home() / ".flwr" / "config.toml"


for epsilon in epsilons:
    # Configure Client Number
    with open(config_path) as f:
        cfg = toml.load(f)
    
    cfg["superlink"]["default"] = "local-simulation"
    cfg["superlink"]["local-simulation"]["options"]["num-supernodes"] = 150 

    with open(config_path,"w") as f:
        toml.dump(cfg, f)


    # Configure Experiment
    with open("pyproject.toml") as f:
        cfg = toml.load(f)

    cfg["tool"]["flwr"]["app"]["config"]["experiment-name"] = "8-attck-across-noise-sgd-fedavg"
    cfg["tool"]["flwr"]["app"]["config"]["num-clients"] = 150        # LBFGS is computationally expensive 
    cfg["tool"]["flwr"]["app"]["config"]["shuffle"] = False 
    cfg["tool"]["flwr"]["app"]["config"]["num-server-rounds"] = 2 
    # Attack Settings
    cfg["tool"]["flwr"]["app"]["config"]["run-inversion"] = True 
    cfg["tool"]["flwr"]["app"]["config"]["attack-lr"] = 1.0         # For LBFGS 
    cfg["tool"]["flwr"]["app"]["config"]["attack-rounds"] = 25 
    cfg["tool"]["flwr"]["app"]["config"]["attack-max-iter"] = 5 
    cfg["tool"]["flwr"]["app"]["config"]["attack-history-size"] = 10 
    cfg["tool"]["flwr"]["app"]["config"]["attack-reg"] = 0 
    cfg["tool"]["flwr"]["app"]["config"]["num-clients-to-attack"] = 10 
    # Run FedAvg
    cfg["tool"]["flwr"]["app"]["config"]["prox-mu"] = 0 
    # No Noise
    cfg["tool"]["flwr"]["app"]["config"]["epsilon"] = epsilon 
    cfg["tool"]["flwr"]["app"]["config"]["delta"] =  0.00001
    cfg["tool"]["flwr"]["app"]["config"]["sensitivity"] = 0.5 
    cfg["tool"]["flwr"]["app"]["config"]["clipping_norm"] = 2 
    # Set variables
    cfg["tool"]["flwr"]["app"]["config"]["optimizer"] = "adam" 
    cfg["tool"]["flwr"]["app"]["config"]["learning-rate"] = 0.001 
    cfg["tool"]["flwr"]["app"]["config"]["batch-size"] = 1 
    cfg["tool"]["flwr"]["app"]["config"]["local-epochs"] = 1 
    cfg["tool"]["flwr"]["app"]["config"]["local-batches"] = 1 
    cfg["tool"]["flwr"]["app"]["config"]["window-size"] = 20 
    
    with open("pyproject.toml", "w") as f:
        toml.dump(cfg, f)

    proc = subprocess.run(["flwr", "run"])


