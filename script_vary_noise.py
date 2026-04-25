"""
BEFORE RUNNING THIS SCRIPT ENSURE YOU HAVE DONE THE FOLLOWING:
    - In pytorchexample/task.py the model should NOT be using batch norm layers
    - In pytorchexample/client_app.py the train method SHOULD be decorated with a noise modifier

EXPERIMENT #5: VARY NOISE ACROSS CLIENT NUMBER -- NO BATCH NORM -- NO ATTACK
    - fixed variables: Adam Optimizer (I have already run the experiment for sgd and batch norm)
         - Adam is less influenced by model initialisation and therefore by noise. It also seems to handle sparse local datasets better than SGD which is ideal for large numbers of clients in this experiment.
"""
import toml 
import json
import re
import subprocess
import random
from pathlib import Path

eps = [10, 7.5, 5, 2.5]
clients = [5, 10, 25, 50, 75, 100,150]

config_path = Path.home() / ".flwr" / "config.toml"
for e in eps:
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

        cfg["tool"]["flwr"]["app"]["config"]["experiment-name"] = "5-vary-noise-across-clients-adam"
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
        cfg["tool"]["flwr"]["app"]["config"]["epsilon"] = e
        cfg["tool"]["flwr"]["app"]["config"]["delta"] =  0.00001
        cfg["tool"]["flwr"]["app"]["config"]["sensitivity"] = 0.5 
        cfg["tool"]["flwr"]["app"]["config"]["clipping_norm"] = 2 
        # Set variables 
        cfg["tool"]["flwr"]["app"]["config"]["optimizer"] = "adam" 
        cfg["tool"]["flwr"]["app"]["config"]["learning-rate"] = 0.001 
        cfg["tool"]["flwr"]["app"]["config"]["batch-size"] = 32
        cfg["tool"]["flwr"]["app"]["config"]["local-epochs"] = 10 
        cfg["tool"]["flwr"]["app"]["config"]["window-size"] = 20 
        
        with open("pyproject.toml", "w") as f:
            toml.dump(cfg, f)

        proc = subprocess.run(["flwr", "run"])
