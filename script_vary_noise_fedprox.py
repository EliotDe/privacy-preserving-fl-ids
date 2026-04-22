import toml 
import json
import re
import subprocess
import random
from pathlib import Path

eps = [10, 7.5, 5, 2.5]
prox_mus = [0.001, 0.0001, 0.00001]
clients = 150

config_path = Path.home() / ".flwr" / "config.toml"
for e in eps:
    for prox_mu in prox_mus:
        # Configure Client Number
        with open(config_path) as f:
            cfg = toml.load(f)
        
        cfg["superlink"]["default"] = "local-simulation"
        cfg["superlink"]["local-simulation"]["options"]["num-supernodes"] = clients 

        with open(config_path,"w") as f:
            toml.dump(cfg, f)


        # Configure Experiment
        with open("pyproject.toml") as f:
            cfg = toml.load(f)

        cfg["tool"]["flwr"]["app"]["config"]["experiment-name"] = "vary-noise-model-sgd-fedprox" 
        cfg["tool"]["flwr"]["app"]["config"]["num-clients"] = clients 
        cfg["tool"]["flwr"]["app"]["config"]["shuffle"] = True 
        cfg["tool"]["flwr"]["app"]["config"]["num-server-rounds"] = 50 
        # No Attack
        cfg["tool"]["flwr"]["app"]["config"]["run-inversion"] = False 
        cfg["tool"]["flwr"]["app"]["config"]["attack-lr"] = 0 
        cfg["tool"]["flwr"]["app"]["config"]["attack-rounds"] = 0 
        cfg["tool"]["flwr"]["app"]["config"]["attack-max-iter"] = 0 
        cfg["tool"]["flwr"]["app"]["config"]["attack-history-size"] = 0 
        cfg["tool"]["flwr"]["app"]["config"]["attack-reg"] = 0 
        # Run FedProx 
        cfg["tool"]["flwr"]["app"]["config"]["prox-mu"] = prox_mu 
        # Add Noise
        cfg["tool"]["flwr"]["app"]["config"]["epsilon"] = e
        cfg["tool"]["flwr"]["app"]["config"]["delta"] =  0.00001
        cfg["tool"]["flwr"]["app"]["config"]["sensitivity"] = 0.5 
        cfg["tool"]["flwr"]["app"]["config"]["clipping_norm"] = 5 
        # Set variables
        cfg["tool"]["flwr"]["app"]["config"]["learning-rate"] = 0.001 
        cfg["tool"]["flwr"]["app"]["config"]["batch-size"] = 32
        cfg["tool"]["flwr"]["app"]["config"]["local-epochs"] = 3 
        cfg["tool"]["flwr"]["app"]["config"]["window-size"] = 30 
        
        with open("pyproject.toml", "w") as f:
            toml.dump(cfg, f)

        proc = subprocess.run(["flwr", "run"])
