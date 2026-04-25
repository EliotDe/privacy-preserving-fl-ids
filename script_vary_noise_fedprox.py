"""
BEFORE RUNNING THIS SCRIPT ENSURE YOU HAVE DONE THE FOLLOWING:
    - In pytorchexample/task.py the model should NOT be using batch norm layers
    - In pytorchexample/client_app.py the train method SHOULD be decorated with a noise modifier
    - Change the name of processed_network_ransom.csv to processed_network.csv and processed_network.csv to whatever you like.

EXPERIMENT #6: HOW DOES FEDPROX MITIGATE BIAS ACROSS NOISE -- 150 CLIENTS -- NO BATCH NORM -- ADAM
    - variables: prox_mu and epsilon
    - fixed variables: num_clients = 150, optimizer = Adam
        - I have ran the experiment with sgd already
"""


import toml 
import json
import re
import subprocess
import random
from pathlib import Path

eps = [10, 7.5, 5, 2.5]
prox_mus = [0.001, 0.0001, 0.00001, 0]
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

        cfg["tool"]["flwr"]["app"]["config"]["experiment-name"] = "6-vary-noise-across-150-clients-adam-fedprox(1)" 
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
        cfg["tool"]["flwr"]["app"]["config"]["optimizer"] = "adam" 
        cfg["tool"]["flwr"]["app"]["config"]["learning-rate"] = 0.001 
        cfg["tool"]["flwr"]["app"]["config"]["batch-size"] = 32
        cfg["tool"]["flwr"]["app"]["config"]["local-epochs"] = 10 
        cfg["tool"]["flwr"]["app"]["config"]["window-size"] = 20 
        
        with open("pyproject.toml", "w") as f:
            toml.dump(cfg, f)

        proc = subprocess.run(["flwr", "run"])
