import toml 
import json
import re
import subprocess
import random
from pathlib import Path


batch_sizes = [1, 32, 64]
num_batches = [1, 8, 16, 32, 64]
num_epochs  = [1, 3, 10]

config_path = Path.home() / ".flwr" / "config.toml"


##====== VARY EACH PARAMATER INDIVIDUALLY ======## 
for local_batches in num_batches:
    # Configure Client Number
    with open(config_path) as f:
        cfg = toml.load(f)
    
    cfg["superlink"]["default"] = "local-simulation"
    cfg["superlink"]["local-simulation"]["options"]["num-supernodes"] = 10 

    with open(config_path,"w") as f:
        toml.dump(cfg, f)


    # Configure Experiment
    with open("pyproject.toml") as f:
        cfg = toml.load(f)

    cfg["tool"]["flwr"]["app"]["config"]["experiment-name"] = "attack-experiment-data-sizes-lbfgs" 
    cfg["tool"]["flwr"]["app"]["config"]["num-clients"] = 10        # LBFGS is computationally expensive 
    cfg["tool"]["flwr"]["app"]["config"]["shuffle"] = False 
    cfg["tool"]["flwr"]["app"]["config"]["num-server-rounds"] = 2 
    # Attack Settings
    cfg["tool"]["flwr"]["app"]["config"]["run-inversion"] = True 
    cfg["tool"]["flwr"]["app"]["config"]["attack-lr"] = 1.0         # For LBFGS 
    cfg["tool"]["flwr"]["app"]["config"]["attack-rounds"] = 25 
    cfg["tool"]["flwr"]["app"]["config"]["attack-max-iter"] = 5 
    cfg["tool"]["flwr"]["app"]["config"]["attack-history-size"] = 10 
    cfg["tool"]["flwr"]["app"]["config"]["attack-reg"] = 0 
    # Run FedAvg
    cfg["tool"]["flwr"]["app"]["config"]["prox-mu"] = 0 
    # No Noise
    cfg["tool"]["flwr"]["app"]["config"]["epsilon"] = 0 
    cfg["tool"]["flwr"]["app"]["config"]["delta"] =  0
    cfg["tool"]["flwr"]["app"]["config"]["sensitivity"] = 0
    cfg["tool"]["flwr"]["app"]["config"]["clipping_norm"] = 0 
    # Set variables
    cfg["tool"]["flwr"]["app"]["config"]["learning-rate"] = 0.001 
    cfg["tool"]["flwr"]["app"]["config"]["batch-size"] = 1 
    cfg["tool"]["flwr"]["app"]["config"]["local-epochs"] = 1 
    cfg["tool"]["flwr"]["app"]["config"]["local-batches"] = local_batches 
    cfg["tool"]["flwr"]["app"]["config"]["window-size"] = 30 
    
    with open("pyproject.toml", "w") as f:
        toml.dump(cfg, f)

    proc = subprocess.run(["flwr", "run"])


for bs in batch_sizes:
    # Configure Client Number
    with open(config_path) as f:
        cfg = toml.load(f)
    
    cfg["superlink"]["default"] = "local-simulation"
    cfg["superlink"]["local-simulation"]["options"]["num-supernodes"] = 10 

    with open(config_path,"w") as f:
        toml.dump(cfg, f)


    # Configure Experiment
    with open("pyproject.toml") as f:
        cfg = toml.load(f)

    cfg["tool"]["flwr"]["app"]["config"]["experiment-name"] = "attack-experiment-data-sizes-lbfgs" 
    cfg["tool"]["flwr"]["app"]["config"]["num-clients"] = 10        # LBFGS is computationally expensive 
    cfg["tool"]["flwr"]["app"]["config"]["shuffle"] = False 
    cfg["tool"]["flwr"]["app"]["config"]["num-server-rounds"] = 2 
    # Attack Settings
    cfg["tool"]["flwr"]["app"]["config"]["run-inversion"] = True 
    cfg["tool"]["flwr"]["app"]["config"]["attack-lr"] = 1.0         # For LBFGS 
    cfg["tool"]["flwr"]["app"]["config"]["attack-rounds"] = 25 
    cfg["tool"]["flwr"]["app"]["config"]["attack-max-iter"] = 5 
    cfg["tool"]["flwr"]["app"]["config"]["attack-history-size"] = 10 
    cfg["tool"]["flwr"]["app"]["config"]["attack-reg"] = 0 
    # Run FedAvg
    cfg["tool"]["flwr"]["app"]["config"]["prox-mu"] = 0 
    # No Noise
    cfg["tool"]["flwr"]["app"]["config"]["epsilon"] = 0 
    cfg["tool"]["flwr"]["app"]["config"]["delta"] =  0
    cfg["tool"]["flwr"]["app"]["config"]["sensitivity"] = 0
    cfg["tool"]["flwr"]["app"]["config"]["clipping_norm"] = 0 
    # Set variables
    cfg["tool"]["flwr"]["app"]["config"]["learning-rate"] = 0.001 
    cfg["tool"]["flwr"]["app"]["config"]["batch-size"] = bs 
    cfg["tool"]["flwr"]["app"]["config"]["local-epochs"] = 1  
    cfg["tool"]["flwr"]["app"]["config"]["local-batches"] = 1 
    cfg["tool"]["flwr"]["app"]["config"]["window-size"] = 30 

    with open("pyproject.toml", "w") as f:
        toml.dump(cfg, f)

    proc = subprocess.run(["flwr", "run"])



for epochs in num_epochs:
    # Configure Client Number
    with open(config_path) as f:
        cfg = toml.load(f)
    
    cfg["superlink"]["default"] = "local-simulation"
    cfg["superlink"]["local-simulation"]["options"]["num-supernodes"] = 10 

    with open(config_path,"w") as f:
        toml.dump(cfg, f)

    # Configure Experiment
    with open("pyproject.toml") as f:
        cfg = toml.load(f)

    cfg["tool"]["flwr"]["app"]["config"]["experiment-name"] = "attack-experiment-data-sizes-lbfgs" 
    cfg["tool"]["flwr"]["app"]["config"]["num-clients"] = 10        # LBFGS is computationally expensive 
    cfg["tool"]["flwr"]["app"]["config"]["shuffle"] = False 
    cfg["tool"]["flwr"]["app"]["config"]["num-server-rounds"] = 2 
    # Attack Settings
    cfg["tool"]["flwr"]["app"]["config"]["run-inversion"] = True 
    cfg["tool"]["flwr"]["app"]["config"]["attack-lr"] = 1.0         # For LBFGS 
    cfg["tool"]["flwr"]["app"]["config"]["attack-rounds"] = 25 
    cfg["tool"]["flwr"]["app"]["config"]["attack-max-iter"] = 5 
    cfg["tool"]["flwr"]["app"]["config"]["attack-history-size"] = 10 
    cfg["tool"]["flwr"]["app"]["config"]["attack-reg"] = 0 
    # Run FedAvg
    cfg["tool"]["flwr"]["app"]["config"]["prox-mu"] = 0 
    # No Noise
    cfg["tool"]["flwr"]["app"]["config"]["epsilon"] = 0 
    cfg["tool"]["flwr"]["app"]["config"]["delta"] =  0
    cfg["tool"]["flwr"]["app"]["config"]["sensitivity"] = 0
    cfg["tool"]["flwr"]["app"]["config"]["clipping_norm"] = 0 
    # Set variables
    cfg["tool"]["flwr"]["app"]["config"]["learning-rate"] = 0.001 
    cfg["tool"]["flwr"]["app"]["config"]["batch-size"] = 1 
    cfg["tool"]["flwr"]["app"]["config"]["local-epochs"] = epochs 
    cfg["tool"]["flwr"]["app"]["config"]["local-batches"] = 1 
    cfg["tool"]["flwr"]["app"]["config"]["window-size"] = 30 

    with open("pyproject.toml", "w") as f:
        toml.dump(cfg, f)

    proc = subprocess.run(["flwr", "run"])


    
