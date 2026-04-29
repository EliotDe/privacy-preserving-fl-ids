"""
BEFORE RUNNING THIS SCRIPT ENSURE YOU HAVE DONE THE FOLLOWING:
    - In flids/task.py you have removed the batch norm layer in NN and the method NN.forward()
    - In flids/client_app.py you have added a noise modifier to the training decorator 
    - Ensure you are using Adam (as the note mentions this has been run for sgd already) 

EXPERIMENT #3: TUNE NOISE WITHOUT BATCH NORM - USING ADAM
    - fixed variables: 
        - num_clients = 100
        - epsilon = 10
"""
import toml 
import json
import re
import subprocess
import random
from pathlib import Path

sensitivities = [0.5,1,2,4]
deltas = [0.00001, 0.000001]
clipping_norms = [0.5,1,2,5,10]

runs = 20

seen = set()
config_path = Path.home() / ".flwr" / "config.toml"
for _ in range(runs):
    while True:
        config = (
            random.choice(sensitivities),
            random.choice(deltas),
            random.choice(clipping_norms),
        )

        if config not in seen:
            seen.add(config)
            break
    sensitivity, delta, clipping_norm = config

    # Configure Client Number
    with open(config_path) as f:
        cfg = toml.load(f)
    
    cfg["superlink"]["default"] = "local-simulation"
    cfg["superlink"]["local-simulation"]["options"]["num-supernodes"] = 100

    with open(config_path,"w") as f:
        toml.dump(cfg, f)

    with open("pyproject.toml") as f:
        cfg = toml.load(f)

    # Use tuning results to set learning parameters
    cfg["tool"]["flwr"]["app"]["config"]["experiment-name"] = "3-tune-noise-adam-fedavg" 
    cfg["tool"]["flwr"]["app"]["config"]["num-clients"] = 100 
    cfg["tool"]["flwr"]["app"]["config"]["shuffle"] = True 
    cfg["tool"]["flwr"]["app"]["config"]["num-server-rounds"] = 35 
    cfg["tool"]["flwr"]["app"]["config"]["optimizer"] = "adam" 
    cfg["tool"]["flwr"]["app"]["config"]["learning-rate"] = 0.001 
    cfg["tool"]["flwr"]["app"]["config"]["batch-size"] = 32 
    cfg["tool"]["flwr"]["app"]["config"]["local-epochs"] = 10 
    cfg["tool"]["flwr"]["app"]["config"]["window-size"] = 20
    # No Attack
    cfg["tool"]["flwr"]["app"]["config"]["run-inversion"] = False 
    cfg["tool"]["flwr"]["app"]["config"]["attack-lr"] = 0 
    cfg["tool"]["flwr"]["app"]["config"]["attack-rounds"] = 0 
    cfg["tool"]["flwr"]["app"]["config"]["attack-max-iter"] = 0 
    cfg["tool"]["flwr"]["app"]["config"]["attack-history-size"] = 0 
    cfg["tool"]["flwr"]["app"]["config"]["attack-reg"] = 0 
    # Run FedAvg
    cfg["tool"]["flwr"]["app"]["config"]["prox-mu"] = 0 
    # Set variables
    cfg["tool"]["flwr"]["app"]["config"]["epsilon"] = 10 
    cfg["tool"]["flwr"]["app"]["config"]["delta"] = delta
    cfg["tool"]["flwr"]["app"]["config"]["sensitivity"] = sensitivity 
    cfg["tool"]["flwr"]["app"]["config"]["clipping_norm"] = clipping_norm 
       
    with open("pyproject.toml", "w") as f:
        toml.dump(cfg, f)

    proc = subprocess.run(["flwr", "run"])
