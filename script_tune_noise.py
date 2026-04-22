import toml 
import json
import re
import subprocess
import random

## Next search space
sensitivities = [0.5,1,2,4]
deltas = [0.00001, 0.000001]
clipping_norms = [0.5,1,2,5,10]

runs = 25 

seen = set()
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

    with open("pyproject.toml") as f:
        cfg = toml.load(f)

    # Use tuning results to set learning parameters
    cfg["tool"]["flwr"]["app"]["config"]["experiment-name"] = "tune-noise-100-clients" 
    cfg["tool"]["flwr"]["app"]["config"]["shuffle"] = True 
    cfg["tool"]["flwr"]["app"]["config"]["num-server-rounds"] = 35 
    cfg["tool"]["flwr"]["app"]["config"]["learning-rate"] = 0.1 
    cfg["tool"]["flwr"]["app"]["config"]["batch-size"] = 32 
    cfg["tool"]["flwr"]["app"]["config"]["local-epochs"] = 5 
    cfg["tool"]["flwr"]["app"]["config"]["window-size"] = 30
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
