import toml 
import json
import re
import subprocess
import random

## Initial search space

#lrs = [0.1, 0.03, 0.01, 0.001, 0.0001]
#local_epochs = [1, 3, 5, 10]
#batch_sizes = [16, 32, 64]
#window_sizes = [20, 30, 40, 50]

## Next search space
sensitivities = [0.5,1,2,4]
deltas = [0.00001, 0.000001]
clipping_norms = [0.5,1,2,5,10]

runs = 30 

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

    cfg["tool"]["flwr"]["app"]["config"]["experiment-name"] = "tune-noise" 
    cfg["tool"]["flwr"]["app"]["config"]["run-inversion"] = False 
    cfg["tool"]["flwr"]["app"]["config"]["shuffle"] = True 
    cfg["tool"]["flwr"]["app"]["config"]["prox-mu"] = 0 
    cfg["tool"]["flwr"]["app"]["config"]["learning-rate"] = 0.001 
    cfg["tool"]["flwr"]["app"]["config"]["batch-size"] = 32 
    cfg["tool"]["flwr"]["app"]["config"]["local-epochs"] = 3 
    cfg["tool"]["flwr"]["app"]["config"]["window-size"] = 30 
    cfg["tool"]["flwr"]["app"]["config"]["epsilon"] = 10    # Fixed epsilon for initial tuning
    cfg["tool"]["flwr"]["app"]["config"]["delta"] = delta 
    cfg["tool"]["flwr"]["app"]["config"]["sensitivity"] = sensitivity 
    cfg["tool"]["flwr"]["app"]["config"]["clipping_norm"] = clipping_norm 

    
    with open("pyproject.toml", "w") as f:
        toml.dump(cfg, f)

    proc = subprocess.run(["flwr", "run"])
