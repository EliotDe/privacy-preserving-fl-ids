import toml 
import json
import re
import subprocess


epsilons = [10.0, 7.5, 5.0, 2.5, 1.0, 0.5, 0.1]


for eps in epsilons:
    with open("pyproject.toml") as f:
        cfg = toml.load(f)

    cfg["tool"]["flwr"]["app"]["config"]["epsilon"] = eps
    
    with open("pyproject.toml", "w") as f:
        toml.dump(cfg, f)

    proc = subprocess.run(["flwr", "run"])

#    outputs = proc.stdout
#    errors = proc.stderr

    # Parse Results 

    
#    record={
#        "epsilon": eps,
#        "flower_output:" outputs,
#        "parsed_results:" 
#    }

#    with open("all_results.jsonl", "a") as f:
#        f.write(json.dumps(record) + "\n")
