import os
import re
import ast
import json
import pandas as pd

## 1. Read results
## 2. Sort by accuracy
## 3. Print hyperparameters (lr, batchsize, windowsize, local_epochs) & final accuracies

def extract_metric_record(results_str, section):
    """
    AI-GENERATED: I didn't write this
    """
    pattern = rf"{section}:(.*?)(?:Aggregated|ServerApp|$)"
    match = re.search(pattern, results_str, re.DOTALL)

    if match:
        dict_str = match.group(1).strip()
        try:
            return ast.literal_eval(dict_str)
        except Exception as e:
            print(f"Error Parsing {section}: {e}")
    return {}


runs = [] 
cwd = os.getcwd()
with open(f"{cwd}/results/model_tuning_results.jsonl","r") as f:
    for line in f: 
        if not line.strip(): continue

        result = json.loads(line)
        # Get relevant hyperparameters
        config = result.get("config",{})
        parameters = { 
            "lr": config.get("learning-rate"),
            "local-epochs": config.get("local-epochs"),
            "batch-size": config.get("batch-size"),
            "window-size": config.get("window-size")
        }

        # Extract server metrics
        rounds = config.get("num-server-rounds")
        flwr_str = result.get("flwr_results","")
        final_server_metrics = extract_metric_record(flwr_str, "ServerApp-side Evaluate Metrics")[rounds]
        
        runs.append({
            "parameters": parameters, 
            "accuracy": float(final_server_metrics.get("accuracy")),
            "loss": float(final_server_metrics.get("loss"))
        })

sorted_runs = sorted(runs, key=lambda item: item["accuracy"], reverse=True)
print(f"======== HIGHEST PERFORMING RUNS BY FINAL GLOBAL ACCURACY ========\n\n")
for i, run in enumerate(sorted_runs[:40]):
    print(f"rank: {i} \tparameters: {run["parameters"]} \taccuracy: {run["accuracy"]}")

