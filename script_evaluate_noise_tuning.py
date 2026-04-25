import os
import re
import ast
import json
import pandas as pd


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
with open(f"{cwd}/results/3-tune-noise-adam-fedavg.jsonl","r") as f:
    for line in f: 
        if not line.strip(): continue

        result = json.loads(line)
        # Get relevant hyperparameters
        config = result.get("config",{})
        parameters = { 
            "sensitivity": config.get("sensitivity"),
            "clipping_norm": config.get("clipping_norm"),
            "delta": config.get("delta")
        }

        # Extract server metrics
        rounds = config.get("num-server-rounds")
        flwr_str = result.get("flwr_results","")
        final_server_metrics = extract_metric_record(flwr_str, "ServerApp-side Evaluate Metrics")[rounds]        
        accuracy = float(final_server_metrics.get("accuracy"))
        loss = float(final_server_metrics.get("loss"))
        f1_score = float(final_server_metrics.get("f1-score"))

        runs.append({
            "parameters": parameters, 
            "final-accuracy": accuracy,
            "loss": loss,
            "f1-score": f1_score
        })

sorted_runs = sorted(runs, key=lambda item: item["final-accuracy"], reverse=True)
print(f"======== HIGHEST PERFORMING RUNS BY FINAL GLOBAL ACCURACY ========\n\n")
for i, run in enumerate(sorted_runs):
    print(f"rank: {i} \tparameters: {run["parameters"]} \taccuracy: {run["final-accuracy"]} \tf1-score: {run["f1-score"]} \tfinal-loss: {run["loss"]}")
