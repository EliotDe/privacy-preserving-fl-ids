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
with open(f"{cwd}/results/tune-attack-lbfgs.jsonl","r") as f:
    for line in f: 
        if not line.strip(): continue

        result = json.loads(line)
        # Get relevant hyperparameters
        config = result.get("config",{})
        parameters = { 
            "attack-lr": config.get("attack-lr"),
            "attack-rounds": config.get("attack-rounds"),
            "attack-max-iter": config.get("attack-max-iter"),
            "attack-history-size": config.get("attack-history-size")
            #"attack-reg": config.get("attack-reg")
        }

        # Extract server metrics
        rounds = config.get("num-server-rounds")
        flwr_str = result.get("flwr_results","")
        final_server_metrics = extract_metric_record(flwr_str, "ServerApp-side Evaluate Metrics")        
        multi_round_attack_metrics = final_server_metrics[-1]
        multi_round_attack_metrics = {
                #"multi_mse_avg": multi_round_attack_metrics.get("multi_mse_avg"),
                "multi_mse_min": float(multi_round_attack_metrics.get("multi_mse_min")),
                #"multi_pcc_avg": multi_round_attack_metrics.get("multi_pcc_avg"),
                "multi_pcc_min": float(multi_round_attack_metrics.get("multi_pcc_min")),
                "multi_pcc_max": float(multi_round_attack_metrics.get("multi_pcc_max")),
        }
        other_metrics = final_server_metrics[rounds]
        dlg_metrics = {
                #"dlg_mse_avg": other_metrics.get("dlg_mse_avg"),
                "dlg_mse_min": float(other_metrics.get("dlg_mse_min")),
                #"dlg_pcc_avg": other_metrics.get("dlg_pcc_avg"),
                "dlg_pcc_min": float(other_metrics.get("dlg_pcc_min")),
                "dlg_pcc_max": float(other_metrics.get("dlg_pcc_max"))
        }
        dlg_cossim_metrics = {
                #"dlg_cossim_mse_avg": other_metrics.get("dlg_cossim_mse_avg"),
                "dlg_cossim_mse_min": float(other_metrics.get("dlg_cossim_mse_min")),
                #"dlg_cossim_pcc_avg": other_metrics.get("dlg_cossim_pcc_avg"),
                "dlg_cossim_pcc_min": float(other_metrics.get("dlg_cossim_pcc_min")),
                "dlg_cossim_pcc_max": float(other_metrics.get("dlg_cossim_pcc_max"))
        }

        runs.append({
            "parameters": parameters, 
            "multi_round_attack_metrics": multi_round_attack_metrics,
            "dlg_attack_metrics": dlg_metrics,
            "dlg_cossim_metrics": dlg_cossim_metrics
        })

sorted_runs = sorted(runs, key=lambda item: item["dlg_cossim_metrics"]["dlg_cossim_mse_min"])
print(f"======== HIGHEST PERFORMING RUNS BY FINAL GLOBAL ACCURACY ========\n\n")
for i, run in enumerate(sorted_runs):
    print(f"rank: {i} \n\tparameters: {run["parameters"]} \n\tmulti round attack metrics: {run["multi_round_attack_metrics"]} \n\tdlg attack metrics: {run["dlg_attack_metrics"]} \n\tdlg cossim metrics: {run["dlg_cossim_metrics"]}")

