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

    
def evaluate_result(experiment_name: str):
    runs = [] 
    cwd = os.getcwd()
    with open(f"{cwd}/results/{experiment_name}.jsonl","r") as f:
        for line in f: 
            if not line.strip(): continue

            result = json.loads(line)
            # Get relevant hyperparameters
            config = result.get("config",{})
            epsilon = config.get("epsilon")
            num_clients = config.get("num-clients")

            if num_clients != 150:
                continue

            # Extract server metrics
            rounds = config.get("num-server-rounds")
            flwr_str = result.get("flwr_results","")

            round_server_metrics = extract_metric_record(flwr_str, "ServerApp-side Evaluate Metrics")[rounds]
            
            accuracy_per_class = {
                "acc_normal": float(round_server_metrics.get("acc_normal")),
                "acc_scanning": float(round_server_metrics.get("acc_scanning")),
                "acc_dos": float(round_server_metrics.get("acc_dos")),
                "acc_injection": float(round_server_metrics.get("acc_injection")),
                "acc_ddos": float(round_server_metrics.get("acc_ddos")),
                "acc_ransomware": float(round_server_metrics.get("acc_ransomware")),
                "acc_xss": float(round_server_metrics.get("acc_xss")),
                "acc_mitm": float(round_server_metrics.get("acc_mitm")),
                "acc_password": float(round_server_metrics.get("acc_password")),
                "acc_backdoor": float(round_server_metrics.get("acc_backdoor")),
            }
            runs.append({
                "epsilon": epsilon, 
                "acc_per_class": accuracy_per_class
            })

    sorted_runs = sorted(runs, key=lambda item: item["epsilon"], reverse=True)
    print(f"======== HIGHEST PERFORMING RUNS BY FINAL GLOBAL ACCURACY ({experiment_name})========\n\n")
    for i, run in enumerate(sorted_runs):
        print(f"rank: {i} \tepsilon: {run["epsilon"]} \taccuracy per class: {run["acc_per_class"]}")

evaluate_result("vary-noise-model-sgd-fedavg")
