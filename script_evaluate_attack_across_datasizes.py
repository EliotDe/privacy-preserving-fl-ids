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


def eval_attack_across_learning_params():
    runs = [] 
    cwd = os.getcwd()
    with open(f"{cwd}/results/7-attack-across-dimensions-sgd-fedavg.jsonl","r") as f:
        for line in f: 
            if not line.strip(): continue

            result = json.loads(line)
            # Get relevant hyperparameters
            config = result.get("config",{})
            parameters = { 
                "batch-size": int(config.get("batch-size")),
                "local-epochs": int(config.get("local-epochs")),
                "local-batches": int(config.get("local-batches")),
                "window-size": int(config.get("window-size")) 
                #"attack-reg": config.get("attack-reg")
            }

            # Extract server metrics
            rounds = config.get("num-server-rounds")
            flwr_str = result.get("flwr_results","")
            final_server_metrics = extract_metric_record(flwr_str, "ServerApp-side Evaluate Metrics")        
            multi_round_attack_metrics = final_server_metrics[-1]
            multi_round_attack_metrics = {
                    "multi_mse_avg": float(multi_round_attack_metrics.get("multi_mse_avg")),
                    "multi_mse_min": float(multi_round_attack_metrics.get("multi_mse_min")),
                    "multi_pcc_avg": float(multi_round_attack_metrics.get("multi_pcc_avg")),
                   # "multi_pcc_min": float(multi_round_attack_metrics.get("multi_pcc_min")),
                   # "multi_pcc_max": float(multi_round_attack_metrics.get("multi_pcc_max")),
            }
            other_metrics = final_server_metrics[rounds]
            dlg_metrics = {
                    "dlg_mse_avg": float(other_metrics.get("dlg_mse_avg")),
                    "dlg_mse_min": float(other_metrics.get("dlg_mse_min")),
                    "dlg_pcc_avg": float(other_metrics.get("dlg_pcc_avg")),
                    #"dlg_pcc_min": float(other_metrics.get("dlg_pcc_min")),
                    #"dlg_pcc_max": float(other_metrics.get("dlg_pcc_max"))
            }
            dlg_cossim_metrics = {
                    "dlg_cossim_mse_avg": float(other_metrics.get("dlg_cossim_mse_avg")),
                    "dlg_cossim_mse_min": float(other_metrics.get("dlg_cossim_mse_min")),
                    "dlg_cossim_pcc_avg": float(other_metrics.get("dlg_cossim_pcc_avg")),
                   # "dlg_cossim_pcc_min": float(other_metrics.get("dlg_cossim_pcc_min")),
                   # "dlg_cossim_pcc_max": float(other_metrics.get("dlg_cossim_pcc_max"))
            }

            runs.append({
                "parameters": parameters, 
                "multi_round_attack_metrics": multi_round_attack_metrics,
                "dlg_metrics": dlg_metrics,
                "dlg_cossim_metrics": dlg_cossim_metrics
            })

        # Construct dataframe to average results
        rows = []
        for run in runs:
            row = {**run["parameters"], **run["multi_round_attack_metrics"], **run["dlg_metrics"], **run["dlg_cossim_metrics"]}
            rows.append(row)
        df = pd.DataFrame(rows)
        group_cols = ["batch-size","local-epochs","local-batches","window-size"]
        averaged = df.groupby(group_cols).agg("mean")

        print(f"======== ATTACK ACROSS LEARNING PARAMS ========\n\n")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(averaged)


    #sorted_runs = sorted(runs, key=lambda item: item["dlg_metrics"]["dlg_mse_avg"])
    #print(f"======== HIGHEST PERFORMING RUNS BY FINAL GLOBAL ACCURACY LEARNING PARAMS ========\n\n")
    #for i, run in enumerate(runs):
    #    print(f"rank: {i} \n\tparameters: {run["parameters"]} \n\tmulti round attack metrics: {run["multi_round_attack_metrics"]} \n\tdlg attack metrics: {run["dlg_metrics"]} \n\tdlg cossim metrics: {run["dlg_cossim_metrics"]}")


def eval_attack_across_noise(optimizer:str):
    runs = [] 
    cwd = os.getcwd()
    with open(f"{cwd}/results/8-attck-across-noise-{optimizer}-fedavg.jsonl","r") as f:
        for line in f: 
            if not line.strip(): continue

            result = json.loads(line)
            # Get relevant hyperparameters
            config = result.get("config",{})
            parameters = { 
                "epsilon": float(config.get("epsilon"))
            }

            # Extract server metrics
            rounds = config.get("num-server-rounds")
            flwr_str = result.get("flwr_results","")
            final_server_metrics = extract_metric_record(flwr_str, "ServerApp-side Evaluate Metrics")        
            multi_round_attack_metrics = final_server_metrics[-1]
            multi_round_attack_metrics = {
                    "multi_mse_avg": float(multi_round_attack_metrics.get("multi_mse_avg")),
                    "multi_mse_min": float(multi_round_attack_metrics.get("multi_mse_min")),
                    "multi_pcc_avg": float(multi_round_attack_metrics.get("multi_pcc_avg")),
                   # "multi_pcc_min": float(multi_round_attack_metrics.get("multi_pcc_min")),
                   # "multi_pcc_max": float(multi_round_attack_metrics.get("multi_pcc_max")),
            }
            other_metrics = final_server_metrics[rounds]
            dlg_metrics = {
                    "dlg_mse_avg": float(other_metrics.get("dlg_mse_avg")),
                    "dlg_mse_min": float(other_metrics.get("dlg_mse_min")),
                    "dlg_pcc_avg": float(other_metrics.get("dlg_pcc_avg")),
                    #"dlg_pcc_min": float(other_metrics.get("dlg_pcc_min")),
                    #"dlg_pcc_max": float(other_metrics.get("dlg_pcc_max"))
            }
            dlg_cossim_metrics = {
                    "dlg_cossim_mse_avg": float(other_metrics.get("dlg_cossim_mse_avg")),
                    "dlg_cossim_mse_min": float(other_metrics.get("dlg_cossim_mse_min")),
                    "dlg_cossim_pcc_avg": float(other_metrics.get("dlg_cossim_pcc_avg")),
                   # "dlg_cossim_pcc_min": float(other_metrics.get("dlg_cossim_pcc_min")),
                   # "dlg_cossim_pcc_max": float(other_metrics.get("dlg_cossim_pcc_max"))
            }

            runs.append({
                "parameters": parameters, 
                "multi_round_attack_metrics": multi_round_attack_metrics,
                "dlg_metrics": dlg_metrics,
                "dlg_cossim_metrics": dlg_cossim_metrics
            })
        rows = []
        for run in runs:
            row = {**run["parameters"], **run["multi_round_attack_metrics"], **run["dlg_metrics"], **run["dlg_cossim_metrics"]}
            rows.append(row)
        df = pd.DataFrame(rows)
        group_cols = ["epsilon"]
        averaged = df.groupby(group_cols).agg("mean")

        print(f"======== ATTACK RESULTS ACROSS NOISE AVERAGED ({optimizer})========\n\n")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(averaged)

eval_attack_across_learning_params()
eval_attack_across_noise("adam")
eval_attack_across_noise("sgd")
