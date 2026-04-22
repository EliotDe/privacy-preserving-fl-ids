import os
import re
import ast
import json
import pandas as pd
import matplotlib.pyplot as plt


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


def make_plot(epsilon: int, client_metrics: dict[int, dict], y_label:str):
    fig, ax = plt.subplots()

    styles = {
        5:   {"color": "#000000", "linestyle": "-",                "marker": "o"},  # black
        10:  {"color": "#0072B2", "linestyle": "--",               "marker": "s"},  # blue
        25:  {"color": "#D55E00", "linestyle": "-.",               "marker": "^"},  # vermillion
        50:  {"color": "#009E73", "linestyle": ":",                "marker": "D"},  # green
        75:  {"color": "#CC79A7", "linestyle": (0, (3,1,1,1)),     "marker": "v"},  # purple
        100: {"color": "#E69F00", "linestyle": (0, (5,1)),         "marker": "P"},  # orange
        150: {"color": "#56B4E9", "linestyle": (0, (1,1)),         "marker": "X"},  # sky blue
    }

    lines = []
    clients = client_metrics.keys()
    for client in clients:
        if y_label == "accuracy":
            if client > 50:
                rounds = client_metrics[client].keys()
                losses = client_metrics[client].values()
                lines += ax.plot(
                        rounds, 
                        losses, 
                        label=f"{client} clients", 
                        marker=styles[client]["marker"], 
                        color=styles[client]["color"],
                        linestyle=styles[client]["linestyle"],
                        linewidth=1.5,
                        markersize=4
                ) 
        else:
            rounds = client_metrics[client].keys()
            losses = client_metrics[client].values()
            lines += ax.plot(
                    rounds, 
                    losses, 
                    label=f"{client} clients", 
                    marker=styles[client]["marker"], 
                    color=styles[client]["color"],
                    linestyle=styles[client]["linestyle"],
                    linewidth=1.5,
                    markersize=4
            )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--',linewidth=0.5,alpha=0.8)
    if y_label == "loss":
        ax.set_yscale("log")
        ax.set_ylim([1,10**6])
    fig.legend(loc="center right", bbox_to_anchor=(1.125,0.5))
    fig.supxlabel("Rounds")
    fig.supylabel(f"{y_label}",x=0.02)
    fig.suptitle(f"{y_label} Over Rounds (epsilon={epsilon})",y=0.97)
    plt.grid(True)
    plt.savefig(f"figures/{y_label}_over_rounds_eps_{epsilon}.png", bbox_inches='tight')



    
def evaluate_result(experiment_name: str):
    # For each pair of (epsilon,num_clients) record the accuracy and loss over the training rounds
    loss_over_rounds: dict[tuple(float,int),list[float]] = {}
    acc_over_rounds: dict[tuple(float,int),list[float]] = {}
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
            parameters = { 
                "epsilon": epsilon,
                "num-clients": num_clients          
            }

            # Extract server metrics
            rounds = config.get("num-server-rounds")
            flwr_str = result.get("flwr_results","")

            losses: dict[int,float] = {} 
            accuracies: dict[int,float] = {} 
            for r in range(1, rounds+1):
                round_server_metrics = extract_metric_record(flwr_str, "ServerApp-side Evaluate Metrics")[r]
                losses[r] = float(round_server_metrics.get("loss"))
                accuracies[r] = float(round_server_metrics.get("accuracy"))
                if r == rounds:
                    runs.append({
                        "parameters": parameters, 
                        "accuracy": float(round_server_metrics.get("accuracy")),
                        "loss": float(round_server_metrics.get("loss")),
                        "f1-score": float(round_server_metrics.get("f1-score"))
                    })
            loss_over_rounds[(epsilon,num_clients)] = losses
            acc_over_rounds[(epsilon,num_clients)] = accuracies

    sorted_runs = sorted(runs, key=lambda item: item["accuracy"], reverse=True)
    #print(f"======== LOSS OVER TRAINING ROUNDS FOR EPSILON========\n\n")
    final_loss_dict = {}
    for k, losses in loss_over_rounds.items():
        epsilon, num_clients = k
        final_loss_dict[epsilon] = {}
    for k, losses in loss_over_rounds.items():
        epsilon, num_clients = k
        final_loss_dict[epsilon][num_clients] = losses

    for k, v in final_loss_dict.items():
        make_plot(k, v, "loss")        
    
    final_acc_dict = {}
    for k, losses in acc_over_rounds.items():
        epsilon, num_clients = k
        final_acc_dict[epsilon] = {}
    for k, losses in acc_over_rounds.items():
        epsilon, num_clients = k
        final_acc_dict[epsilon][num_clients] = losses

    for k, v in final_acc_dict.items():
        make_plot(k, v, "accuracy")        


evaluate_result("vary-noise-model-sgd-fedavg")
