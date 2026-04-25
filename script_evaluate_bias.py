import os
import re
import ast
import json
import pandas as pd
import plotly.graph_objects as go


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


def generate_polar_plot(data: dict[float, list[float]], categories, fig_name:str, save_to:str):
    fig = go.Figure()
    for name, values in data.items():
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=name,
            opacity=0.7
        ))
    fig.update_layout(
        title=fig_name,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0,1],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                direction="clockwise"
            )
        ),
        legend=dict(
            x=1.05,
            y=1
        )
    )
    fig.write_image(save_to)
        

def get_acc_per_class(round_server_metrics):
    accuracy_per_class = [ 
        float(round_server_metrics.get("acc_normal")),
        float(round_server_metrics.get("acc_ddos")),
        float(round_server_metrics.get("acc_dos")),
        float(round_server_metrics.get("acc_xss")),
        float(round_server_metrics.get("acc_scanning")),
        float(round_server_metrics.get("acc_injection")),
        float(round_server_metrics.get("acc_mitm")),
        float(round_server_metrics.get("acc_ransomware")),
        float(round_server_metrics.get("acc_password")),
        float(round_server_metrics.get("acc_backdoor"))
    ] 
    return accuracy_per_class


    
def evaluate_result(no_noise_experiment_name: str, noise_experiment_name: str, fig_name: str, save_to: str):
    runs = {} 
    cwd = os.getcwd()

    categories = [
        "Normal", "DDoS", "DoS", "XSS", "Scanning", 
        "Injection", "MITM", "Ransomware", "Password", "Backdoor"
    ]


    with open(f"{cwd}/results/{no_noise_experiment_name}.jsonl","r") as f:
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
            
            accuracy_per_class = get_acc_per_class(round_server_metrics) 
            runs["No Noise"] = accuracy_per_class

    
    with open(f"{cwd}/results/{noise_experiment_name}.jsonl","r") as f:
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
            
            accuracy_per_class = get_acc_per_class(round_server_metrics)
            runs[f"eps = {epsilon}"] = accuracy_per_class

     
    generate_polar_plot(runs, categories, fig_name=fig_name, save_to=save_to)

evaluate_result("sgd-baseline-fedavg-across-clients" , "vary-noise-model-sgd-fedavg", fig_name="Acc Per Class -- FedAvg", save_to="figures/acc_per_class_fedavg.png")
evaluate_result("sgd-baseline-fedavg-across-clients" , "vary-noise-model-sgd-fedprox", fig_name="Acc Per Class -- FedProx", save_to="figures/acc_per_class_fedprox.png")
