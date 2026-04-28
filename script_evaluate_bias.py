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

    theta_cats = categories + [categories[0]]

    
    for name, values in data.items():
        if "No Noise" in name:
            line_style = dict(width=2, dash="dash")
            fill_style = 'toself'
            mode = 'lines+markers'
            opacity = 0.2
        else:
            line_style = dict(width=2)
            fill_style = 'toself'
            mode = 'lines+markers'
            opacity = 0.2
        
        r_values = values + [values[0]]
        fig.add_trace(go.Scatterpolar(
            r=values + r_values,
            theta=theta_cats,
            fill=fill_style,
            name=name,
            mode=mode,
            opacity=1.0,
            #fillopacity=opacity,
            line=line_style,
            marker=dict(size=8)
        ))

    fig.update_layout(
        template="plotly_white",
        title=dict(text=fig_name, font=dict(size=20)),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0,1],
                gridcolor="lightgrey",
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                direction="clockwise",
                gridcolor="lightgrey",
                linecolor="black"
            )
        ),
        legend=dict(
            x=1.1,
            y=1
        ),
        margin=dict(
            l=80, r=80, t=100, b=80
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
    acc_sum = sum(accuracy_per_class)
    macro_recall = acc_sum / 10
    return accuracy_per_class, macro_recall


    
def evaluate_result(no_noise_experiment_name: str, noise_experiment_name: str, fig_name: str):
    runs = {} 
    cwd = os.getcwd()

    categories = [
        "Normal", "DDoS", "DoS", "XSS", "Scanning", 
        "Injection", "MITM", "Ransomware", "Password", "Backdoor"
    ]


    # Get Baseline
    baselines = {}
    with open(f"{cwd}/results/{no_noise_experiment_name}.jsonl","r") as f:
        for line in f:
            if not line.strip(): continue

            result = json.loads(line)
            # Get relevant hyperparameters
            config = result.get("config",{})
            epsilon = config.get("epsilon")
            num_clients = config.get("num-clients")
            
            if num_clients not in [150, 10]:
                continue

            # Extract server metrics
            rounds = config.get("num-server-rounds")
            flwr_str = result.get("flwr_results","")

            round_server_metrics = extract_metric_record(flwr_str, "ServerApp-side Evaluate Metrics")[rounds]
            
            accuracy_per_class, macro_recall = get_acc_per_class(round_server_metrics) 
            #runs["No Noise"] = accuracy_per_class
            baselines[f"No Noise {num_clients} Clients"] = accuracy_per_class
            print(f"No Noise ({num_clients} clients), macro recall: {macro_recall}")

    
    with open(f"{cwd}/results/{noise_experiment_name}.jsonl","r") as f:
        for line in f: 
            if not line.strip(): continue

            result = json.loads(line)
            # Get relevant hyperparameters
            config = result.get("config",{})
            epsilon = config.get("epsilon")
            num_clients = config.get("num-clients")
            prox_mu = config.get("prox-mu")

            if num_clients != 150:
                continue

            # Extract server metrics
            rounds = config.get("num-server-rounds")
            flwr_str = result.get("flwr_results","")

            round_server_metrics = extract_metric_record(flwr_str, "ServerApp-side Evaluate Metrics")[rounds]
            
            accuracy_per_class, macro_recall = get_acc_per_class(round_server_metrics)

            print(f"epsilon={epsilon}, prox-mu={prox_mu} ({num_clients} clients), macro recall: {macro_recall}")
            
            if prox_mu not in runs:
                runs[prox_mu] = {}
            runs[prox_mu][f"eps = {epsilon}"] = accuracy_per_class

    count = 0 
    for prox_mu, eps_acc_perclass_dict in runs.items():
        ## Ordered dict makes plot more aesthetically pleasing
        ordered_dict = {}
        ordered_dict["No Noise 10 Clients"]  = baselines["No Noise 10 Clients"]
        ordered_dict["No Noise 150 Clients"] = baselines["No Noise 150 Clients"]
        for k, v in eps_acc_perclass_dict.items():
            ordered_dict[k] = v
        generate_polar_plot(ordered_dict, categories, fig_name=f"{fig_name} prox-mu: {prox_mu}", save_to=f"figures/acc_per_class_{count}.png")
        count += 1

evaluate_result("5-baseline-for-noise-across-clients-adam","6-vary-noise-across-150-clients-adam-fedprox(1)",fig_name="Acc Per Class")
