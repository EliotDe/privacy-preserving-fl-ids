"""pytorchexample: A Flower / PyTorch app."""

import json
import torch
import os
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, Result
from sklearn.metrics import confusion_matrix
from pytorchexample.task import NN, load_centralized_dataset, test, get_class_names_from_labels, get_class_names
from pytorchexample.task import get_and_parse_config_yaml
from pytorchexample.custom_fed_avg import CustomFedAvg
from pytorchexample.custom_fed_prox import CustomFedProx

NUM_FEATURES=34
WINDOW_SIZE=20
CWD = os.getcwd()
RESULTS_PATH=f"{CWD}/results"
ALL_RESULTS_PATH=f"{RESULTS_PATH}/all_results.jsonl"
ATTACK_TUNING_RESULTS_PATH = f"{RESULTS_PATH}/attack_tuning.jsonl"

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    ## NOTE: I'm not sure the eval function is safe since it executes arbitrary inputs -- Since this is only intented for experimentation its ok but it shouldn't be used in production
    run_inversion_attack:  bool = context.run_config["run-inversion"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    weight_decay: float = context.run_config["weight-decay"]
    batch_size: int = context.run_config["batch-size"]
    shuffle_train: bool = context.run_config["shuffle"]
    window_size: int = context.run_config["window-size"]
    prox_mu:float = context.run_config["prox-mu"]
    seed: int = context.run_config["seed"]


    # Load global model
    global_model = NN()
    state_dict = {
            k: v for k,v in global_model.state_dict().items() 
            if "num_batches_tracked" not in k
    }
    arrays = ArrayRecord(state_dict)

    # Initialize FedAvg strategy
    #print(f"\n\n\nrun inversion attack config type: {type(experiment_cfg["inversion"]["run_inversion_attack"])}\n\n\n")
    if run_inversion_attack:
        attack_lr = context.run_config["attack-lr"]
        attack_rounds = context.run_config["attack-rounds"]
        attack_reg = context.run_config["attack-reg"]
        if prox_mu != 0:
            strategy = CustomFedProx(proximal_mu=prox_mu, fraction_evaluate=fraction_evaluate)
        else:
            strategy = CustomFedAvg(fraction_evaluate=fraction_evaluate)

        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr, "weight_decay":weight_decay, "attack_lr": attack_lr, "attack_rounds": attack_rounds, "attack_reg": attack_reg, "seed": seed, "shuffle": shuffle_train}),
            num_rounds=num_rounds,
            evaluate_fn=get_evaluate_fn(window_size=window_size),
        )
        save_result(result,context)
    else:

        if prox_mu != 0:
            strategy = FedProx(proximal_mu=prox_mu, fraction_evaluate=fraction_evaluate)
        else:
            strategy = FedAvg(fraction_evaluate=fraction_evaluate)
        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr, "weight_decay":weight_decay}),
            num_rounds=num_rounds,
            evaluate_fn=get_evaluate_fn(window_size=window_size),
        )
        save_result(result, context)

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def get_evaluate_fn(window_size: int):
    """
    Flower doesn't let you pass parameters to the evaluate_fn when instantiating a strategy. Instead of modifying flowers in-built strategies (FedAvg for example) I thought it would be better to use a closure.
    """
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""

        # Load the model and initialize it with the received weights
        model = NN()
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load entire test set
        test_dataloader = load_centralized_dataset(window_size)

        # Evaluate the global model on the test set
        test_loss, test_acc, predictions, labels = test(model, test_dataloader, device)
        record = {"accuracy": test_acc, "loss": test_loss}

        # Recover class names and labels from label encoding
        class_names = get_class_names() 
        predictions = get_class_names_from_labels(predictions)
        labels = get_class_names_from_labels(labels)

        # Append accuracy per class to metric record
        acc_per_class = get_accuracy_per_class(predictions, labels, class_names)
        for k, v in acc_per_class.items():
            record[k] = v

        # Return the evaluation metrics
        return MetricRecord(record)
    return global_evaluate


def get_accuracy_per_class(y_pred,y_true,labels):
    class_dict = {}
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    accuracies = cm.diagonal()

    for i,label in enumerate(labels):
        class_dict[f"acc_{label}"] = accuracies[i]

    return class_dict 



def save_result(results: Result, context: Context):
    res_str = str(results).replace("\n","").replace("\t"," ")
    experiment_name: str = context.run_config["experiment-name"]
    config = context.run_config
    record = {
            "config": config,
            "flwr_results": res_str
    }
    
    with open(f"{RESULTS_PATH}/{experiment_name}.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")



