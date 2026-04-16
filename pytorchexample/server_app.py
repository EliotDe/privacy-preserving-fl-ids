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

NUM_FEATURES=34
WINDOW_SIZE=20
CWD = os.getcwd()
RESULTS_PATH=f"{CWD}/results"
ALL_RESULTS_PATH=f"{RESULTS_PATH}/all_results.jsonl"

# Create ServerApp
app = ServerApp()

#experiment_cfg = get_and_parse_config_yaml()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
#    global experiment_cfg

    # Read run config
    ## NOTE: I'm not sure the eval function is safe since it executes arbitrary inputs -- Since this is only intented for experimentation its ok but it shouldn't be used in production
    run_inversion_attack:  bool = context.run_config["run-inversion"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    weight_decay: float = context.run_config["weight-decay"]
    batch_size: int = context.run_config["batch-size"]

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
        strategy = CustomFedAvg(fraction_evaluate=fraction_evaluate)

        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr, "weight_decay":weight_decay}),
            num_rounds=num_rounds,
            evaluate_fn=global_evaluate,
        )
        save_result(result,context)
    else:
        strategy = FedAvg(fraction_evaluate=fraction_evaluate)

        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr, "weight_decay":weight_decay}),
            num_rounds=num_rounds,
            evaluate_fn=global_evaluate,
        )
        save_result(result, context)

    

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = NN()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

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


def get_accuracy_per_class(y_pred,y_true,labels):
    class_dict = {}
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    accuracies = cm.diagonal()

    for i,label in enumerate(labels):
        class_dict[f"acc_{label}"] = accuracies[i]

    return class_dict 



def save_result(results: Result, context: Context):
    res_str = str(results).replace("\n","").replace("\t"," ")
    config = context.run_config
    record = {
            "config": config,
            "flwr_results": res_str
    }
    
    with open(ALL_RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


