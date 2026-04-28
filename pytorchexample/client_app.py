"""
Instantiate the ClientApp app object and use decorators to "upload" train and evaluation functions
"""

import torch
import copy
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.clientapp.mod import LocalDpMod
from pytorchexample.task import NN, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn
from pytorchexample.task import inversion_train as inv_train_fn
from pytorchexample.train_process_metadata import TrainProcessMetadata
from pytorchexample.local_train import LocalTrainingContext, LocalTrainingNormal, LocalTrainingWithInversion
from pytorchexample.dynamic_dp_mod import DynamicDpMod

# Flower ClientApp
app = ClientApp()


## TODO: If you are running differential privacy experiments uncomment the dynamic dp modifier
## TODO: If you aren't running with DP, comment them out.
#local_dp_obj = DynamicDpMod()

## TODO: If you are running the differential privacy experiments uncomment the modifier
## TODO: If you aren't running with DP comment the modifier out.
@app.train()#mods=[local_dp_obj])
def train(msg: Message, context: Context):
    """Train the model on local data."""
    ## -------- Load the config and data----------##
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    running_inversion = context.run_config["run-inversion"]
    batch_size = context.run_config["batch-size"]
    train_shuffle = context.run_config["shuffle"]
    window_size = context.run_config["window-size"]
    seed = context.run_config["seed"]
    prox_mu = context.run_config["prox-mu"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size, window_size, train_shuffle, seed)

    # Load the model and initialize it with the received weights
    model = NN()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # The regularizing term in Prox-Mu requires you to calculate the distance between the original global
    # parameters and the learned parameters.
    if prox_mu != 0:
        global_params = copy.deepcopy(model).parameters()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Call the training function
    if running_inversion:
        training_context = LocalTrainingContext(LocalTrainingWithInversion())
        if prox_mu != 0:
            return training_context.train(msg, context, model, trainloader, device, prox_mu, global_params)
        else:
            return training_context.train(msg, context, model, trainloader, device)
    else:
        training_context = LocalTrainingContext(LocalTrainingNormal())
        if prox_mu != 0:
            return training_context.train(msg, context, model, trainloader, device, prox_mu, global_params)
        else:
            return training_context.train(msg, context, model, trainloader, device)
   

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    # Load the model and initialize it with the received weights
    model = NN()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    window_size = context.run_config["window-size"]
    seed = context.run_config["seed"]

    _, valloader = load_data(partition_id, num_partitions, batch_size, window_size, seed=seed)

    # Call the evaluation function
    eval_loss, eval_acc, _, _ = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
