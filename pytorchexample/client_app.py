"""

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
from pytorchexample.task import get_and_parse_config_yaml
from pytorchexample.train_process_metadata import TrainProcessMetadata
from pytorchexample.local_train import LocalTrainingContext, LocalTrainingNormal, LocalTrainingWithInversion
from pytorchexample.dynamic_dp_mod import DynamicDpMod

# Flower ClientApp
app = ClientApp()



#local_dp_obj = DynamicDpMod()

@app.train()#mods=[local_dp_obj])
def train(msg: Message, context: Context):
    """Train the model on local data."""
    ## -------- Load the config and data----------##

    ## NOTE: This is purely for experimental purposes; so that we can use a training function with more control on the local data being used -- in actuality the client would not know an attack is being run by the server
    ## NOTE: I'm not sure the eval function is safe since it executes arbitrary inputs -- since its only used for local experimentation this is ok but it should not be used in production
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
    running_inversion = context.run_config["run-inversion"]
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
