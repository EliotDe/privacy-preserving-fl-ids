import torch
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

#experiment_cfg = get_and_parse_config_yaml()


#local_dp_obj = LocalDpMod(clipping_norm=1, sensitivity=0.3, epsilon=5.0, delta=0.0001)
local_dp_obj = DynamicDpMod()

@app.train(mods=[local_dp_obj])
def train(msg: Message, context: Context):
    """Train the model on local data."""
    #global experiment_cfg

    # Load the model and initialize it with the received weights
    model = NN()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ## -------- Load the config and data----------##

    ## NOTE: This is purely for experimental purposes; so that we can use a training function with more control on the local data being used -- in actuality the client would not know an attack is being run by the server
    ## NOTE: I'm not sure the eval function is safe since it executes arbitrary inputs -- since its only used for local experimentation this is ok but it should not be used in production
    running_inversion = context.run_config["run-inversion"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)


    # Call the training function
    if running_inversion:
        training_context = LocalTrainingContext(LocalTrainingWithInversion())
        return training_context.train(msg, context, model, trainloader, device)
    else:
        training_context = LocalTrainingContext(LocalTrainingNormal())
        return training_context.train(msg, context, model, trainloader, device)

   
## NOTE: This is the old method, keep it just in case
def old_train(msg: Message, context: Context):
    """Train the model on local data."""
    global experiment_cfg

    # Load the model and initialize it with the received weights
    model = NN()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Track original model to calculate the l2-norm for the model update
    original_model_ndarrays = next(iter(msg.content.array_records.values())).to_numpy_ndarrays()

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Call the training function
    if experiment_cfg["inversion"]["run_inversion_attack"] is True:
        train_metadata = inv_train_fn(
           model,
           trainloader,
           int(cfg["num_local_batches"]),
           context.run_config["local-epochs"],
           device=device,
           lr=0.001,
        )
    else:
        train_loss = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            msg.content["config"]["weight_decay"],
            device,
        )


    # Construct and return reply Message

    state_dict = {
            k: v for k,v in model.state_dict().items()
            if "num_batches_tracked" not in k
    }

    model_record = ArrayRecord(state_dict)

    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})


    # Track trained model to calculate the l2-norm for the model update
    trained_model_ndarrays = next(iter(content.array_records.values())).to_numpy_ndarrays()

    # Calculate Model Update and l2-norm
    model_update = [np.subtract(x,y) for (x,y) in zip(trained_model_ndarrays, original_model_ndarrays, strict=True)]
    norms = [np.linalg.norm(array.flat) for array in model_update] 
    norm = float(np.sqrt(sum([norm**2 for norm in norms])))
    print(f"L2-Norm of Client update: {norm}")

    return Message(content=content, reply_to=msg)


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
    _, valloader = load_data(partition_id, num_partitions, batch_size)

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
