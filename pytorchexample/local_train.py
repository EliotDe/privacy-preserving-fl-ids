"""
This contains the ClientApp train methods for normal training and for the inversion attack experiments.
"""
import pickle
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord
from abc import ABC, abstractmethod
from dataclasses import asdict
from pytorchexample.task import NN, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn
from pytorchexample.task import inversion_train as inv_train_fn
from pytorchexample.train_process_metadata import TrainProcessMetadata

class LocalTrainingStrategy(ABC):
    @abstractmethod
    def train(self, msg: Message, context: Context, model: NN, trainloader, device):
        pass

class LocalTrainingNormal(LocalTrainingStrategy):

    def train(self, msg: Message, context: Context, model: NN, trainloader, device): 
        train_loss = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            msg.content["config"]["weight_decay"],
            device,
        )

        # Only do this if using noise
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


        # Calculate Model Update and l2-norm
        trained_model_ndarrays = next(iter(content.array_records.values())).to_numpy_ndarrays()
        original_model_ndarrays = next(iter(msg.content.array_records.values())).to_numpy_ndarrays()
        model_update = [np.subtract(x,y) for (x,y) in zip(trained_model_ndarrays, original_model_ndarrays, strict=True)]
        norms = [np.linalg.norm(array.flat) for array in model_update] 
        norm = float(np.sqrt(sum([norm**2 for norm in norms])))
        print(f"L2-Norm of Client update: {norm}")

        return Message(content=content, reply_to=msg)



class LocalTrainingWithInversion(LocalTrainingStrategy):
    def train(self, msg: Message, context: Context, model: NN, trainloader, device):
        num_local_batches = int(context.run_config["local_batches"])
        train_metadata = inv_train_fn(
           model,
           trainloader,
           num_local_batches,
           context.run_config["local-epochs"],
           lr=msg.content["config"]["lr"],
           device=device,
        )
        
        train_loss = asdict(train_metadata)["train_loss"]
        train_meta_bytes = pickle.dumps(train_metadata)
        config_record = ConfigRecord({"meta": train_meta_bytes})

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

        content = RecordDict({
            "arrays": model_record, 
            "metrics": metric_record,
            "train_metadata": config_record
        })

        return Message(content=content, reply_to=msg) 


class LocalTrainingContext():
    
    def __init__(self, training_strategy: LocalTrainingStrategy) -> None:
        self._strategy: LocalTrainingStrategy = training_strategy

    @property
    def training_strategy(self) -> LocalTrainingStrategy:
        return self._strategy

    @training_strategy.setter
    def training_strategy(self, strategy: LocalTrainingStrategy) -> None:
        self._strategy = strategy

    def train(self, msg: Message, context: Context, model: NN, trainloader, device):
        return self._strategy.train(msg=msg, context=context, model=model, trainloader=trainloader, device=device)


   


