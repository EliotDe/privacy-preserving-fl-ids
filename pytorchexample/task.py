import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
#from flwr_datasets.partitioner import IidPartitioner
from .temporal_partitioner import TemporalPartitioner
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from pytorchexample.train_process_metadata import TrainProcessMetadata
from pytorchexample.data_utils import set_windows
#from torchvision.transforms import Compose, Normalize, ToTensor

THRESHOLD_FOR_UNRELATED_S = 600   # At what difference in seconds do we consider two records temporally unrelated

class CustomDataset(Dataset):
    def __init__(self,inputs,labels,window_size):
        self.labels = torch.tensor(labels,dtype=torch.long)
        self.inputs = torch.tensor(inputs.values, dtype=torch.float32)
        self.window: int = window_size

    def __len__(self) -> int:
        return len(self.inputs) // self.window

    def __getitem__(self,idx):
        x = self.inputs[idx*self.window: (idx+1)*self.window].T
        y = self.labels[idx*self.window]  # All types in a window are consistent
        return x,y



class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=34, out_channels=64, kernel_size=5)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=2)

        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
       # x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.bn2(self.conv2(x)))
       # x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
       # x = self.relu(self.conv3(x))
        x = self.pool(x)

        x = self.adaptive_pool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


fds = None  # Cache FederatedDataset
partitioner = None
scaler = StandardScaler()
le = LabelEncoder()
fitted = False

def fit_global_transforms(dataset):
    df = dataset.to_pandas()
    #print(f"====== FIT GLOBAL TRANSFORMS =====\n\t ----> df.head(): \n {df.head}")
    labels = df["type"]
    features = df.drop(columns=["type","ts","label"])#,"window_id","delta_t"])
    scaler.fit(features)
    le.fit(labels)

def is_feature(feat) -> bool:
    return feat != "type" and feat != "ts" and feat!="label"

def apply_transforms(batch):
    labels = batch["type"]
    features = {k: v for k, v in batch.items() if is_feature(k)}

    X = np.column_stack([features[k] for k in features])
    X = scaler.transform(X)

    y = le.transform(labels)

    batch["x"] = torch.tensor(X,dtype=torch.float32)
    batch["y"] = torch.tensor(y,dtype=torch.long)

    return batch


def get_class_names_from_labels(labels):
    return le.inverse_transform(labels)

def get_class_names():
    return le.classes_


    
## TODO: This needs to be refined
def labeling_rule(window_types):
    type_set = set(window_types)
    if len(type_set) == 1 and "normal" in type_set:
        return "normal"
    else:
        for t in type_set:
            if t != "normal":
                return t

     
def load_data(partition_id: int, num_partitions: int, batch_size: int, window_size: int, seed: int, shuffle_train=True):
    # Only initialize `FederatedDataset` once
    global fds,partitioner,fitted
    if fds is None:
        partitioner = TemporalPartitioner(
                num_partitions=num_partitions,
                partition_by='type',
                window_size=window_size,
                alpha=[1]*num_partitions,
                threshold_for_unrelated_s = THRESHOLD_FOR_UNRELATED_S,
                time_feature_name = 'ts',
                seed=seed
        )
        data_files = "dataset/processed_network.csv" 
        dataset = load_dataset("csv", data_files=data_files, split="train")
        ## TODO: REMOVE THIS!!!
    #    check_nan_dataset = dataset.to_pandas()
        #print(f"NaNs in dataset before scaling: {check_nan_dataset.isna().sum().sum()}")
        #print(f"NaNs per column:\n{check_nan_dataset.isna().sum()}")

        #dataset = dataset.class_encode_column("type")
        partitioner.dataset = dataset

        if not fitted:
            fit_global_transforms(dataset)
            fitted = True

        fds = dataset

    partition = partitioner.load_partition(partition_id)#fds.load_partition(partition_id)
    partition_df = partition.to_pandas()
    #print(f"NaNs in partition before scaling: {partition_df.isna().sum().sum()}")
    #print(f"NaNs per column:\n{partition_df.isna().sum()}")
    #print(f"Inf values: {np.isinf(partition_df.values).sum()}")
    #cwd = os.getcwd()
    #if not os.path.isfile(f"{cwd}/dataset/{partition_id}.csv"):
    #    partition_df.to_csv(f"{cwd}/dataset/{partition_id}.csv")

    p_window_types = partition_df.groupby("window_id")["type"].agg(labeling_rule).reset_index()
    unique_classes, class_counts = np.unique(p_window_types['type'],return_counts=True)

    if class_counts.min() < 2:
        train_windows, test_windows = train_test_split(p_window_types['window_id'],random_state=seed)
    else:
        train_windows, test_windows = train_test_split(p_window_types['window_id'], stratify=p_window_types['type'],random_state=seed)


    train_df = partition_df[partition_df['window_id'].isin(train_windows)]
    test_df = partition_df[partition_df['window_id'].isin(test_windows)]

    
    X_train = train_df.drop(columns=["label","ts","window_id","delta_t","type"])
    X_test = test_df.drop(columns=["label","ts","window_id", "delta_t", "type"])
    y_train = train_df["type"]
    y_test = test_df["type"]

    feature_cols = X_train.columns.to_list()
    #print(f"NaNs in X_train before scaling: {X_train.isna().sum().sum()}")
    #print(f"NaNs per column:\n{X_train.isna().sum()}")
    #print(f"Inf values: {np.isinf(X_train.values).sum()}")
    X_train = scaler.transform(X_train)
    #print(f"X_train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    #print(f"X_train min: {X_train.min():.4f}, max: {X_train.max():.4f}")
    X_test = scaler.transform(X_test)

    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    train_ds = CustomDataset(pd.DataFrame(X_train,columns=feature_cols), y_train, window_size=window_size)
    test_ds = CustomDataset(pd.DataFrame(X_test,columns=feature_cols), y_test, window_size=window_size)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    testloader = DataLoader(test_ds, batch_size=batch_size)

    return trainloader,testloader


def load_centralized_dataset(window_size: int):
    """Load test set and return dataloader."""
    global fitted

    data_files = "dataset/processed_network.csv"

    dataset = load_dataset("csv", data_files=data_files, split="train")
    #dataset = set_windows(dataset, window_size, THRESHOLD_FOR_UNRELATED_S, 'ts')

    if not fitted:
        fit_global_transforms(dataset)
        fitted=True

    
    dataset = set_windows(dataset, window_size, THRESHOLD_FOR_UNRELATED_S, 'ts')
    dataset = dataset.to_pandas()
    
    df_window_types = dataset.groupby("window_id")["type"].agg(labeling_rule).reset_index()
    _, test_windows = train_test_split(df_window_types['window_id'], stratify=df_window_types['type'])

    test_df = dataset[dataset['window_id'].isin(test_windows)]
    test_df = test_df.drop(columns=["label","ts","window_id","delta_t"])
    X_test = test_df.drop(columns=["type"])
    X_test = scaler.transform(X_test)
    y_test = test_df["type"]
    y_test = le.transform(y_test)
    
    ds = CustomDataset(pd.DataFrame(X_test),y_test,window_size=window_size)

    return DataLoader(ds, batch_size=32,shuffle=False)


def train(net, trainloader, epochs, lr, weight_decay, device, prox_mu=0, global_params=None):
    """
    Train the model on the training set.

    - prox_mu: This is used in the FedProx aggregation scheme. By default it is 0. When prox_mu=0 the aggregation scheme is FedAvg.
    """
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=weight_decay)
    net.train()

    if global_params is not None:
        global_params = [p.detach().to(device) for p in global_params]

    for i in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs,labels in trainloader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs,labels)
            # For FedProx
            if global_params is not None and prox_mu != 0:
                proximal_term = 0.0
                for local_weights, global_weights in zip(net.parameters(),global_params):
                    proximal_term += (local_weights - global_weights).norm(2)**2
                    loss += (prox_mu/2) * proximal_term
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _,predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
        print(f"epoch {i} loss: {running_loss/len(trainloader)} train_acc: {100*correct/total}") 
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def inversion_train(net, trainloader, num_batches, epochs, lr, device, prox_mu=0, global_params=None):
    """
    When running Inversion Attack experiments, more control is needed over 
    the data being trained on and the optimization. This method provides 
    more control over the number of local batches and epochs and the
    optimizer.

    - prox_mu: This is used in the FedProx aggregation scheme. By default it is 0. When prox_mu=0 the aggregation scheme is FedAvg.

    """
    #print(f"\n\ntraining learning rate: {lr}")
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.train()

    if global_params is not None:
        global_params = [p.detach().to(device) for p in global_params]
    
    all_inputs = []
    all_labels = []
    timestamps = 0
    for i in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_inputs = []
        epoch_labels = []
        loader_iter = iter(trainloader)
        for j in range(num_batches):
            # Get and track inputs and labels
            inputs,labels = next(loader_iter)
            inputs = inputs.to(device)
            epoch_inputs.append(inputs)
            labels = labels.to(device)
            epoch_labels.append(labels)

            # Train on batch
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels)

            # For FedProx if using
            if global_params is not None and prox_mu != 0:
                proximal_term = 0.0
                for local_weights, global_weights in zip(net.parameters(),global_params):
                    proximal_term += (local_weights - global_weights).norm(2)**2
                    loss += (prox_mu/2) * proximal_term

            loss.backward()
            optimizer.step()

            # Evaluate
            running_loss += loss.item()
            _,predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

            timestamps += 1

        print(f"epoch {i} loss: {running_loss/num_batches} train_acc: {100*correct/total}") 
        if i==0:
            all_inputs.append(torch.cat(epoch_inputs,0))
            all_labels.append(torch.cat(epoch_labels,0))
    avg_trainloss = running_loss / num_batches 
    X = torch.cat(all_inputs,0)
    y = torch.cat(all_labels,0)
    meta = TrainProcessMetadata(
            X_shape=X.shape,
            y_shape=y.shape,
            X=X.detach().cpu(),
            y=y.detach().cpu(),
            train_loss=avg_trainloss,
            timestamps=timestamps
    )    
    return meta 




## TODO: Add accuracy per class
def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    all_labels = []
    all_predictions = []
    net.eval()
    with torch.no_grad():
        for inputs,labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            all_labels.append(labels)
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            predictions = torch.max(outputs.data, 1)[1]
            correct += (predictions == labels).sum().item()
            all_predictions.append(predictions)
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    return loss, accuracy, predictions, labels


def get_and_parse_config_yaml():
    #print(f"\n\n\n\n{os.getcwd()}\n\n\n\n")
    experiment_cfg = {}
    with open('experiment_config.yaml','r') as f:
        experiment_cfg = yaml.full_load(f)
    return experiment_cfg


