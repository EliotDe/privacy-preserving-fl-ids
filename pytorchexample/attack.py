import numpy
import math
import json
import torch
import torch.nn.functional as F
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from pytorchexample.task import NN
from torch.autograd import grad

def soft_label_cross_entropy(pred, true):
    return torch.mean(torch.sum(- true * F.log_softmax(pred, dim=-1),1))

def attack(origin_params, client_grad, input_shape, label_shape, num_classes):
        """
        Deep Leakage from Gradients
        """
        device="cpu"
        model = NN()
        model.load_state_dict(origin_params)

        dummy_data = torch.randn(input_shape).to(device).requires_grad_(True)
        #dummy_label = torch.randint(0, num_classes, label_shape)
        dummy_label = torch.randn(label_shape).to(device).requires_grad_(True)
        optimizer = torch.optim.Adam([dummy_data, dummy_label])
        criterion = soft_label_cross_entropy 
        for i in range(500):
            def closure():
                optimizer.zero_grad()
                dummy_pred = model(dummy_data)
#                dummy_loss = torch.mean(torch.sum(- dummy_pred * F.log_softmax(F.softmax(dummy_label,dim=-1), dim=-1),1))
                dummy_loss = criterion(dummy_pred, dummy_label)#F.softmax(dummy_label))
                dummy_grad = grad(dummy_loss, model.parameters(), create_graph=True)
                shapes = [t.shape for t in dummy_grad]

                grad_diff = sum(((dummy_grad - client_grad) ** 2).sum() \
                        for dummy_grad, client_grad in zip(dummy_grad, client_grad))

                grad_diff.backward()
                return grad_diff
            optimizer.step(closure)

        return dummy_data, dummy_label

def evaluate_inversion(X, recovered_X, y, recovered_y):
    """
    TODO: Consider per-window recovery statistics.
    """
    # MSE for training data
    X_diff = sum(((recovered_X - X) ** 2).sum() for recovered_X, X in zip(recovered_X, X))
    # PSNR for training data
    #X_psnr = (20*(math.log(20,10))) - (10*(math.log(X_diff,10)))

    # PCC between original and recovered
    X_pcc = torch.corrcoef(torch.stack((X.flatten(),recovered_X.flatten())))[0,1]
    return X_diff, X_pcc



def get_client_grad(trained_params, origin_params, lr):
    print("\n\n\ngetting client gradients....")
    client_grad: dict[str, NDArray] = {}
    shape = {}

    for key, value in origin_params.items():
        client_grad[key] = value.numpy() 

    client_grad_array = []

    for record_item in trained_params:
        for key, value in record_item.items():
            if "running_mean" in key or "running_var" in key:
                continue
            g = torch.from_numpy((client_grad[key] - value.numpy())/lr)
            
            ## TODO: REMOVE THIS -- DEBUGGING
            client_grad[key] = (client_grad[key] - value.numpy())/lr
            shape[key] = client_grad[key].shape

            client_grad_array.append(g)

    print("writing to file ...\n\n\n")
    with open('client_grad.txt','w') as f:
        print("writing to file ...\n\n\n")
        f.write(str(client_grad))
        f.write("\n\n\n")
        f.write("SHAPES:")
        f.write("\n\n\n")
        f.write(str(shape))

    return client_grad_array



