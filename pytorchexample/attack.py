import os
import numpy as np
import math
import json
import torch
import random
import torch.nn.functional as F
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from pytorchexample.task import NN
from torch.autograd import grad

def soft_label_cross_entropy(pred, true):
    return torch.mean(torch.sum(- true * F.log_softmax(pred, dim=-1),1))

#def attack(origin_params, client_grad, input_shape, label_shape, num_classes, lr, rounds, reg_coeff, seed, cosine_similarity=False):
def attack(origin_params, client_grad, input_shape, label_shape, num_classes, max_iter, history_size, rounds, reg_coeff, seed, lr=1.0, cosine_similarity=False):
    """
    Deep Leakage from Gradients
    """
    device="cpu"
    model = NN()
    model.load_state_dict(origin_params)
    model.train()
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dummy_data = torch.randn(input_shape).to(device).requires_grad_(True)
    #dummy_data = torch.sigmoid(z)
    initial_data = dummy_data.detach().clone()
    

    batch_size = label_shape[0]
    # Add 10 (number of classes) for soft labels
    dummy_label = torch.randn(batch_size, 10).to(device).requires_grad_(True)

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr,max_iter=max_iter,history_size=history_size,line_search_fn="strong_wolfe")
   # optimizer = torch.optim.Adam([dummy_data,dummy_label],lr=lr)
   
    criterion = soft_label_cross_entropy 
    for i in range(rounds):
        #print("attacking")
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            target = F.softmax(dummy_label, dim=-1)
            assert dummy_pred.ndim == 2, dummy_pred.shape
            assert target.ndim == 2, target.shape
            assert dummy_pred.shape == target.shape, (dummy_pred.shape, target.shape)
            dummy_loss = criterion(dummy_pred, F.softmax(dummy_label,dim=-1))#F.softmax(dummy_label))
            dummy_grad = grad(dummy_loss, model.parameters(), create_graph=True)
            shapes = [t.shape for t in dummy_grad]

            if cosine_similarity:
                # Dot Product
                dot_prod = sum((d * c).sum() for d,c in zip(dummy_grad,client_grad))
                # L2-Norms
                dummy_norm_sq = sum(((d)**2).sum() for d in dummy_grad)
                client_norm_sq = sum(((c)**2).sum() for c in client_grad)

                # Cosine Similarity - Equation in torch docs: torch.nn.CosineSimilarity
                grad_diff = dot_prod / max(dummy_norm_sq*client_norm_sq, 1e-8)
            else:
                grad_diff = sum(((d - c) ** 2).sum() \
                        for d, c in zip(dummy_grad, client_grad))
            reg = reg_coeff * (dummy_data**2).mean()
            loss = grad_diff + reg
            loss.backward()
            return loss 
        optimizer.step(closure)

    return dummy_data.detach(), dummy_label.detach(), initial_data



#def fed_avg_attack(origin_params, num_training_rounds, weight_at_timestamp, gradient_at_timestamp, input_shape, label_shape, lr, attack_rounds, reg_coeff, seed):
def fed_avg_attack(origin_params, num_training_rounds, weight_at_timestamp, gradient_at_timestamp, input_shape, label_shape, attack_rounds, max_iter, history_size, reg_coeff, seed, lr=1.0):
    """
    From the paper: Improved Gradient Inversion Attacks and Defenses in Federated Learning.
    - The paper applies the attacks to multiple local training rounds where the server has access to intermediate weight and gradient updates, however we apply it to the federated learning process as a whole.
    """
    device="cpu"
    #model = NN()
    #model.load_state_dict(origin_params)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Initialize Dummy Image
    dummy_data = torch.randn(input_shape).to(device).requires_grad_(True)
    initial_data = dummy_data.detach().clone()
    ##TODO: Recover Labels using Zero-shot approach
    batch_size = label_shape[0]
    dummy_label = torch.randn(batch_size, 10).to(device).requires_grad_(True)

    #optimizer = torch.optim.Adam([dummy_data, dummy_label],lr=lr)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr,max_iter=max_iter,history_size=history_size,line_search_fn="strong_wolfe")
    criterion = soft_label_cross_entropy 
    for i in range(attack_rounds):
        def closure():
            optimizer.zero_grad()
            grad_diffs = []
            for t in range(num_training_rounds):
                local_model = NN().to(device)
                local_model.load_state_dict(weight_at_timestamp[t])
                dummy_pred = local_model(dummy_data)
                dummy_loss = criterion(dummy_pred, F.softmax(dummy_label,dim=-1))
                dummy_grad = grad(dummy_loss, local_model.parameters(), create_graph=True)
                grad_diff = sum(((dummy_grad - client_gradient_at_t) ** 2).sum() \
                        for dummy_grad, client_gradient_at_t in zip(dummy_grad, gradient_at_timestamp[t]))
                grad_diffs.append(grad_diff)

            diff = sum(grad_diffs) / num_training_rounds 
            reg = reg_coeff * (dummy_data**2).mean()
            loss = diff + reg
            loss.backward()
            return loss 
        optimizer.step(closure)
    return dummy_data.detach(), dummy_label.detach(), initial_data



def evaluate_inversion(X, recovered_X, y, recovered_y, initial_dummy_data):
    """
    TODO: Consider per-window recovery statistics.
            - For each window in the original data, is
              there a window in the recovered data that 
              is similar
            - Max similarity metric?
    """
    # MSE for training data
    X_diff = ((recovered_X - X) ** 2).mean().item()
    # PCC between original and recovered
    X_pcc = torch.corrcoef(torch.stack((X.flatten(),recovered_X.flatten())))[0,1].item()


    #cwd = os.getcwd()
    with open("inversion_debugging.txt","w") as f:
        f.write("Dummy Data")
        f.write("\n\n\n")
        f.write(str(recovered_X))
        f.write("\n\n\n")
        f.write("Actual Data")
        f.write("\n\n\n")
        f.write(str(X))
        f.write("\n\n\n")
        f.write("Initial Dummy Data")
        f.write("\n\n\n")
        f.write(str(initial_dummy_data))
        f.write("\n\n\n")


    return X_diff, X_pcc



def get_client_grad(trained_params, origin_params, lr, timesteps):
    """
    Gradient estimation strategy in: Improved Gradient Inversion Attacks and Defenses in Federated Learning.
    """
    #print("\n\n\ngetting client gradients....")
    client_grad: dict[str, NDArray] = {}
    shape = {}

    #print(f"\n\nlearning rate: {lr}\n\n")
    for key, value in origin_params.items():
        client_grad[key] = value.numpy() 

    client_grad_array = []

    for record_item in trained_params:
        for key, value in record_item.items():
            if "running_mean" in key or "running_var" in key:
                continue
            g = torch.from_numpy((client_grad[key] - value.numpy())/(lr*timesteps))
            
            ## TODO: REMOVE THIS -- DEBUGGING
            client_grad[key] = (client_grad[key] - value.numpy())/lr
            shape[key] = client_grad[key].shape

            client_grad_array.append(g)

    #print("writing to file ...\n\n\n")
    #with open('client_grad.txt','w') as f:
    #    print("writing to file ...\n\n\n")
    #    f.write(str(client_grad))
    #    f.write("\n\n\n")
    #    f.write("SHAPES:")
    #    f.write("\n\n\n")
    #    f.write(str(shape))

    return client_grad_array



