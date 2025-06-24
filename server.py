import socket
from typing import Dict
from time import monotonic
import flwr as fl

import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision.datasets import CIFAR10, MNIST

import utils
import saveCSV
import config_FL
import fedavg_strategy
from models.lenet import lenet
from models.conv8 import conv8
from xmkckks_wrapper import Ciphertext

class ToNumpyArray(object):
    """Convert a image data to a numpy array."""
    def __call__(self, pic):
        return np.array(pic)

start_time = monotonic()
M = config_FL.num_client()

DEVICE = torch.device(config_FL.get_cudaid() if torch.cuda.is_available() else "cpu")  # Try "cuda" to train on GPU
input_size = 28 * 28           
hidden_size = 84
num_classes = 10
mask=None           
apply_mask=False        

if config_FL.get_masking_type() == "maser":
    # For continuous masking
    ROUND_MASK = 1 # Mask Generation when server_round % 3 = 1 
    ROUND_ENC = 2 # Encrypt Params when server_round % 3 = 2
    ROUND_DEC = 0 # Decrypt Params when server_round % 3 = 0
    ROUND_MOD = 3 # Modulus for continutous masking
elif config_FL.get_masking_type() == "grasp" or config_FL.get_masking_type() == "random":
    ROUND_MASK = 1 # Mask Generation when server_round = 1 
    ROUND_ENC = 0 # Encrypt Params when server_round % 2 = 0
    ROUND_DEC = 1 # Decrypt Params when server_round % 2 = 1 and server_round != 1
    ROUND_MOD = 2 # Modulus for continutous masking
else:
    raise ValueError("Unsupported masking type:", config_FL.get_masking_type())

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round,}  

def print_model_shapes(model):
    for name, param in model.named_parameters():
        print(f'Parameter: {name}, Shape: {param.shape}')

def zeros_like_model(model):
    zero_tensors = []
    for param in model.parameters():
        zero_tensors.append(torch.zeros_like(param))
    return zero_tensors

def copy_params_to_model(params_1d, model, zero_tensors):
    start_index = 0
    for param, zero_tensor in zip(model.parameters(), zero_tensors):
        numel = param.numel()
        param.data.copy_(torch.tensor(params_1d[start_index:start_index+numel]).reshape(param.shape))
        start_index += numel
    return model

def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""
    # Load test data here to avoid the overhead of doing it in `evaluate` itself 

    if config_FL.get_dataset_name()=="mnist":
        transform = transforms.Compose([
        
        # transforms.ToTensor(),  #image to tensor
        ToNumpyArray(),
        transforms.ToPILImage(), #tensor to image for the range adjustment
        transforms.ToTensor(),  #image to tensor to be normalized
        transforms.Normalize((0.1307,), (0.3081,))
    ])

        # Download CIFAR-10 testing dataset
        #testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

        testset = MNIST(root='./data', train=False, download=True, transform=transform)

        # DataLoader for batching and shuffling
        test_loader = DataLoader(testset, batch_size=128, shuffle=False, drop_last=True)
    elif config_FL.get_dataset_name()=="cifar10":
        transform = transforms.Compose([
        # transforms.ToTensor(),
        ToNumpyArray(),
        transforms.ToPILImage(), 
        transforms.ToTensor(),  
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))])
        testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

        # DataLoader for batching and shuffling
        test_loader = DataLoader(testset, batch_size=128, shuffle=False, drop_last=True)
        
    else:
        raise ValueError(f"Unsupported dataset: {config_FL.get_dataset_name()}")

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        if (config_FL.sys_mode()):
            if(config_FL.get_masking_type()=="maser"):
                if (server_round % ROUND_MOD==0 or server_round==0):

                    utils.set_parameters(model, parameters)
                    loss, accuracy = utils.test(model, test_loader)
                    path = f'eval_metrics/fl_enc/global_test_acc_{config_FL.get_dataset_name()}_{config_FL.get_distribution_type()}_{config_FL.get_masking_type()}-{config_FL.get_masking_threshold()}% clients-'
                    saveCSV.save(path, accuracy,'Test_Acc', int(server_round / 3) if server_round >= 3 else server_round, 'Server_Round')

                    return float(loss), {"Accuracy": accuracy}
            elif (config_FL.get_masking_type()=="grasp" or config_FL.get_masking_type()=="random" ):   
                if (server_round % ROUND_MOD==1 or server_round==0) and (server_round!=1):
                    utils.set_parameters(model, parameters)
                    loss, accuracy = utils.test(model, test_loader)
                    path = f'eval_metrics/fl_enc/global_test_acc_{config_FL.get_dataset_name()}_{config_FL.get_distribution_type()}_{config_FL.get_masking_type()}-{config_FL.get_masking_threshold()}% clients-'
                    saveCSV.save(path, accuracy,'Test_Acc', int(server_round / 2) if server_round > 0 else server_round, 'Server_Round')

                    return float(loss), {"Accuracy": accuracy}
            else:
                raise ValueError(f"Unsupported masking_type: {config_FL.get_masking_type()}")
           

        else:
            # utils.set_model_params(model, parameters)
            utils.set_parameters(model, parameters)
            roundTime = monotonic() - start_time
            pathRT = 'eval_metrics/fl/roundTime'
            saveCSV.save(pathRT, roundTime,'Time', server_round, 'num round')

            loss, accuracy = utils.test(model, test_loader)
            path = f'eval_metrics/fl/global_test_acc_clients-'
            saveCSV.save(path, accuracy,'Test_Acc',server_round, 'Server_Round')


            
            return float(loss), {"Accuracy": accuracy}

    return evaluate


# Start Flower server for 1000 rounds of federated learning
if __name__ == "__main__":
    if config_FL.get_model_name()=="lenet" and config_FL.get_dataset_name()=="mnist":
        model = lenet().to(DEVICE)  
    elif config_FL.get_model_name()=="conv8" and  config_FL.get_dataset_name()=="cifar10":
        model = conv8().to(DEVICE)
    else:
        raise ValueError(f"Unsupported model_name: {config_FL.get_model_name()}\n or the model and dataset pair is not supported")      #mask and applymask
    
    
    params = utils.get_parameters(model) 
    grpc_max_message_length: int =1074177918
    round_timeout: float=600
    strategy = fedavg_strategy.FedAvg(
            fraction_fit=1.0,
            round_timeout=600,
            min_fit_clients=M,
            min_available_clients=M,
            evaluate_fn=get_evaluate_fn(model),
            on_fit_config_fn=fit_round,
            initial_parameters=fl.common.ndarrays_to_parameters(params),
    )
    
    
    if(config_FL.sys_mode()):
        if(config_FL.get_masking_type()=='maser'):
            nb_round = ROUND_MOD*config_FL.nb_round()
        elif (config_FL.get_masking_type()=='grasp') or (config_FL.get_masking_type()=='random' ):
            nb_round = (ROUND_MOD*config_FL.nb_round())+1
        else:
            raise ValueError(f"Unsupported masking_type: {config_FL.get_masking_type()}")
    else:
        nb_round = config_FL.nb_round()
    ipv6_address = socket.getaddrinfo("localhost", None, socket.AF_INET6)[0][4][0]

# Construct the IPv6 address with port
    server_ip, server_port = config_FL.get_server_addr()
    server_address = server_ip + ":" +str(server_port)

    history = fl.server.start_server(
        server_address = server_address,
        grpc_max_message_length= 1974177918,
        #round_timeout=120,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=nb_round,round_timeout=120000.0,),
    )
    
    runtime = monotonic() - start_time
    
    if config_FL.sys_mode():
        path = 'eval_metrics/fl_enc/runtime'
    else:
        path = 'eval_metrics/fl/runtime'

    saveCSV.save(path, runtime,'Time', M, 'num client')
 