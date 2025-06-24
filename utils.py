from typing import Tuple, Union, List
from collections import OrderedDict
import numpy as np
import config_FL
import saveCSV
import torch
import torch.nn as nn
import decimal
import copy

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# def set_parameters(net, parameters: List[np.ndarray]):
#     params_dict = zip(net.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#     net.load_state_dict(state_dict, strict=True)

def set_parameters(net, parameters: List[np.ndarray]): # to set the global parameters on model
    if len(parameters) == 0:
        print("Error: Empty parameters list")
        return
    
    if len(net.state_dict().keys()) != len(parameters):
        print("Error: Mismatch between number of parameters and keys in the state dictionary")
        return
    
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    #print("State dictionary:", state_dict)  # Debugging statement
    net.load_state_dict(state_dict, strict=False)


def train(net, trainloader, server_round, epochs: int): #to train the iid data on model
    """Train the network on the training set."""
    """FedProx: https://github.com/adap/flower/blob/67bbd4d32a1fbce7d610c7e056eab20ac3bf319e/src/py/flwr/server/strategy/fedprox.py"""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config_FL.get_local_lr()) #,weight_decay=0.001)#,weight_decay=0.1
    net.train()
    DEVICE = torch.device(config_FL.get_cudaid() if torch.cuda.is_available() else "cpu")
    net=net.to(DEVICE)

    if config_FL.get_distribution_type() == "non_iid":
        proximal_mu = config_FL.get_mu_value()   # mu = 0 is equivalent to FedAvg
        global_params = copy.deepcopy(net).parameters()
    else:
        pass # iid, use regular training


    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)

            if config_FL.get_distribution_type() == "non_iid":
                proximal_term = 0.0
                for local_weights, global_weights in zip(net.parameters(), global_params):
                    proximal_term += (local_weights - global_weights).norm(2)
                loss = criterion(outputs, labels) + (proximal_mu / 2) * proximal_term
            elif config_FL.get_distribution_type() == "iid":
                loss = criterion(outputs, labels)
            else:
                raise ValueError("utils.train() unsupported distribution type: "+ config_FL.get_distribution_type())

            #loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0) # TODO: Xin: clip grad for FedProx as well?
            optimizer.step()
            # Metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    path = 'eval_metrics/fl_enc/Avg_Accuracy'
    saveCSV.save(path, epoch_acc,'global Accuracy', server_round, 'Server_Round')


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    
    DEVICE = torch.device(config_FL.get_cudaid() if torch.cuda.is_available() else "cpu")
    
    net.to(DEVICE)
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    
    return loss, accuracy


def print_parameters(model):
    """Print the parameters of the given PyTorch model."""
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(param.data)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def delete_zero_elements(matrix):
    nonzero_indices = np.nonzero(matrix)
    result = np.delete(matrix, nonzero_indices[0], axis=0)  # Delete rows with zero elements
    result = np.delete(result, nonzero_indices[1], axis=1)  # Delete columns with zero elements
    return result

def keep_nonzero_elements(matrix):
    # Find indices of nonzero elements
    nonzero_indices = np.nonzero(matrix)

    # Extract nonzero elements using fancy indexing
    nonzero_elements = matrix[nonzero_indices]

    return nonzero_elements

# def round_elements_nested_str(nested_list, decimals):
#     def process_element(value):
#         if isinstance(value, (np.ndarray, list)):
#             return [process_element(subvalue) for subvalue in value]
#         else:
#             value_str = f"{float(value):.{decimals}f}"
#             return float(value_str)

#     return [process_element(sublist) for sublist in nested_list]

def check_element_types(params_list):
    types_list = []
    for param in params_list:
        param_types = [type(num) for num in param]
        types_list.append(param_types)
        print(f"Types in current param array: {param_types}")
    return types_list

# def round_decimal(number, decimals=2):
#     try:
#         # Convert to a string with enough precision to avoid scientific notation issues
#         decimal_number = decimal.Decimal(f"{number:.10e}")
#         # Return the rounded float value
#         return float(decimal_number.quantize(decimal.Decimal('1.' + '0' * decimals), rounding=decimal.ROUND_HALF_UP))
#     except (decimal.InvalidOperation, ValueError):
#         # Handle invalid operations or values gracefully
#         print(f"Invalid number encountered: {number}")
#         return number
# def round_large_number(number, decimals=2):
#     try:
#         # Use string formatting to round the number
#         format_string = f"{{:.{decimals}f}}"
#         return float(format_string.format(number))
#     except (ValueError, TypeError) as e:
#         # Handle invalid operations or values gracefully
#         print(f"Invalid number encountered: {number} with error {e}")
#         return number

# Function to round a single number using scientific notation for large numbers
def round_large_number(number, decimals=2):
    try:
        # Check if the number is extremely large or small
        if abs(number) > 1e10 or abs(number) < 1e-10:
            # Use scientific notation for the calculation but convert back to float for display
            formatted = f"{number:.{decimals}e}"
            return float(formatted)
        else:
            # Use regular formatting for normal-sized numbers
            return round(number, decimals)
    except (ValueError, TypeError) as e:
        # Handle invalid operations or values gracefully
        print(f"Invalid number encountered: {number} with error {e}")
        return number 
def round_and_format(number, decimals=2):
    try:
        if abs(number) > 1e10 or abs(number) < 1e-10:
            # Use scientific notation for rounding
            formatted = f"{number:.{decimals}e}"
            # Convert to float to remove trailing zeros
            rounded_number = float(formatted)
            # Convert back to string in fixed-point notation
            return f"{rounded_number:.{decimals}f}"
        else:
            # Use regular rounding and formatting
            return f"{round(number, decimals):.{decimals}f}"
    except (ValueError, TypeError) as e:
        print(f"Invalid number encountered: {number} with error {e}")
        return str(number)


def check_magnitude(params_list):
    for param in params_list:
        for num in param:
            pass
            #print(f"Number: {num}, Magnitude: {abs(num)}")

def get_first_three_digits(number):
    # Convert to scientific notation
    sci_str = f"{number:.2e}"  # Using .2e to get scientific notation with 2 decimal places
    # Split into mantissa and exponent
    mantissa, exponent = sci_str.split('e')
    # Extract the first three significant digits from the mantissa
    first_three_digits = mantissa.replace('.', '')[:3]
    # Combine the first three digits with the exponent
    new_sci_str = f"{first_three_digits}e{exponent}"
    # Convert back to float
    new_number = float(new_sci_str)
    return new_number

def print_local_params(local_params): # to print local [parameters upto 20 values]
    for layer_id in range(len(local_params)):
        if not isinstance(local_params[layer_id][0], np.ndarray):     
            # bias
            print(local_params[layer_id][:20])
        else:
            # weight:
            if layer_id == 0:
                print(local_params[layer_id][0][:20])

            else:
                for i in range(3):
                    print(local_params[layer_id][i][:20])
