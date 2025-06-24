import gc
import copy 
import pickle
import socket
import json
import struct
import argparse
import numpy as np
import time
from time import monotonic
from pympler import asizeof

import flwr as fl

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from torchvision import transforms

import utils
import saveCSV
import config_FL
from models.lenet import lenet
from models.conv8 import conv8

from masking_techniques.maser import maser
from masking_techniques.grasp import grasp
from masking_techniques.random import random_mask
from xmkckks_wrapper import *

if __name__ == "__main__":
    #NOTE Parse the argument beforehand to use client_id
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, config_FL.num_client()), required=True)
    args = parser.parse_args()
    client_id = args.partition + 1 # client_id starts from 1, 0 is preserved for server/key manager

    # Perform public key aggregation if using HE
    if(config_FL.sys_mode()):

        client = MPHEServer(server_id = client_id) # Initialize client HE instance

        #key generation and communication with key_manager starts
        def send_public_key_to_key_manager(public_key, host, port): 
            # NOTE This function is for sending keys for aggregation to Key_manager()
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as key_manager_socket:
                    key_manager_socket.connect((host, port))

                    # Convert public_key to pickle-encoded bytes
                    serialized_data = pickle.dumps(public_key, protocol=2)

                    # Send the length of the data as a 4-byte integer
                    key_manager_socket.sendall(struct.pack('>I', len(serialized_data)))

                    # Send the actual pickle data
                    key_manager_socket.sendall(serialized_data)

                print("Public key sent successfully to Key_manager.")
                time.sleep(3)
            except socket.error as e:
                print(f"Socket error while sending public key: {e}")
            except Exception as e:
                print(f"Error while sending public key: {e}")

        # id for FL client starts from 1, can be any unique non-0 integer

        client_public_key = client.pk[0].P  # Class type
        client_key_size = asizeof.asizeof(client_public_key)
        #print(f"Size of the public key polynomial: {client_key_size} bytes")
        km_ip, km_port_send, km_port_recv = config_FL.get_key_manager_addr()
        send_public_key_to_key_manager(client_public_key, km_ip, km_port_send)

        time.sleep(5)  # NOTE this is mandatory to keep this sleep for parallel processing for 10 c_is 2, 50 c_is- 5, 100 c_is-10

        client_soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_soc.connect((km_ip, km_port_recv))

        # NOTE Usual socket receive (See Key_manager)
        def recvall(sock, n):
            # Helper function to recv n bytes or return None if EOF is hit
            data = bytearray()
            while len(data) < n:
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            return data

        def recv_msg(sock):
            # Read message length and unpack it into an integer
            raw_msglen = recvall(sock, 4)
            if not raw_msglen:
                return None
            msglen = struct.unpack('>I', raw_msglen)[0]
            # print(msglen)
            # Read the message data
            return recvall(sock, msglen)

        params = pickle.loads(recv_msg(client_soc))
        client.pk[0].P = copy.deepcopy(params)
        print('Aggregated Public key received from Key Manager.')

    time.sleep(10)
    # End of communication with key_manager

    DEVICE= torch.device(config_FL.get_cudaid() if torch.cuda.is_available() else "cpu") # getting the device for the model operations
    
    # Load dataset and select distribution type
    if config_FL.get_distribution_type()=="iid":
        subset_data = np.load('data_distribution/' + str(config_FL.get_dataset_name()) + '/' + config_FL.get_distribution_type() + '/' + str(config_FL.num_client()) + 'clients/client' + str(client_id-1) + '.npz')
    elif config_FL.get_distribution_type()=="non_iid":
        subset_data = np.load('data_distribution/' + str(config_FL.get_dataset_name()) + '/' + config_FL.get_distribution_type() + '/'+ 'alpha_'+ str(config_FL.get_alpha_value())+'/'+ str(config_FL.num_client()) + 'clients/client' + str(client_id-1) + '.npz')
    else:
        raise ValueError(f"Unsupported distribution_type: {config_FL.get_distribution_type()} ")

    X, y = subset_data['data'], subset_data['targets']

    # FL rounds for continuous masking/one-time masking
    if config_FL.get_masking_type() == "maser":
        # For continuous masking
        ROUND_MASK = 1 # Mask Generation when server_round % 3 = 1 
        ROUND_ENC = 2 # Encrypt Params when server_round % 3 = 2
        ROUND_DEC = 0 # Decrypt Params when server_round % 3 = 0
        ROUND_MOD = 3 # Modulus for continutous masking
    elif config_FL.get_masking_type() == "grasp" or config_FL.get_masking_type() == "random":
        # For one-time masking
        ROUND_MASK = 1 # Mask Generation when server_round = 1 
        ROUND_ENC = 0 # Encrypt Params when server_round % 2 = 0
        ROUND_DEC = 1 # Decrypt Params when server_round % 2 = 1 and server_round != 1
        ROUND_MOD = 2 # Modulus for continutous masking
    else:
        raise ValueError("Unsupported masking type:", config_FL.get_masking_type())

    # Load model based on dataset
    if config_FL.get_model_name()=="lenet":
        model = lenet().to(DEVICE)  
    elif config_FL.get_model_name()=="conv8":
        model = conv8().to(DEVICE)
    else:
        raise ValueError(f"Unsupported model_name: {config_FL.get_model_name()}")

    # Print model shape
    for name, param in model.named_parameters():
        print(name, param.shape)

    class CustomAugmentedDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = data
            self.targets = targets
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image = self.data[idx]
            label = self.targets[idx]
            
            if self.transform:
                image = self.transform(image)

            return image, label

    class MnistClient(fl.client.NumPyClient):
        def __init__(self):
            super().__init__()
            self.grpc_max_message_length: int = 1074177918,
            self.round_timeout: float=600,
            self.mask = None
            self.aggregated_params = []
            self.aggregated_metrics = None

        def get_parameters(self, config):  # get parameters from the local model to process/encrypt
            self.aggregated_params, _, self.aggregated_metrics = super().get_parameters(config)
            return utils.get_parameters(model)

        def fit(self, parameters, config): # fit method works closely with the aggregate fit at fedavg_strategy
                
            # Apply aggregated mask & prepare slices
            def prepare_slices(local_params, mask, client_id):
                print("Client" + str(client_id) + ": Applying aggregated mask and preparing slices...")

                # Prepare full_mask that has the same shape as local_params to count important params 
                full_mask = [np.ones_like(arr) for arr in local_params] # same shape as local_params but initialized with ones
                mask_layer_id = 0
                for layer_id in range(len(full_mask)):
                    if not isinstance(full_mask[layer_id][0], np.ndarray):
                        pass # bias, do nothing
                    else:
                        # weight:
                        full_mask[layer_id] = np.multiply(mask[mask_layer_id], full_mask[layer_id])
                        mask_layer_id += 1
                
                # Count important params from mask
                flattened_global_mask = np.concatenate([arr.flatten() for arr in full_mask]) # flatten global mask
                nonzero_indices = np.flatnonzero(flattened_global_mask)  # indices for important params including biases
                num_important_params = nonzero_indices.shape[0]

                # Flatten params and apply mask to get important ones
                flattened_params = np.concatenate([arr.flatten() for arr in local_params])  # flatten masked local_params
                flattened_important_params = flattened_params[nonzero_indices]  # flatten important params including biases
                
                # Prepare slices
                slice_size = config_FL.get_slot_size()
                num_slices = int(np.ceil(num_important_params / slice_size))  # total slices needed
                if num_slices * slice_size >= num_important_params:
                    # Pad 0s to the last slice to reshape as [num_slices, slice_size]
                    padded_flattened_important_params = np.pad(flattened_important_params, (0, num_slices * slice_size - num_important_params), mode='constant', constant_values=0)
                else:
                    raise ValueError('ERROR: slices contain fewer elements than important params!')

                slices = np.reshape(padded_flattened_important_params, (num_slices, slice_size))  # slices to be encrypted, contains important params only
                print("Client" + str(client_id) + ": Slice preparation complete.")
                return slices
            
            # Encrypt slices using HE client instance
            def encrypt_slices(slices, client, client_id): 
                print('Client' + str(client_id) + ": Encryption begins...")
                slices_cts = [None] * slices.shape[0]
                for slice_id in range(slices.shape[0]):
                    encrypted = client.encryptFromPk(list(slices[slice_id]))
                    slices_cts[slice_id] = encrypted
                print('Client ' + str(client_id) + ": Encryption complete.")
                return slices_cts
            
            # Partial decryption of aggregated ciphertexts using HE client instance
            def partial_decrypt(agg_cts, client, client_id):
                print('Client ' + str(client_id) + ": Partial decryption begins...")
                local_pd_list = [None] * len(agg_cts)
                for slice_id in range(len(agg_cts)):
                    loaded = pickle.loads(agg_cts[slice_id])
                    pd_element = client.partialDecrypt(loaded)
                    local_pd_list[slice_id] = pd_element
                print('Client ' + str(client_id) + ": Partial decryption complete.")
                return local_pd_list
            
            def get_train_transform(config_FL):
                dataset_name = config_FL.get_dataset_name()
                
                if dataset_name == "cifar10":
                    train_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        # transforms.RandomCrop(32, padding=4),     #Augmentation
                        # transforms.RandomHorizontalFlip(),        #Augmentation
                        transforms.ToTensor(),
                        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))
                    ])
                elif dataset_name == "mnist":
                    train_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        # transforms.RandomCrop(28, padding=4),
                        # transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
                else:
                    raise ValueError(f"Unsupported dataset: {dataset_name}")
                
                return train_transform

            if (config_FL.sys_mode()):
                # Continuous masking (i.e., maser)
                if (config_FL.get_masking_type() == "maser"): 
                    # Maser: model training + mask generation
                    if (config['server_round'] % ROUND_MOD == ROUND_MASK): 
                        print('Client ' + str(client_id) + ": Local training & mask generation (continuous) begins...")
                        utils.set_parameters(model, parameters) # update local model params with the global params

                        print("Pre-processing the data...")
                        train_dataset = CustomAugmentedDataset(X, y, transform=get_train_transform(config_FL))
                        trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
                        #drop_last must be true for non_iid to avoid size mismatch (got input: [10], target: [1])

                        utils.train(model, trainloader, config['server_round'], epochs=config_FL.get_local_ep())

                        self.mask = maser(utils.get_parameters(model), config_FL.get_masking_threshold()) # generate mask on the trained model
                        mask_list_json = json.dumps(self.mask) # serialize local mask
                        
                        print("Client " + str(client_id) + ": Mask gen (continuous) complete.")
                        return [], len(X), {"mask": mask_list_json}
                                            
                    # Prepare slices and encrypt
                    elif (config['server_round'] % ROUND_MOD == ROUND_ENC): 
                        self.mask = parameters # receive aggregated mask
                        slices = prepare_slices(utils.get_parameters(model), self.mask, client_id) # apply aggregated mask and prepare slices

                        slices_cts = encrypt_slices(slices, client, client_id) # encrypt slices
                        slices_cts = pickle.dumps(slices_cts) # serialize encrypted slices
                        return [], len(X), {"slices_cts": slices_cts}
                    
                    # Partial decryption
                    else: # config['server_round'] % ROUND_MOD == ROUND_DEC  
                        local_pd_list = partial_decrypt(parameters, client, client_id)

                        global_slices_cts_compressed = pickle.dumps(local_pd_list) # serialize partially decrypted ciphertexts
                        del local_pd_list
                        gc.collect()
                        return [], len(X), {"global_slices_cts": global_slices_cts_compressed, "client_id": client_id} 
                    
                        
                    
                    

                elif (config_FL.get_masking_type() == "grasp" or config_FL.get_masking_type() == "random" ): # for 1 time masking 
                    start_time_round=time.time()
                    path = 'eval_metrics/fl_enc/round_start_time'
                    saveCSV.save(path, start_time_round, 'round start time', config['server_round'], 'server round')
                    # One-time Mask Generation
                    if (config['server_round'] == ROUND_MASK):
                        print("Client " + str(client_id) + ": Mask generation (one-time) begins...")
                        utils.set_parameters(model, parameters) # set initial model params
                        
                        # Mask Generation
                        if config_FL.get_masking_type() == "grasp":
                            train_dataset = CustomAugmentedDataset(X, y, transform=get_train_transform(config_FL))
                            trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
                            #drop_last must be true for non_iid to avoid size mismatch (got input: [10], target: [1])
                            self.mask = grasp(model, config_FL.get_masking_threshold(), trainloader, DEVICE)

                        elif config_FL.get_masking_type() == "random":
                            self.mask = random_mask(utils.get_parameters(model), config_FL.get_masking_threshold())

                        else:
                            raise ValueError(f"Unsupported making_type: {config_FL.get_masking_type()}")
                            
                        mask_list_json = json.dumps(self.mask)

                        print("Client " + str(client_id) + ": Mask generation (one-time) complete.")
                        return [], len(X), {"mask": mask_list_json}

                    # Local Training + Slice Preparation + Encryption
                    elif (config['server_round'] % ROUND_MOD == ROUND_ENC):                   
                        print("Pre-processing the data...")
                        train_dataset = CustomAugmentedDataset(X, y, transform=get_train_transform(config_FL))
                        trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)

                        #drop_last must be true for non_iid to avoid size mismatch (got input: [10], target: [1])

                        if config['server_round'] == 2:  
                            self.mask = parameters # receive aggregated mask in round 2
                        else:
                            utils.set_parameters(model, parameters) # receive aggregated params in future even rounds

                        utils.train(model, trainloader, config['server_round'], epochs=config_FL.get_local_ep())

                        slices = prepare_slices(utils.get_parameters(model), self.mask, client_id)

                        # Encryption
                        slices_cts = encrypt_slices(slices, client, client_id) 
                        slices_cts = pickle.dumps(slices_cts) # serialize encrypted slices

                        return [], len(X), {"slices_cts": slices_cts}
                    
                    # Partial Decryption
                    else: # config['server_round'] % ROUND_MOD == ROUND_DEC
                        local_pd_list = partial_decrypt(parameters, client, client_id)
                        global_slices_cts_compressed = pickle.dumps(local_pd_list)
                        del local_pd_list
                        gc.collect()
                        return [], len(X), {"global_slices_cts": global_slices_cts_compressed, "client_id": client_id}
            #Vanilla FL
            else: 
                print("Pre-processing the data...")
                train_dataset = CustomAugmentedDataset(X, y, transform=get_train_transform(config_FL))
                trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
                #drop_last should be true to avoid size mismatch (got input: [10], target: [1])
                utils.set_parameters(model, parameters)

                utils.train(model,trainloader,config['server_round'], epochs=config_FL.get_local_ep())
                
                local_model = utils.get_parameters(model)
                return local_model, len(X), {}

    # Start Flower client
    server_ip, server_port = config_FL.get_server_addr()
    server_address = server_ip + ":" + str(server_port)
    fl.client.start_numpy_client(server_address=server_address, grpc_max_message_length= 1974177918, client=MnistClient())