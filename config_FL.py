def num_client(): # number of clients to be used in FL
    return 5

def sys_mode():
    # FL with Encryption : True / Normal FL : False
    return True

def nb_round():
    return 10    # TODO for encryption the round # will be multiplied by 2 For Vanilla FL round # will stay the same as given Change accordingly

def get_slot_size():
    log_slots = 13 # LogSlots used in the ckks.ParametersLiteral, refer to ./export.go
    return 2**log_slots

def get_cudaid(): # for GPUs, cuda id; otherwise the program will use CPU
    cuda_id = "cuda:2"
    return cuda_id

def get_masking_threshold(): # masking threshold
    threshold = 90
    return threshold

def get_masking_type():
    masking_type = "maser" # maser (Continuous), grasp (1 time masking) or random (1 time masking)
    return masking_type

def get_model_name():
    model_name = "conv8" # lenet or conv8 (conv8 for cifar10)
    return model_name

def get_dataset_name():
    dataset_name = "cifar10" # mnist or cifar10
    return dataset_name

def get_distribution_type():
    distribution_type = "non_iid" # iid or non_iid
    return distribution_type

def get_alpha_value():  # alpha for non_iid data distribution
    alpha = float(10)
    return alpha

def get_server_addr(): # localhost for one machine
    # ip= "xxx.xxx.xxx.xxx"
    ip= "0.0.0.0"
    port = 5042
    return ip, port

def get_key_manager_addr(): # localhost for one machine
    # ip= "xxx.xxx.xxx.xxx" # same as server address
    ip= "0.0.0.0" # same as server address
    port_key_send = 5040 # port for clients sending pub keys for key aggregation
    port_key_recv = 5041 # port for clients receiving the aggregated pub key
    return ip, port_key_send, port_key_recv

def get_localhost():
    ip = "0.0.0.0"
    return ip

def get_local_ep():
    return 1

def get_local_lr():
    return 1e-3

def get_mu_value(): # to train on non_iid data, mu value is needed
    mu=1 # mu = 0 is equivalent to FedAvg
    return mu