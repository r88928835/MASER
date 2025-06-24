"""https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py"""

import copy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return X, y

def grasp(net, threshold, train_dataloader, device, num_classes=10, samples_per_class=25, num_iters=1, T=200, reinit=True):
    eps = 1e-10
    keep_ratio = threshold/100

    old_net = copy.deepcopy(net)
    net = net.to(device)

    net.zero_grad()

    weights = []

    for layer in net.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            if isinstance(layer, nn.Linear) and reinit:
                nn.init.xavier_normal_(layer.weight)
            weights.append(layer.weight)

    inputs_one = []
    targets_one = []

    grad_w = None
    for w in weights:
        w.requires_grad_(True)

    for it in range(num_iters):
        inputs, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
        N = inputs.shape[0]
        inputs, targets = inputs.to(device), targets.to(device)

        din = copy.deepcopy(inputs)
        dtarget = copy.deepcopy(targets)
        inputs_one.append(din[:N // 2])
        targets_one.append(dtarget[:N // 2])
        inputs_one.append(din[N // 2:])
        targets_one.append(dtarget[N // 2:])


        outputs = net.forward(inputs[:N // 2]) / T
        loss = F.cross_entropy(outputs, targets[:N // 2])

        grad_w_p = autograd.grad(loss, weights)
        if grad_w is None:
            grad_w = [g.to(device) for g in grad_w_p]
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx].to(device)

        outputs = net.forward(inputs[N // 2:]) / T
        loss = F.cross_entropy(outputs, targets[N // 2:])
        grad_w_p = autograd.grad(loss, weights, create_graph=False)
        if grad_w is None:
            grad_w = [g.to(device) for g in grad_w_p]
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx].to(device)

    ret_inputs = []
    ret_targets = []

    for it in range(len(inputs_one)):
        inputs = inputs_one.pop(0).to(device)
        targets = targets_one.pop(0).to(device)
        ret_inputs.append(inputs)
        ret_targets.append(targets)
        outputs = net.forward(inputs) / T
        loss = F.cross_entropy(outputs, targets)

        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for layer in net.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    old_net.cpu()
    net.cpu()
    grads = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            grads[old_modules[idx]] = -layer.weight.data * layer.weight.grad  # -theta_q Hg

    # Gather all scores in a single vector and normalize
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    #TODO: to normalize search for #to avoid normalization and uncomment those lines
    #norm_factor = torch.abs(torch.sum(all_scores)) + eps   #to avoid normalization
    #all_scores.div_(norm_factor)  #to avoid normalization

    num_params_to_rm = int(len(all_scores) * (1 - keep_ratio))

    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = list()
    for _, g in grads.items():
        # Move the tensor to the specified device
        # g = g.to(device)
        # g = g.detach().cpu()

        # Move norm_factor to the same device as g
        #norm_factor = norm_factor.to(device)  #to avoid normalization

        # Use .item() to get a Python scalar
        keep_masks.append( ((g <= acceptable_score).float()).numpy().tolist() )
        #keep_masks.append(((g / norm_factor) <= acceptable_score).float())         #to avoid normalization

    # for layer_idx, mask in enumerate(keep_masks):    To see the masks of each layer
    #     print(f"Layer {layer_idx} Mask:")
    #     print(mask)

    # names = [name for name, _ in net.named_modules()]
    # Print the size and shape of the masks   
    #TODO: to print the scores of each layer uncomment the rest of the part
    # for name, mask in zip(names, keep_masks):
    #     print(f"Mask for {name} - Size: {mask.size()}, Shape: {mask.shape}")   #To print the selected params

    # for printing selected parameters for each layer
    # for name, mask in zip(names, keep_masks):
    #     selected_params = mask.nonzero(as_tuple=True)
    #     print(f"Selected Parameters in {name}:", selected_params)

    #numel, numprune = print_scores(keep_masks, names)    #Uncomment for Printing ##########
    #print the important parameters

    # for name, mask in zip(names, keep_masks):
    #     selected_params = mask.nonzero(as_tuple=True)
    #     print(f"Selected Parameters in {name}:", selected_params)


    #print(f"- Intended prune ratio:\t{1-keep_ratio}")
    #print(f"- Actual prune ratio:\t{1 - (numprune / numel) if numel != 0 else 'N/A'}")
    #print(f"- Threshold:           {acceptable_score}")

    # release GPU memory
    del old_net, net
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    
    gc.collect()
    return keep_masks


def print_scores(keep_masks, names):
    head_str = f"| {'Layer':<17}| {'Before':<14}| {'After':<14}| {'Ratio':<10}|"
    head_sep = "=" * len(head_str)
    print(head_sep)
    print(head_str)
    print(head_sep)

    full_numel = 0
    full_numprune = 0
    for name, mask in zip(names, keep_masks): 
        numel = torch.numel(torch.tensor(mask).clone().detach())#numel = torch.numel(mask)
        numprune = torch.sum(mask).item()
        
        # Check for zero division
        if numel == 0:
            ratio = "N/A"
        else:
            ratio = str(np.round(numprune / numel, 4))

        layer_info = f"| - {name:<15}| {numel:<14}| {numprune:<14}| {ratio:<10}|"
        print(layer_info)

        full_numel += numel
        full_numprune += numprune

    print(head_sep, "\n")
    
    # Check for zero division in the total ratio calculation
    total_ratio = "N/A" if full_numel == 0 else str(np.round(full_numprune / full_numel, 4))
    
    print(f"- Total prune ratio:\t{1 - (full_numprune / full_numel) if full_numel != 0 else 'N/A'}")
    print(f"- Total number of parameters:\t{full_numel}")
    print(f"- Total number of pruned parameters:\t{full_numprune}")
    print(f"- Total prune ratio (alternative calculation):\t{total_ratio}")
    
    return full_numel, full_numprune
