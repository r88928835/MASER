import torch
import torch.nn.utils.prune as prune
import numpy as np

def get_l1_norm(parameters):
    """
    Calculate the L1 norm for each parameter.

    Args:
    - parameters: Iterable containing model parameters

    Returns:
    - norms: List of L1 norms for each parameter
    """
    norms = [torch.norm(param, p=1).item() for param in parameters]
    return norms


def maser(local_params, percent):
    """
    Set the top percentage of parameters as 1 in the mask.

    Args:
    - local_params: List of NumPy arrays containing model parameters
    - percent: Percentage of parameters to set as 1

    Returns:
    - mask_list: Binary mask for weights (no biases), format is a list of list to use json serialization
    """

    # Extract weights from local_params and discard biases
    local_weights = []
    weights_shapes = []
    for layer_id in range(len(local_params)):
        if not isinstance(local_params[layer_id][0], np.ndarray):
            pass # bias, do nothing
        else:
            # weight:
            local_weights.append(local_params[layer_id])
            weights_shapes.append(local_params[layer_id].shape)
    
    flattened_weights = np.concatenate([arr.flatten() for arr in local_weights]) # flatten weights
    norms_weights = np.absolute(flattened_weights) # norms of flattened weights only, no biases

    threshold = np.percentile(norms_weights, 100 - percent)
    
    flattened_mask = np.zeros_like(flattened_weights)
    flattened_mask[norms_weights >= threshold] = 1

    # Construct the mask according to the shape of model weights
    mask_list = [np.zeros_like(arr) for arr in local_weights]
    start = 0
    for i, arr in enumerate(mask_list):
        end = start + arr.size
        mask_list[i] = flattened_mask[start:end].reshape(arr.shape)
        start = end
   
    mask_list = [arr.tolist() for arr in mask_list]  # return a list of list for json serialization
    return mask_list