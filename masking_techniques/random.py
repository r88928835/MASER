import numpy as np

def random_mask(local_params, threshold):
    """Generate random binary masks for weights in a neural network.

    Args:
        local_params: List of NumPy arrays containing model parameters
        percentage (float): The percentage of total parameters to set as 1 in each layer's mask.

    Returns:
        mask_list: Binary mask for weights (no biases), format is a list of list to use json serialization
    """
    local_weights = []
    weights_shapes = []
    for layer_id in range(len(local_params)):
        if not isinstance(local_params[layer_id][0], np.ndarray):
            pass # bias, do nothing
        else:
            # weight:
            local_weights.append(local_params[layer_id])
            weights_shapes.append(local_params[layer_id].shape)
    
    flattened_mask = np.concatenate([np.zeros_like(arr.flatten()) for arr in local_weights]) # flatten weights
    num_elements_to_select = int(flattened_mask.shape[0] * threshold/100)
    random_indices = np.random.choice(flattened_mask.shape[0], num_elements_to_select, replace=False)
    flattened_mask[random_indices] = 1

    # Construct the mask according to the shape of model weights
    mask_list = [np.zeros_like(arr) for arr in local_weights]
    start = 0
    for i, arr in enumerate(mask_list):
        end = start + arr.size
        mask_list[i] = flattened_mask[start:end].reshape(arr.shape)
        start = end
   
    mask_list = [arr.tolist() for arr in mask_list]  # return a list of list for json serialization
    return mask_list