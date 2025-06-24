import gc
import json
import pickle
import numpy as np
import time
from pympler import asizeof
from logging import WARNING
from time import monotonic

from flwr.common import (
    GRPC_MAX_MESSAGE_LENGTH,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy

import torch

import utils
import config_FL
import saveCSV
from models.lenet import lenet
from models.conv8 import conv8
from xmkckks_wrapper import *

from typing import Callable, Dict, List, Optional, Tuple, Union

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

# flake8: noqa: E501
class FedAvg(Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        round_timeout: float=600,
        grpc_max_message_length: int =1074177918,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. In case `min_fit_clients`
            is larger than `fraction_fit * available_clients`, `min_fit_clients`
            will still be sampled. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. In case `min_evaluate_clients`
            is larger than `fraction_evaluate * available_clients`, `min_evaluate_clients`
            will still be sampled. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        """
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)
        self.grpc_max_message_length=1074177918
        self.round_timeout=600

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.num_params = 0
        self.nonzero_indices = None

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        #time.sleep(20)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        nb_client= config_FL.num_client()
        #n_classes, n_features = config_FL.params_dataset()
        server=MPHEServer(server_id=0)

        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # DEVICE= torch.device(config_FL.get_cudaid() if torch.cuda.is_available() else "cpu")

        if config_FL.get_model_name() == "lenet":
            dummy_model = lenet()
        elif config_FL.get_model_name() == "conv8":
            dummy_model = conv8()
        else:
            raise ValueError(f"Unsupported model_name: {config_FL.get_model_name()}") 
        
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

        dummy_params = utils.get_parameters(dummy_model)

        def aggregate_masks(fit_metrics):
            threshold = nb_client / 2
            print("Server: Mask aggregation threshold:", threshold)

            agg_mask = None
            for i in range(nb_client):
                mask_list = json.loads(fit_metrics[i][1]['mask'])
                if i == 0:
                    agg_mask = [np.zeros_like(arr) for arr in mask_list]

                print(f"Server: Mask aggregation - Processing client {i}:")
                for j, sublist in enumerate(mask_list):
                    sublist_array = np.array(sublist)
                    agg_mask[j] += sublist_array

            thresholded_agg_mask = [np.where(sublist < threshold, 0, 1) for sublist in agg_mask]
            
            full_mask = [np.ones_like(arr) for arr in dummy_params] # list of np array that is the same as dummy_params but initialized with 1s
            mask_layer_id = 0
            for layer_id in range(len(full_mask)):
                if not isinstance(full_mask[layer_id][0], np.ndarray):     
                    pass # bias
                else:
                    # weight:
                    full_mask[layer_id] = np.multiply(thresholded_agg_mask[mask_layer_id], full_mask[layer_id])
                    mask_layer_id+=1
            flattened_mask = np.concatenate([arr.flatten() for arr in full_mask]) # flatten masked local_params
            num_params = flattened_mask.shape[0]
            nonzero_indices = np.flatnonzero(flattened_mask) # indices for important params

            parameters_aggregated = ndarrays_to_parameters(thresholded_agg_mask)
            print("Server: Mask aggregation complete.")
            return parameters_aggregated, {}, nonzero_indices, num_params

        def add_cts(fit_metrics):
            global_layers_ct0s = []
            # Aggregate CTs from all clients
            for i in range(nb_client):
                local_ct0s = pickle.loads(fit_metrics[i][1]['slices_cts'])
                print(f"Server: Adding CTs - Client {i}")
                if i == 0:
                    global_layers_ct0s = local_ct0s
                else:
                    for layer_id in range(len(local_ct0s)):
                        global_layers_ct0s[layer_id] = server.addCTs(global_layers_ct0s[layer_id], local_ct0s[layer_id])

            # Serialize global_layers_ct0s for further use
            parameters_aggregated = [pickle.dumps(layer) for layer in global_layers_ct0s]
            parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)
            metrics_aggregated = {}
            return parameters_aggregated, metrics_aggregated

        def aggregate_pd_n_decrypt(fit_metrics, nonzero_indices, num_params):
            client_ids = [None for _ in range(nb_client)]
            slice_size = config_FL.get_slot_size()
            num_slices = int(np.ceil(nonzero_indices.shape[0] / slice_size))  # total slices needed
            processed_objects = [[None for _ in range(nb_client)] for _ in range(num_slices)]

            # Aggregate decryption slices from clients
            for rcv_id in range(nb_client):
                client_id = fit_metrics[rcv_id][1]['client_id']
                client_ids[rcv_id] = client_id

                print("Server: Aggregate PD cts - Processing client:", client_id)
                pickled_global_slices_cts = pickle.loads(fit_metrics[rcv_id][1]['global_slices_cts']) 
                for slice_id in range(len(pickled_global_slices_cts)):
                    processed_objects[slice_id][rcv_id] = pickled_global_slices_cts[slice_id]

            # Aggregation of partially decrypted ciphertexts and decryption
            dec_slices = [server.decodeAfterPartialDecrypt(server.aggregate_pds(pd_slices, client_ids))
                for pd_slices in processed_objects]
            
            flattened_dec = np.array(dec_slices).flatten()
            flattened_dec = flattened_dec / nb_client # Average decrypted params by the number of participating clients TODO: is nb_clients same as M

            # Initialize and fill decrypted parameters, with mask applied
            flattened_dec_params = np.zeros(num_params)
            flattened_dec_params[nonzero_indices] = flattened_dec[:nonzero_indices.shape[0]]

            # Construct model params following the model architecture
            dec_params = [np.zeros_like(arr) for arr in dummy_params]
            start = 0
            for i, arr in enumerate(dec_params):
                end = start + arr.size
                dec_params[i] = flattened_dec_params[start:end].reshape(arr.shape)
                start = end

            parameters_aggregated = ndarrays_to_parameters(dec_params)
            metrics_aggregated = {}
            return parameters_aggregated, metrics_aggregated

        if (config_FL.sys_mode()):  

            if (config_FL.get_masking_type() == "maser" and server_round % ROUND_MOD == ROUND_MASK) or ( (config_FL.get_masking_type() == "grasp" or config_FL.get_masking_type() == "random") and server_round == ROUND_MASK):
                print("Server: Aggregate masks... Round:", server_round)
                parameters_aggregated, metrics_aggregated, self.nonzero_indices, self.num_params = aggregate_masks(fit_metrics)
                return parameters_aggregated, metrics_aggregated

            elif (server_round % ROUND_MOD == ROUND_ENC):
                print("Server: FedAvg begins, aggregate ciphertexts... Round:", server_round)
                parameters_aggregated, metrics_aggregated = add_cts(fit_metrics) 
                return parameters_aggregated, metrics_aggregated 

            else: # server_round % ROUND_MOD == ROUND_DEC:
                print('Server: Aggregate PDs & decryption begins... Round:', server_round)
                parameters_aggregated, metrics_aggregated = aggregate_pd_n_decrypt(fit_metrics, self.nonzero_indices, self.num_params)
                return parameters_aggregated, metrics_aggregated 
        
        else:  # For Vanilla FL
            global_layers = {}
            for layer_id in range(len(dummy_params)):
                global_layers[layer_id] = np.zeros_like(dummy_params[layer_id])
            weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
                       
            for i in range(nb_client):
                local_weights = weights_results[i][0]
                for layer_id in range(len(dummy_params)):
                    if not isinstance(dummy_params[layer_id][0], np.ndarray):
                        global_layers[layer_id]= [var1 + var2 for var1, var2 in zip(global_layers[layer_id],local_weights[layer_id])]
                    else:
                        for j in range(len(global_layers[layer_id])):
                            global_layers[layer_id][j]= [var1 + var2 for var1, var2 in zip(global_layers[layer_id][j],local_weights[layer_id][j])]              

            for layer_id in range(len(dummy_params)):
                if not isinstance(dummy_params[layer_id][0], np.ndarray):
                        global_layers[layer_id] = [element / nb_client for element in global_layers[layer_id]]
                else:
                    for i in range(len(global_layers[layer_id])):
                        global_layers[layer_id][i] = [element / nb_client for element in global_layers[layer_id][i]]
                        
            global_params = []
            for layer_id in range(len(dummy_params)):
                global_params.append(np.round(global_layers[layer_id],2)) 
            parameters_aggregated = ndarrays_to_parameters (global_params)

            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")
            return parameters_aggregated, metrics_aggregated
   
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        
        loss_aggregated={}
        metrics_aggregated={}

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        print("Fed Avg in fitstate Agg_Eva:", server_round)
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            print("Fed Avg in fitstate Agg_Eva in:", server_round)
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated