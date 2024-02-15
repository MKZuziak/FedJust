from functools import partial
from typing import Any
import copy

import numpy as np

from generative_fl.model.federated_model import FederatedModel
from generative_fl.node.federated_node import FederatedNode


# import datasets 
# import copy
# import numpy as np
# from multiprocessing import Pool
# from torch import nn
# from typing import Union

# set_start_method set to 'spawn' to ensure compatibility across platforms.
from multiprocessing import set_start_method
set_start_method("spawn", force=True)


class Simulation():
    """Simulation class representing a generic simulation type.
        
        Attributes
        ----------
        model_template : FederatedModel
            Initialized instance of a Federated Model class that is uploaded to every client.
        node_template : FederatedNode
            Initialized instance of a Federated Node class that is used to simulate nodes.
        data : dict
            Local data used for the training in a dictionary format, mapping each client to its respective dataset.
        """
    
    
    def __init__(self, 
                 model_template: FederatedModel,
                 node_template: FederatedNode,
                 **kwargs
                 ) -> None:
        """Creating simulation instant requires providing an already created instance of model template
        and node template. Those instances then will be copied n-times to create n different nodes, each
        with a different dataset. Additionally, a data for local nodes should be passed in form of a dictionary,
        maping dataset to each respective client.

        Parameters
        ----------
        model_template : FederatedModel
            Initialized instance of a Federated Model class that will be uploaded to every client.
        node_template : FederatedNode
            Initialized instance of a Federated Node class that will be used to simulate nodes.
        **kwargs : dict, optional
            Extra arguments to enable selected features of the Orchestrator.
            passing full_debug to **kwargs, allow to enter a full debug mode.

        Returns
        -------
        None"""
        self.model_template = model_template
        self.node_template = node_template
        self.network = {}
    
    
    def attach_orchestrator_model(self,
                                  orchestrator_data: Any):
        """
        Attaches model of the orchestrator that is saved as an instance attribute.
        
        Parameters
        ----------
        orchestrator_data: Any 
            Orchestrator data that should be attached to the orchestrator model.
        
        Returns
        -------
        None"""
        self.orchestrator_model = copy.deepcopy(self.model_template)
        self.orchestrator_model.attach_dataset_id(local_dataset=[orchestrator_data],
                                                  node_name='orchestrator',
                                                  only_test=True)
    
    
    def attach_node_model(self,
                          nodes_data: dict):
        """
        Attaches models of the 
        
        Parameters
        ----------
        orchestrator_data: Any
            Orchestrator data that should be attached to nodes models.
        Returns
        -------
        None"""
        for node_id, data in nodes_data.items():
            self.network[node_id] = copy.deepcopy(self.node_template)
            self.network[node_id].connect_data_id(node_id = node_id,
                                                  model = copy.deepcopy(self.model_template),
                                                  data=data)
    

    # def train_protocol(self) -> None:
    #     """Performs a full federated training according to the initialized
    #     settings. The train_protocol of the generic_orchestrator.Orchestrator
    #     follows a classic FedAvg algorithm - it averages the local weights 
    #     and aggregates them taking a weighted average.
    #     SOURCE: Communication-Efficient Learning of
    #     Deep Networks from Decentralized Data, H.B. McMahan et al.

    #     Parameters
    #     ----------
        
    #     Returns
    #     -------
    #     int
    #         Returns 0 on the successful completion of the training."""
    #     # TRAINING PHASE ----- FEDAVG
    #     # FEDAVG - CREATE POOL OF WORKERS
    #     for iteration in range(self.iterations):
    #         self.orchestrator_logger.info(f"Iteration {iteration}")
    #         weights = {}
    #         training_results = {}
            
    #         # Checking for connectivity
    #         connected_nodes = [node for node in self.network]
    #         if len(connected_nodes) < self.sample_size:
    #             self.orchestrator_logger.warning(f"Not enough connected nodes to draw a full sample! Skipping an iteration {iteration}")
    #             continue
    #         else:
    #             self.orchestrator_logger.info(f"Nodes connected at round {iteration}: {[node.node_id for node in connected_nodes]}")
            
    #         # Weights dispatched before the training
    #         self.orchestrator_logger.info(f"Iteration {iteration}, dispatching nodes to connected clients.")
    #         for node in connected_nodes:
    #             node.model.update_weights(copy.deepcopy(self.central_model.get_weights()))
            
    #         # Sampling nodes and asynchronously apply the function
    #         sampled_nodes = sample_nodes(
    #             nodes = connected_nodes, 
    #             sample_size = self.sample_size,
    #             generator = self.generator
    #             ) # SAMPLING FUNCTION
    #         # FEDAVG - TRAINING PHASE
    #         # OPTION: BATCH TRAINING
    #         if self.batch_job:
    #             self.orchestrator_logger.info(f"Entering batched job, size of the batch {self.batch}")
    #             for batch in Helpers.chunker(sampled_nodes, size=self.batch):
    #                 with Pool(len(list(batch))) as pool:
    #                     results = [pool.apply_async(train_nodes, (node, iteration)) for node in batch]
    #                     for result in results:
    #                         node_id, model_weights, loss_list, accuracy_list = result.get()
    #                         weights[node_id] = model_weights
    #                         training_results[node_id] = {
    #                             "iteration": iteration,
    #                             "node_id": node_id,
    #                             "loss": loss_list[-1], 
    #                             "accuracy": accuracy_list[-1]
    #                             }
    #         # OPTION: NON-BATCH TRAINING
    #         else:
    #             with Pool(self.sample_size) as pool:
    #                 results = [pool.apply_async(train_nodes, (node, iteration)) for node in sampled_nodes]
    #                 for result in results:
    #                         node_id, model_weights, loss_list, accuracy_list = result.get()
    #                         weights[node_id] = model_weights
    #                         training_results[node_id] = {
    #                             "iteration": iteration,
    #                             "node_id": node_id,
    #                             "loss": loss_list[-1], 
    #                             "accuracy": accuracy_list[-1]
    #                             }
    #         # TRAINING AND TESTING RESULTS BEFORE THE MODEL UPDATE
    #         # METRICS: PRESERVING TRAINING ON NODES RESULTS
    #         if self.settings.save_training_metrics:
    #             save_training_metrics(
    #                 file = training_results,
    #                 saving_path = self.settings.results_path,
    #                 file_name = "training_metrics.csv"
    #                 )
    #         # METRICS: TEST RESULTS ON NODES (TRAINED MODEL)
    #             for node in sampled_nodes:
    #                 save_model_metrics(
    #                     iteration = iteration,
    #                     model = node.model,
    #                     logger = self.orchestrator_logger,
    #                     saving_path = self.settings.results_path,
    #                     file_name = 'local_model_on_nodes.csv'
    #                     )
                    
    #         # FEDAVG: AGGREGATING FUNCTION
    #         avg = Aggregators.compute_average(copy.deepcopy(weights)) # AGGREGATING FUNCTION
    #         # FEDAVG: UPDATING THE NODES
    #         for node in connected_nodes:
    #             node.model.update_weights(copy.deepcopy(avg))
    #         # FEDAVG: UPDATING THE CENTRAL MODEL 
    #         self.central_model.update_weights(copy.deepcopy(avg))

    #         # TESTING RESULTS AFTER THE MODEL UPDATE
    #         if self.settings.save_training_metrics:
    #             save_model_metrics(
    #                 iteration = iteration,
    #                 model = self.central_model,
    #                 logger = self.orchestrator_logger,
    #                 saving_path = self.settings.results_path,
    #                 file_name = "global_model_on_orchestrator.csv"
    #             )
    #             for node in connected_nodes:
    #                 save_model_metrics(
    #                     iteration = iteration,
    #                     model = node.model,
    #                     logger = self.orchestrator_logger,
    #                     saving_path = self.settings.results_path,
    #                     file_name = "global_model_on_nodes.csv")
                            
    #         if self.full_debug == True:
    #             log_gpu_memory(iteration=iteration)

    #     self.orchestrator_logger.critical("Training complete")
    #     return 0
                        