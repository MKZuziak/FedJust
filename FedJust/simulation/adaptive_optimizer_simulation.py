from functools import partial
from collections import OrderedDict
from typing import Any
import copy
from multiprocessing import Pool
import os

import numpy as np

from FedJust.model.federated_model import FederatedModel
from FedJust.node.federated_node import FederatedNode
from FedJust.simulation.simulation import Simulation
from FedJust.operations.orchestrations import train_nodes, sample_nodes
from FedJust.aggregators.aggregator import Aggregator
from FedJust.operations.evaluations import evaluate_model, automatic_node_evaluation
from FedJust.files.handlers import save_nested_dict_ascsv
from FedJust.files.loggers import orchestrator_logger
from FedJust.utils.computations import average_of_weigts


# set_start_method set to 'spawn' to ensure compatibility across platforms.
from multiprocessing import set_start_method
set_start_method("spawn", force=True)
# Setting up the orchestrator logger
orchestrator_logger = orchestrator_logger()

class Adaptive_Optimizer_Simulation(Simulation):
    """Simulation class representing a generic simulation type.
        
        Attributes
        ----------
        model_template : FederatedModel
            Initialized instance of a Federated Model class that is uploaded to every client.
        node_template : FederatedNode
            Initialized instance of a Federated Node class that is used to simulate nodes.
        data : dict
            Local data used for the training in a dictionary format, mapping each client to its respective dataset."""
    
    
    def __init__(
        self, 
        model_template: FederatedModel, 
        node_template: FederatedNode, 
        seed: int = 42, 
        **kwargs
        ) -> None:
        super().__init__(model_template, node_template, seed, **kwargs)
    
    
    def training_protocol(
        self,
        iterations: int,
        sample_size: int,
        local_epochs: int,
        aggrgator: Aggregator,
        learning_rate: float,
        metrics_savepath: str,
        nodes_models_savepath: str,
        orchestrator_models_savepath: str
        ) -> None:
        """Performs a full federated training according to the initialized
        settings. The train_protocol of the generic_orchestrator.Orchestrator
        follows a classic FedOpt algorithm - it averages the local gradients 
        and aggregates them using a selecred optimizer.
        SOURCE: 

        Parameters
        ----------
        iterations: int
            Number of (global) iterations // epochs to train the models for.
        sample_size: int
            Size of the sample
        local_epochs: int
            Number of local epochs for which the local model should
            be trained.
        aggregator: Aggregator
            Instance of the Aggregator object that will be used to aggregate the result each round
        learning_rate: float
            Learning rate to be used for optimization.
        metrics_savepath: str
            Path for saving the metrics
        nodes_models_savepath: str
            Path for saving the models in the .pt format.
        orchestrator_models_savepath: str
            Path for saving the orchestrator models.
            
        Returns
        -------
        int
            Returns 0 on the successful completion of the training.
        """
        for iteration in range(iterations):
            orchestrator_logger.info(f"Iteration {iteration}")
            
            # # Updating weights for every node
            # for node in self.network.values():
            #     node.update_weights(copy.deepcopy(self.central_model.get_weights()))
            
            # Sampling nodes
            sampled_nodes = sample_nodes(
                nodes = self.network,
                sample_size = sample_size,
                generator = self.generator
            )
            
            # Training nodes
            training_results, gradients = self.train_epoch(
                sampled_nodes=sampled_nodes,
                iteration=iteration,
                local_epochs=local_epochs,
                mode='gradients',
                save_model=True,
                save_path=nodes_models_savepath
            )
            
            # Preserving metrics of the training
            save_nested_dict_ascsv(
                data=training_results,
                save_path=os.path.join(metrics_savepath, 'training_metrics.csv'))
            
            # Testing nodes on the local dataset before the model update (only sampled nodes).
            automatic_node_evaluation(
                iteration=iteration,
                nodes=sampled_nodes,
                save_path=os.path.join(metrics_savepath, "before_update_metrics.csv"))
            
            avg_gradients = average_of_weigts(gradients)
            # Updating weights
            new_weights = aggrgator.optimize_weights(
                weights=self.orchestrator_model.get_weights(),
                gradients = avg_gradients,
                learning_rate = learning_rate,
                )
            
            # Updating weights for each node in the network
            for node in self.network.values():
                node.update_weights(new_weights)
            # Updating the weights for the central model
            self.orchestrator_model.update_weights(new_weights)
            
            # Preserving the orchestrator's model
            self.orchestrator_model.store_model_on_disk(iteration=iteration,
                                                        path=orchestrator_models_savepath)
            
            # Evaluating the new set of weights on local datasets.
            automatic_node_evaluation(
                iteration=iteration,
                nodes=self.network,
                save_path=os.path.join(metrics_savepath, "after_update_metrics.csv"))
            # Evaluating the new set of weights on orchestrator's dataset.
            evaluate_model(
                iteration=iteration,
                model=self.orchestrator_model,
                save_path=os.path.join(metrics_savepath, "orchestrator_metrics.csv"))
        # self.orchestrator_logger.critical("Training complete")
        # return 0