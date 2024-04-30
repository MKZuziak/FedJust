from typing import Any
from functools import partial
from collections import OrderedDict
import copy

from FedJust.model.federated_model import FederatedModel
from FedJust.files.loggers import node_logger

# import torch
# from numpy import array
# from numpy.random import default_rng
# from datasets import arrow_dataset
# from forcha.models.federated_model import FederatedModel
# from forcha.utils.loggers import Loggers
# from forcha.utils.helpers import find_nearest
# from forcha.components.settings.settings import Settings

# Setting up the node logger
node_logger = node_logger()

class FederatedNode:
    def __init__(self) -> None:
        """An abstract object representing a single node in the federated training.
        
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        self.node_id = None
        self.model = None
    
    
    def connect_data_id(
        self,
        node_id: int | str,
        model: FederatedModel,
        data: Any,
        orchestrator: bool = False
    ) -> None:
        """Attaches dataset and id to a node, creating an individualised version.
        
        Parameters
        ----------
        node_id: int | str
            ID of the node
        model: FederatedModel
            FederatedModel template that should be copied and attached to the node
        data: Any
            Data to be attached to a FederatedModel.
        orchestrator: bool (default to False)
            Boolean flag if the node is belonging to the orchestrator
        Returns
        -------
        None
        """
        self.node_id = node_id
        self.model = copy.deepcopy(model)
        self.model.attach_dataset_id(
            local_dataset=data,
            node_name=self.node_id,
            only_test=orchestrator)
    
    
    def get_weights(self) -> OrderedDict:
        """Extended API call to recover model weights. Causes the same effect as calling
        node.model.get_weigts()
        Parameters
        ----------
        None
        Returns
        OrderedDict
        None
        """
        return self.model.get_weights()
    
    
    def update_weights(self,
                       weights: OrderedDict) -> None:
        """Extended API call to update model. Causes the same effect as calling
        node.model.update_weights()
        Parameters
        ----------
        weights:OrderedDict
            OrderedDict to be uploaded onto the model.
        Returns
        None
        """
        self.model.update_weights(weights)
            

    def train_local_model(self,
                          iteration: int,
                          local_epochs: int,
                          mode: str = 'weights',
                          save_model: bool = False,
                          save_path: str = None) -> tuple[int, OrderedDict, list[float], list[float]]:
        """This function starts the server phase of the federated learning.
        In particular, it trains the model locally and then sends the weights.
        Then the updated weights are received and used to update
        the global model.
        
        Parameters
        ----------
        node: FederatedNode 
            Node that we want to train.
        iteration: int
            Current global iteration
        local_epochs: int
            Number of local epochs for which the node should be training.
        mode: str (default to 'weights')
            Mode = 'weights': Node will return model's weights.
            Mode = 'gradients': Node will return model's gradients.
        save_model: bool (default to False)
            Boolean flag to save a model.
        save_path: str (defualt to None)
            Path object used to save a model
        
        Returns
        -------
            Tuple[int, OrderedDict, List[float], List[float]]:
        """
        node_logger.info(f"[ITERATION {iteration} | NODE {self.node_id}] Starting training on node {self.node_id}")
        loss_list: list[float] = []
        accuracy_list: list[float] = []
        
        # If mode is set to "gradients" -> preserve a local model to calculate gradients
        if mode == 'gradients':
            self.model.preserve_initial_model()
        
        for epoch in range(local_epochs):
            metrics = self.local_training(
                iteration=iteration, 
                epoch=epoch
                )
            loss_list.append(metrics["loss"])
            accuracy_list.append(metrics["accuracy"])
        if save_model:
            self.model.store_model_on_disk(
                iteration=iteration, 
                path=save_path
                )
        node_logger.debug(f"[ITERATION {iteration} | NODE {self.node_id}] Results of training on node {self.node_id}: {accuracy_list}")
        
        if mode == 'weights':
            return (
                self.node_id,
                self.model.get_weights(),
                loss_list,
                accuracy_list
                )
        elif mode == 'gradients':
            return (
                self.node_id,
                self.model.get_gradients(),
                loss_list,
                accuracy_list
                )
        else:
            raise NameError()
    
    
    def local_training(self,
                       iteration: int,
                       epoch: int
                       ) -> dict[int, int]:
        """Helper method for performing one epoch of local training.
        Performs one round of Federated Training and pack the
        results (metrics) into the appropiate data structure.
        
        Parameters
        ----------
    
        Returns
        -------
            dict[int, int]: metrics from the training.
        """
        loss, accuracy = self.model.train(iteration=iteration, epoch=epoch)
        return {"loss": loss, "accuracy": accuracy}