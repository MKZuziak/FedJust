from typing import List

import numpy as np  

from generative_fl.node.federated_node import FederatedNode

def train_nodes(
    node: FederatedNode,
    iteration: int,
    local_epochs: int,
    mode: str = 'weights',
    save_model: bool = False,
    save_path: str = None) -> tuple[int, List[float]]:
    """Used to command the node to start the local training.
    Invokes .train_local_model method and returns the results.
    Parameters
    ----------
    node: FederatedNode 
        Node that we want to train.
    iteration: int
        Current (global) iteration.
    local_epochs: int
        Number of local epochs for which to train a node
    mode: str (default to False)
        Mode of the training. 
        Mode = 'weights': Node will return model's weights.
        Mode = 'gradients': Node will return model's gradients.
    save_model: bool (default to False)
        Boolean flag to enable model saving.
    save_path: str (default to None)
        Save path for preserving a model (applicable only when save_model = True)
    Returns
    -------
    tuple(node_id: str, weights)
    """
    node_id, weights, loss_list, accuracy_list = node.train_local_model(
        iteration = iteration,
        local_epochs = local_epochs,
        mode = mode,
        save_model = save_model,
        save_path=save_path)
    return (node_id, weights, loss_list, accuracy_list)


def sample_nodes(nodes: dict[int: FederatedNode], 
                 sample_size: int,
                 generator: np.random.Generator) -> dict[id: FederatedNode]:
    """Sample the nodes given the provided sample size. If sample_size is bigger
    or equal to the number of av. nodes, the sampler will return the original list.
    
    Parameters
    ----------
        nodes: dict[int: FederatedNode]) 
            Original dictionary of nodes to be sampled from.
        sample_size: int,
            Size of the sample
        generator: np.random.Generator
            A numpy generator initialized on the server side.
    
    Returns
    -------
        dict[id: FederatedNode]
    """
    sample = generator.choice(list(nodes.values()), size=sample_size, replace=False) # Conversion to array
    sample = {node.node_id: node for node in sample} # Back-conversion to dicitonary
    return sample


def sample_weighted_nodes(nodes: dict[int: FederatedNode], 
                          sample_size: int,
                          generator: np.random.Generator,
                          sampling_array: np.array) -> dict[id: FederatedNode]:
    """Sample the nodes given the provided sample size. It requires passing a sampling array
    containing list of weights associated with each node.
    
    Parameters
    ----------
        nodes: dict[int: FederatedNode]) 
            Original dictionary of nodes to be sampled from.
        sample_size: int,
            Size of the sample
        generator: np.random.Generator
            A numpy generator initialized on the server side.
        sampling_array: np.array
            Sampling array containing weights for the sampling
    
    Returns
    -------
        dict[id: FederatedNode]
    """
    sample = generator.choice(nodes, size=sample_size, p = sampling_array, replace=False)
    sample = {node.node_id: node for node in sample} # Back-conversion to dicitonary
    return sample