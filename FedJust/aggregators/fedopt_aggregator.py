from collections import OrderedDict
import copy

from torch import zeros
from torch import div

from FedJust.aggregators.aggregator import Aggregator

class Fedopt_Optimizer(Aggregator):
    """Fedopt Optimizer that performs a generalized 
    version of Federated Averaging. Suitable for performing
    Federated Optimization based on gradients, with 
    verying learning rates.
    
    Attributes
    ----------
    None
    """
    
    def __init__(self) -> None:
        super().__init__()
    
    
    def optimize_weights(
        self,
        weights: dict[int:OrderedDict],
        gradients: dict[int: OrderedDict],
        learning_rate : float,
        ) -> OrderedDict:
        """FedOpt Aggregation Function (equal to FedAvg when lr=1.0) that returns the 
        updated version of the weights.
        
        Parameters
        ----------
        weights: dict[int: OrderedDict]
            Weights of the previous (central) model.
        gradients: dict[int: OrderedDict]
            Gradients (defined as trainedmodel - dispatched model)
        learning_rate: float
            Learning rate used to 
        
        Returns
        -------
        OrderedDict"""        
        updated_weights = OrderedDict((key, zeros(weights[key].size())) for key in weights.keys())
        for key in weights:
            updated_weights[key] = weights[key] + (learning_rate * (gradients[key]))
        return updated_weights
