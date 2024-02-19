from collections import OrderedDict
import copy

import torch

class Aggregator():
    """Basic class for all Federated Aggregators"""
    def __init__(self) -> None:
        pass
    
    
    def aggregate_weights(
        self,
        weights: dict[int: OrderedDict]) -> OrderedDict:
        """Basic aggregate function (equal to FedAvg) that returns the aggregate version of the
        weights. Perform deepcopy on the passed parameters.
        
        Parameters
        ----------
        weights: dict[int: OrderedDict]
        
        Returns
        -------
        OrderedDict"""
        
        results = OrderedDict()
        for params in weights.values():
            for key in params:
                if results.get(key) is None:
                    results[key] = copy.deepcopy(params[key]) #Here we could add copy and deepcopy
                else:
                    results[key] += copy.deepcopy(params[key])

        for key in results:
            results[key] = torch.div(results[key], len(weights))
        return results