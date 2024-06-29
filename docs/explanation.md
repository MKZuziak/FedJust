# 1. Quick set-up guide

FedJust was made to be easy to set up and customize. This brief walkthrough will show you how to launch a basic simulation using the library.

```
import os
from functools import partial

from torch import optim

from (...) import DATASET
from (...) import NEURAL_NETWORK
from FedJust.model.federated_model import FederatedModel
from FedJust.node.federated_node import FederatedNode
from FedJust.simulation.adaptive_optimizer_simulation import Adaptive_Optimizer_Simulation
from FedJust.aggregators.fedopt_aggregator import Fedopt_Optimizer
from FedJust.files.archive import create_archive

def exemplary_simulation():
    (metrics_savepath, 
     nodes_models_savepath, 
     orchestrator_model_savepath) = create_archive(os.getcwd())
  
    train, test = DATASET
    net_architecture = NEURAL_NETWORK()
    optimizer_architecture = partial(optim.SGD, lr=0.001)
    model_tempate = FederatedModel(
        net=net_architecture,
        optimizer_template=optimizer_architecture,
        loader_batch_size=32
    )
    node_template = FederatedNode()
    fed_avg_aggregator = Fedopt_Optimizer()
  
    simulation_instace = Adaptive_Optimizer_Simulation(model_template=model_tempate,
                                    node_template=node_template)
    simulation_instace.attach_orchestrator_model(orchestrator_data=test)
    simulation_instace.attach_node_model({
        3: [train, test],
        7: [train, test],
        11: [train, test],
        12: [train, test]
    })
    simulation_instace.training_protocol(
        iterations=5,
        sample_size=2,
        local_epochs=2,
        aggrgator=fed_avg_aggregator,
        learning_rate=1.0,
        metrics_savepath=metrics_savepath,
        nodes_models_savepath=nodes_models_savepath,
        orchestrator_models_savepath=orchestrator_model_savepath
    )
```

#### Defining the Imports

At the beginning of the script, we import several modules. Except for FedJust modules, we also import Optim, DATASET and NEURAL_NETWORK architecture. [Optim](https://pytorch.org/docs/stable/optim.html) is a PyTorch's package implementing various optimization algorithms that we will use with Python's [functools.partial](https://docs.python.org/3/library/functools.html) to create a [partial object](https://docs.python.org/3/library/functools.html#partial-objects) containg an Optimizer. DATASET is your custom dataset used for Federated Learning and NEURAL_NETWORK is your custom model defined as a child of a [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) - a standard convention for all architectures defined in PyTorch.

#### Create Archives

Since the FedJust was made to simulate federatred environment, it comes with a number of convienient classes and functions to make your life easier. `create_archive()`
