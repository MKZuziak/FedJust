
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
```

#### Create Archives

Since the FedJust was made to simulate a federated environment, it comes with a number of convenient classes and functions to make your life easier. `create_archive(path: str,    archive_name: str = time.strftime("%d %m %Y %H %M %S", time.gmtime())) -> tuple[str, str, str]` is one of such classes. It creates a directory with appropriate directories and returns an absolute path to a subdirectory for preserving metrics, nodes' models and orchestrator's model. Although calling a `create_archive()` is not strictly necessary, users may find it convenient that there is a simple function for creating an appropriate file structure and obtaining absolute paths to it.

```
    (metrics_savepath, 
     nodes_models_savepath, 
     orchestrator_model_savepath) = create_archive(os.getcwd())
```

#### Instantiate the Model Template

`Federated Model` is a baseline building block of all simulations within the FedJust. It is an abstract container for a model - understood as a combination of an already initiated Neural Network, Data Shufflers and Utility Methods.

```
    net_architecture = NEURAL_NETWORK()
    optimizer_architecture = partial(optim.SGD, lr=0.001)
    model_tempate = FederatedModel(
        net=net_architecture,
        optimizer_template=optimizer_architecture,
        loader_batch_size=32
    )
```

The template requires a torch.nn.Module, optimizer template and an indication of a loader batch size (to initialize PyTorch's [Data Loaders](https://pytorch.org/docs/stable/data.html)).

#### Initialize the Node Template and Define an Aggregation Strategy

`Federated Node` is another abstract container that serves as a boilerplate for writing your custom Federated Node. It is a combination of a particular `Federed Model` and a unique identifier.

`Fedopt_Optimizer` defines a concrete strategy of mixing the weights by the central entity.

```
    node_template = FederatedNode()
    fed_avg_aggregator = Fedopt_Optimizer()
```

#### Create a Simulation Instance, Attach Models and Datasets

The next step will consist of creating a simulation instance. It is an abstract object containing all the information about the nodes, states and the environment. We initialize it by passing a template of the `Federated Model` and `Federated Node` - those will be cloned separately for each individual client. Subsequently, we attach a dataset to each individual client by passing a unique identifier and a corresponding dataset (train and test set) for all the nodes we want to initialize in our network.

```
simulation_instace = Adaptive_Optimizer_Simulation(model_template=model_tempate,node_template=node_template)
```

Remark: The datasets should be either in [HuggingFace format](https://huggingface.co/docs/datasets/about_arrow) or in standard [PyTorch Format](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), so they can be cloned appropriately and loaded using PyTorch's [Data Loaders](https://pytorch.org/docs/stable/data.html). The end user can change this behaviour by modifying the Federated Model base code (more on this in [how-to-guides](how-to-guides.md)).

```
    simulation_instace = Adaptive_Optimizer_Simulation( model_template=model_tempate, node_template=node_template)
    simulation_instace.attach_orchestrator_model(orchestrator_data=test)
    simulation_instace.attach_node_model({
        3: [train, test],
        7: [train, test],
        11: [train, test],
        12: [train, test]
    })
```

#### Launch the Simulation

Finally, we are ready to launch the simulation using our predefined components. To this, execute the `simulation.training_protocol` method.

```
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

Except for mandatory arguments such as iterations, sample size or the number of local epochs, the method accepts global paths to a place where the metrics, nodes and orchestrator models should be saved. We obtained those paths by calling a `create_archive` function before, but we could also omit this step, passing other global paths to respective directories.
