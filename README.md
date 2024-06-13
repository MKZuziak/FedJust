# What is FedJust?
FedJust was projected to be a fully adjustable code base that takes off the need to write boilerplate code for Federated Learning. Its components are fully modifiable building blocks that can be adjusted to your needs in a few lines of code. FedJust was designed to be a minimalist Python boilerplate for running Federated simulations (on a single or multiple GPUs at the same time). It does not aim to work as a Federated framework for actual communication between the devices, as there are already frameworks for doing that, e.g. [Flower](https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html) or part of [PySyft](https://github.com/OpenMined/PySyft). However, the niche of a well-documented and easily adjustable framework for the customization of FL experiments is still open - hence a FedJust tries to deliver the most straightforward boilerplate ecosystem that you may need for performing decentralised simulations in a controlled environment.

# How to use FedJust?
As FedJust was designed to be an easy-to-use boilerplate, it comes in several different ways to use it.
**Python pip**
As FedJust is registered in the Python Package Index (PyPI), it can be downloaded using the pip install command. Simply run:
```pip install fedjust```
**Template Cloning**
This GitHub repository is set up as a public template repository. It means, that it is possible to simply clone it and use it as a standalone repository without actually forking it. The package is built with the help of [Poetry](https://python-poetry.org/). To install the project together with a virtual environment, navigate to the main folder and run:
```poetry install```.
The virtual environment will be installed together with required dependencies.

# FedJust - Adjustable Components
Here, we present a few practical tips that may come in handy when projecting your own simulation
### Writing your own dataset loading method

To change the dataset loading method you should change the Federated Model attach_dataset_id method. This method always allows to assign a particular ID and a dataset to a node, hence it is important to remember, that Forcha Base assigns those two values at one method call (otherwise, you may mistakanely assign a certain dataset to a wrong client, which may influence the simulation's outcome). The baseline attach_dataset_id method is as follows:

```
    def attach_dataset_id(
        self,
        local_dataset: Any,
        node_name: int | str,
        only_test: bool = False,
        batch_size: int = 32
        ) -> None:

		### Here you should input your code
		self.trainloader = torch.utils.data.DataLoader(
                	your_training_data,
                	batch_size=batch_size,
                	shuffle=True,
                	num_workers=0,
            	)
		self.testloader = torch.utils.data.DataLoader(
                	your_test_data,
                	batch_size=batch_size,
                	shuffle=False,
                	num_workers=0,
            	)
```

Irrespective of how the data is passed to the method call, it should always be loaded using torch.utils.data.DataLoader. Otherwise, you may need to write a fully custom training loop

### Metrics and data structures guidelines

The use of internal data structures should be (if possible) coherent across the whole library. Here we provide a few baseline examples

```
SimulationInstance.network <- dict[id: FederatedNode]
```

SimulationInstance.network is a dictionary mapping each node to its respective identificator. It is also a central attribute of the Simulation instance, as it preserves all the nodes participating in the training over the whole instance's lifecycle.

`weights <- dict[id: OrderedDict]`

Weights (or gradients) of the participant should always be passed in the form of python dictionary, mapping node's ID number to an OrderedDict containing the particular set of weights. This guarantess integrity of all the functions that either accept weights as their argument or return in after processing.


```
training_results <- [node_id] = 
			{
			"iteration": [int] iteration,
			"node_id": [int] node_id
			"loss": [float] loss_list[-1],  
			"accuracy": [float] accuracy_list[-1],  
			"full_loss": [list] loss_list,  
			"full_accuracy": [list] accuracy_list}
```

Training results should be preserved in nested dictionary, mapping node_id to the internal dictionary with a following mapping: `{str: int, str: int, str: float, str: float, str: list, str: list}.` While it is more relax requirement, it may keep the internal structure of the preserved metrix intact.
