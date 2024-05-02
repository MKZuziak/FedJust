import os
from functools import partial

from torch import optim

from tests.test_props.datasets import return_mnist
from tests.test_props.nets import NeuralNetwork
from FedJust.model.federated_model import FederatedModel
from FedJust.node.federated_node import FederatedNode
from FedJust.simulation.simulation import Simulation
from FedJust.aggregators.aggregator import Aggregator
from FedJust.files.archive import create_archive

def integration_test():
    (metrics_savepath, 
     nodes_models_savepath, 
     orchestrator_model_savepath) = create_archive(os.getcwd())
    
    
    train, test = return_mnist()
    net_architecture = NeuralNetwork()
    optimizer_architecture = partial(optim.SGD, lr=0.001)
    model_tempate = FederatedModel(
        net=net_architecture,
        optimizer_template=optimizer_architecture,
        loader_batch_size=32
    )
    node_template = FederatedNode()
    fed_avg_aggregator = Aggregator()
    
    simulation_instace = Simulation(model_template=model_tempate,
                                    node_template=node_template)
    simulation_instace.attach_orchestrator_model(orchestrator_data=test)
    simulation_instace.attach_node_model({
        3: [train, test]
    })
    simulation_instace.training_protocol(
        iterations=5,
        sample_size=1,
        local_epochs=2,
        aggrgator=fed_avg_aggregator,
        metrics_savepath=metrics_savepath,
        nodes_models_savepath=nodes_models_savepath,
        orchestrator_models_savepath=orchestrator_model_savepath
    )


if __name__ == "__main__":
    integration_test()