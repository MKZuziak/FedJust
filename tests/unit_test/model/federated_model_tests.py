import unittest
from functools import partial

import torch

from tests.test_props.nets import NeuralNetwork
from tests.test_props.datasets import return_mnist
from generative_fl.model.federated_model import FederatedModel

class ModelTests(unittest.TestCase):
    
    def test_init(self):
        # Initialization
        model = NeuralNetwork()
        optimizer_template = partial(torch.optim.SGD,lr=0.001)
        model_prop = FederatedModel(
            net=model,
            optimizer_template=optimizer_template,
            force_cpu=True)
        
        # Tests
        self.assertIsNotNone(model_prop)
        self.assertIsNotNone(model_prop.optimizer)
    
    
    def test_huggingface_loading(self):
        model = NeuralNetwork()
        optimizer_template = partial(torch.optim.SGD, lr=0.001)
        batch_size = 32
        model_prop_node = FederatedModel(
            net=model,
            optimizer_template=optimizer_template,
            force_cpu=False
        )
        model_prop_orchestrator = FederatedModel(
            net=model,
            optimizer_template=optimizer_template,
            force_cpu=False
        )
        train, test = return_mnist()
        model_prop_node.attach_huggingface_dataset(
            local_dataset = [train, test],
            node_name = 42,
            batch_size=32
        )
        model_prop_orchestrator.attach_huggingface_dataset(
            local_dataset = [test],
            node_name = 'orchestrator',
            only_test = True,
            batch_size=32
        )
        
        self.assertIsNotNone(model_prop_node.trainloader)
        self.assertIsNotNone(model_prop_node.testloader)
        self.assertEqual(model_prop_node.node_name, 42)
        self.assertIsNotNone(model_prop_orchestrator.testloader)
        self.assertEqual(model_prop_orchestrator.node_name, 'orchestrator')

if __name__ == "__main__":
    unittest.main()