import unittest
from functools import partial
from collections import OrderedDict
import os

import torch

from generative_fl.node.federated_node import FederatedNode
from generative_fl.model.federated_model import FederatedModel
from tests.test_props.datasets import return_mnist
from tests.test_props.nets import NeuralNetwork


class NodeTests(unittest.TestCase):

    def test_init(self):
        # Initialization
        node = FederatedNode()
        # Tests
        self.assertIsNotNone(node)
    
    
    def test_attach_dataset(self):
        # Initializaiton
        node = FederatedNode()
        train, test = return_mnist()
        # Preliminary tests
        self.assertIsNone(node.node_id)
        self.assertIsNone(node.model)
        model = NeuralNetwork()
        optimizer_template = partial(torch.optim.SGD,lr=0.001)
        model_prop = FederatedModel(
            net=model,
            optimizer_template=optimizer_template,
            loader_batch_size=32,
            force_cpu=True)
        node.connect_data_id(
            node_id = 42,
            model=model_prop,
            data=[train, test]
        )
        
        # Tests
        self.assertIsNotNone(node.node_id)
        self.assertIsNotNone(node.model)
    
    
    def test_train_node(self):
        # Initializaiton
        node = FederatedNode()
        train, test = return_mnist()
        # Preliminary tests
        self.assertIsNone(node.node_id)
        self.assertIsNone(node.model)
        model = NeuralNetwork()
        optimizer_template = partial(torch.optim.SGD,lr=0.001)
        model_prop = FederatedModel(
            net=model,
            optimizer_template=optimizer_template,
            loader_batch_size=32,
            force_cpu=True)
        node.connect_data_id(
            node_id = 42,
            model=model_prop,
            data=[train, test]
        )
        # Training
        path = os.getcwd()
        # Removing saved model
        iteration = 21
        epochs = 3
        (node_id, weights, acc, loss) = node.train_local_model(iteration=iteration,
                               local_epochs=epochs,
                               mode='weights',
                               save_model=True,
                               save_path=path)
        #Tests
        self.assertTrue(os.path.exists(os.path.join(path, "node_42_iteration_21.pt")))
        self.assertEqual(type(node_id), int)
        self.assertEqual(type(weights), OrderedDict)
        self.assertEqual(type(acc), list)
        self.assertEqual(type(loss), list)
        self.assertEqual(type(node_id), type(42))
        
        # Cleaning
        os.remove(os.path.join(path, "node_42_iteration_21.pt"))
    
    
    def test_receive_update_weights(self):
        # Initializaiton
        node = FederatedNode()
        orchestrator = FederatedNode()
        train, test = return_mnist()
        # Preliminary tests
        self.assertIsNone(node.node_id)
        self.assertIsNone(node.model)
        self.assertIsNone(orchestrator.node_id)
        self.assertIsNone(orchestrator.model)
        model = NeuralNetwork()
        optimizer_template = partial(torch.optim.SGD,lr=0.001)
        # Note Initialization
        model_prop = FederatedModel(
            net=model,
            optimizer_template=optimizer_template,
            loader_batch_size=32,
            force_cpu=True)
        node.connect_data_id(
            node_id = 42,
            model=model_prop,
            data=[train, test]
        )
        # Orchestrator Initialization
        orchestrator_prop = FederatedModel(
            net=model,
            optimizer_template=optimizer_template,
            loader_batch_size=32,
            force_cpu=False
        )
        orchestrator.connect_data_id(
            node_id='orchestrator',
            model=orchestrator_prop,
            data = [test],
            orchestrator=True)
        
        # Weights retireve
        weights_node = node.get_weights()
        weights_orchestrator = orchestrator.get_weights()
        # Tests -> weights should be the same
        for (layer_node, layer_orchestrator) in zip(weights_node.values(), weights_orchestrator.values()):
            self.assertTrue(torch.allclose(layer_node, layer_orchestrator))
        
        # Fake weights generation and upload
        fake_weights = OrderedDict()
        for key, layer in weights_node.items():
            fake_weights[key] = torch.rand(layer.shape)
        node.update_weights(fake_weights)
        weights_node = node.get_weights()
        weights_orchestrator = orchestrator.get_weights()
        # Test -> weights should not be the same
        for (layer_node, layer_orchestrator) in zip(weights_node.values(), weights_orchestrator.values()):
            self.assertFalse(torch.allclose(layer_node, layer_orchestrator))
        

if __name__ == "__main__":
    unittest.main()