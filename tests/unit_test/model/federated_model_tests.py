import unittest
from functools import partial
from collections import OrderedDict
import os

import torch
import numpy as np

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
        # Initialization
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
        # Dataset Loading
        train, test = return_mnist()
        model_prop_node.attach_huggingface_dataset(
            local_dataset = [train, test],
            node_name = 42,
            batch_size=batch_size
        )
        model_prop_orchestrator.attach_huggingface_dataset(
            local_dataset = [test],
            node_name = 'orchestrator',
            only_test = True,
            batch_size=batch_size
        )
        # Tests
        self.assertIsNotNone(model_prop_node.trainloader)
        self.assertIsNotNone(model_prop_node.testloader)
        self.assertEqual(model_prop_node.node_name, 42)
        self.assertIsNotNone(model_prop_orchestrator.testloader)
        self.assertEqual(model_prop_orchestrator.node_name, 'orchestrator')
    
    
    def test_get_send_weights(self):
        # Initialization
        model = NeuralNetwork()
        optimizer_template = partial(torch.optim.SGD, lr=0.001)
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
        # Weights retireve
        weights_node = model_prop_node.get_weights()
        weights_orchestrator = model_prop_orchestrator.get_weights()
        # Tests -> weights should be the same
        for (layer_node, layer_orchestrator) in zip(weights_node.values(), weights_orchestrator.values()):
            self.assertTrue(torch.allclose(layer_node, layer_orchestrator))
        
        # Fake weights generation and upload
        fake_weights = OrderedDict()
        for key, layer in weights_node.items():
            fake_weights[key] = torch.rand(layer.shape)
        model_prop_node.update_weights(fake_weights)
        weights_node = model_prop_node.get_weights()
        weights_orchestrator = model_prop_orchestrator.get_weights()
        # Test -> weights should not be the same
        for (layer_node, layer_orchestrator) in zip(weights_node.values(), weights_orchestrator.values()):
            self.assertFalse(torch.allclose(layer_node, layer_orchestrator))
    
    
    def test_saving_model(self):
        # Initialization
        model = NeuralNetwork()
        optimizer_template = partial(torch.optim.SGD, lr=0.001)
        model_prop_node = FederatedModel(
            net=model,
            optimizer_template=optimizer_template,
            force_cpu=False
        )
        # Dataset Loading
        train, test = return_mnist()
        model_prop_node.attach_huggingface_dataset(
            local_dataset = [train, test],
            node_name = 42,
            batch_size=32
        )
        
        # Saving
        path = os.getcwd()
        model_prop_node.store_model_on_disk(iteration=5,
                                            path=path)
        # Tests
        self.assertTrue(os.path.exists(os.path.join(path, "node_42_iteration_5.pt")))
        # Removing saved model
        os.remove(os.path.join(path, "node_42_iteration_5.pt"))
    
    
    def test_training_model(self):
        # Initialization
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
        # Dataset Loading
        train, test = return_mnist()
        model_prop_node.attach_huggingface_dataset(
            local_dataset = [train, test],
            node_name = 42,
            batch_size=batch_size
        )
        model_prop_orchestrator.attach_huggingface_dataset(
            local_dataset = [test],
            node_name = 'orchestrator',
            only_test = True,
            batch_size=batch_size
        )
        # Training and test
        for epoch in range(10):
            loss, accuracy = model_prop_node.train(iteration=5, epoch=epoch)
            self.assertIsNotNone(loss)
            self.assertIsNotNone(accuracy)
            # Tests -> weights should be the same
            weights_node = model_prop_node.get_weights()
            weights_orchestrator = model_prop_orchestrator.get_weights()
            for (layer_node, layer_orchestrator) in zip(weights_node.values(), weights_orchestrator.values()):
                self.assertFalse(torch.allclose(layer_node, layer_orchestrator))
            model_prop_orchestrator.update_weights(weights_node)
            
            weights_node = model_prop_node.get_weights()
            weights_orchestrator = model_prop_orchestrator.get_weights()
            for (layer_node, layer_orchestrator) in zip(weights_node.values(), weights_orchestrator.values()):
                self.assertTrue(torch.allclose(layer_node, layer_orchestrator))
            
        # Tests -> evaluation
        (test_loss,
        accuracy,
        f1score,
        precision,
        recall,
        accuracy_per_class,
        true_positive_rate,
        false_positive_rate) = model_prop_node.evaluate_model()
        
        self.assertIsNotNone(test_loss)
        self.assertIsNotNone(accuracy)
        self.assertIsNotNone(f1score)
        self.assertIsNotNone(precision)
        self.assertIsNotNone(recall)
        
                    
if __name__ == "__main__":
    unittest.main()