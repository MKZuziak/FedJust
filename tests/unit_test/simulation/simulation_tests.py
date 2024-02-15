import unittest
from functools import partial
from collections import OrderedDict
import os
import copy

import torch

from generative_fl.node.federated_node import FederatedNode
from generative_fl.model.federated_model import FederatedModel
from generative_fl.simulation.simulation import Simulation
from tests.test_props.datasets import return_mnist
from tests.test_props.nets import NeuralNetwork


class SimulationTests(unittest.TestCase):
    
    def test_init(self):
        model = NeuralNetwork()
        optimizer_template = partial(torch.optim.SGD,lr=0.001)
        
        node = FederatedNode()
        model_prop = FederatedModel(
            net=model,
            optimizer_template=optimizer_template,
            loader_batch_size=32,
            force_cpu=False)
        
        simulation = Simulation(model_template=model_prop,
                                node_template=node)
        self.assertIsNotNone(simulation)
        self.assertEqual(simulation.model_template, model_prop)
        self.assertEqual(simulation.node_template, node)
    
    
    def test_orchestrator_attach(self):
        model = NeuralNetwork()
        optimizer_template = partial(torch.optim.SGD,lr=0.001)
        
        node = FederatedNode()
        model_prop = FederatedModel(
            net=model,
            optimizer_template=optimizer_template,
            loader_batch_size=32,
            force_cpu=False)
        
        simulation = Simulation(model_template=model_prop,
                                node_template=node)
        _, test = return_mnist()
        simulation.attach_orchestrator_model(orchestrator_data=test)
        self.assertIsNotNone(simulation.orchestrator_model)
        self.assertIsNotNone(simulation.orchestrator_model.node_name)
        self.assertIsNotNone(simulation.orchestrator_model.device)
        self.assertIsNotNone(simulation.orchestrator_model.testloader)
    
    
    def test_orchestrator_nodes(self):
        model = NeuralNetwork()
        optimizer_template = partial(torch.optim.SGD,lr=0.001)
        
        node = FederatedNode()
        model_prop = FederatedModel(
            net=model,
            optimizer_template=optimizer_template,
            loader_batch_size=32,
            force_cpu=False)
        simulation = Simulation(model_template=model_prop,
                                node_template=node)
        train, test = return_mnist()
        fake_data = {1: [copy.deepcopy(train), copy.deepcopy(test)], 
                     2: [copy.deepcopy(train), copy.deepcopy(test)],
                     21: [copy.deepcopy(train), copy.deepcopy(test)],
                     41: [copy.deepcopy(train), copy.deepcopy(test)]}
        simulation.attach_node_model(nodes_data=fake_data)
        
        self.assertIsNotNone(
            simulation.network[1]
        )
        self.assertIsNotNone(
            simulation.network[2]
        )
        self.assertIsNotNone(
            simulation.network[21]
        )
        self.assertIsNotNone(
            simulation.network[41]
        )
        
        self.assertIsNotNone(
            simulation.network[1].model
        )
        self.assertIsNotNone(
            simulation.network[2].model
        )
        self.assertIsNotNone(
            simulation.network[21].model
        )
        self.assertIsNotNone(
            simulation.network[41].model
        )
                
        self.assertIsNotNone(
            simulation.network[1].model.net
        )
        self.assertIsNotNone(
            simulation.network[2].model.net
        )
        self.assertIsNotNone(
            simulation.network[21].model.net
        )
        self.assertIsNotNone(
            simulation.network[41].model.net
        )
        
        self.assertIsNotNone(
            simulation.network[1].model.trainloader
        )
        self.assertIsNotNone(
            simulation.network[2].model.trainloader
        )
        self.assertIsNotNone(
            simulation.network[21].model.trainloader
        )
        self.assertIsNotNone(
            simulation.network[41].model.trainloader
        )
        
        self.assertIsNotNone(
            simulation.network[1].model.testloader
        )
        self.assertIsNotNone(
            simulation.network[2].model.testloader
        )
        self.assertIsNotNone(
            simulation.network[21].model.testloader
        )
        self.assertIsNotNone(
            simulation.network[41].model.testloader
        )
        
        self.assertEqual(
            simulation.network[1].node_id, 1
        )
        self.assertEqual(
            simulation.network[2].node_id, 2
        )
        self.assertEqual(
            simulation.network[21].node_id, 21
        )
        self.assertEqual(
            simulation.network[41].node_id, 41
        )
        
        numbers = [1,2,21,41]
        for number in numbers:
            for number_2 in numbers:
                if number == number_2: 
                    self.assertEqual(simulation.network[number], simulation.network[number_2])
                    self.assertEqual(simulation.network[number].model, simulation.network[number_2].model)
                    self.assertEqual(simulation.network[number].model.net, simulation.network[number_2].model.net)
                    self.assertEqual(simulation.network[number].model.trainloader, simulation.network[number_2].model.trainloader)
                    self.assertEqual(simulation.network[number].model.testloader, simulation.network[number_2].model.testloader)
                else:
                    self.assertNotEqual(simulation.network[number], simulation.network[number_2])
                    self.assertNotEqual(simulation.network[number].model, simulation.network[number_2].model)
                    self.assertNotEqual(simulation.network[number].model.net, simulation.network[number_2].model.net)
                    self.assertNotEqual(simulation.network[number].model.trainloader, simulation.network[number_2].model.trainloader)
                    self.assertNotEqual(simulation.network[number].model.testloader, simulation.network[number_2].model.testloader)
        

if __name__ == '__main__':
    unittest.main()
        
        
        