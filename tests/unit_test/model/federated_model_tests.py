import unittest
from functools import partial

import torch

from tests.test_props.model import NeuralNetwork
from generative_fl.model.federated_model import FederatedModel

class ModelTests(unittest.TestCase):
    
    def test_init(self):
        # Initialization
        model = NeuralNetwork()
        optimizer_template = partial(torch.optim.SGD,lr=0.001)
        model_prop = FederatedModel(
            net=model,
            optimizer_template=optimizer_template,
            node_name=42,
            force_cpu=True)
        
        # Tests
        self.assertIsNotNone(model_prop)
        self.assertIsNotNone(model_prop.optimizer)
        self.assertEqual(model_prop.node_name, 42)

if __name__ == "__main__":
    unittest.main()