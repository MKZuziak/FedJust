## Adjustable Components

Forcha Base was projected to be fully adjustable code base that takes off the need to write boilerplate code for Federated Learning. Its components are fully modifiable building blocks that can be adjusted to your needs in a few lines of code. Here, we present a few practical tips that make come in handy when projecting your own simulation.

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
