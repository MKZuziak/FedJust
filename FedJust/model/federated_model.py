import torch
from functools import partial
import copy
import os
from collections import OrderedDict

import numpy as np
import torch
import datasets
from torchvision import transforms
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from FedJust.files.loggers import model_logger


# Setting up the model logger
model_logger = model_logger()


class FederatedModel:
    """This class is used to encapsulate the (PyTorch) federated model that
    we will train. It accepts only the PyTorch models and 
    provides a utility functions to initialize the model, 
    retrieve the weights or perform an indicated number of traning
    epochs.
    """
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer_template: partial,
        loader_batch_size: int,
        force_cpu: bool = False) -> None:
        """Initialize the Federated Model. This model will be attached to a 
        specific client and will wait for further instructionss
        
        Parameters
        ----------
        net: nn.Module 
            The Neural Network architecture that we want to use.
        optimizer_template: functools.partial
            The partial function of the optimizer that will be used as a template
        node_name: int | str
            The name of the node
        loader_batch_size: int
            Batch size of the trainloader and testloader.
        force_gpu: bool = False
            Option to force the calculations on cpu even if the gpu is available.
        
        Returns
        -------
        None
        """
        # Primary computation device: GPU or CPU
        if force_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Second computaiton device to offload parameters: CPU
        self.cpu = torch.device("cpu")
        
        self.initial_model = None
        self.optimizer = None  
        self.net = copy.deepcopy(net)
        self.node_name = None
        self.batch_size = loader_batch_size
        # List containing all the parameters to update
        params_to_update = []
        for _, param in self.net.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)
        self.optimizer = optimizer_template(params_to_update)
        

    def attach_dataset_id(
        self,
        local_dataset: list[datasets.arrow_dataset.Dataset, datasets.arrow_dataset.Dataset] | list[datasets.arrow_dataset.Dataset],
        node_name: int | str,
        only_test: bool = False,
        hugging_face_map: bool = True
        ) -> None:
        """Attaches huggingface dataset to the model by firstly converting it into a pytorch-appropiate standard.
        
        Parameters
        ----------
        local_dataset: list[datasets.arrow_dataset.Dataset] 
            A local dataset that should be loaded into DataLoader
        node_name: int | str
            The name of the node attributed to particular dataset
        only_test: bool [default to False]: 
            If true, only a test set will be returned
        batch_size: int [default to 32]:
            Batch size used in test and train loader
        higging_face_map: bool [default to True]:
            If set to True, will use hugging face map function,
            that takes more time to process but results in a more
            stable and reversable transformation of the results.
        Returns
        -------------
        None
        """
        self.node_name = node_name
        if only_test == False:
            if hugging_face_map:
                convert_tensor = transforms.ToTensor()
                local_dataset[0] = local_dataset[0].map(lambda sample: {"image": convert_tensor(sample['image'])})
                local_dataset[0].set_format("pt", columns=["image"], output_all_columns=True)
                    
                local_dataset[1] = local_dataset[1].map(lambda sample: {"image": convert_tensor(sample['image'])})
                local_dataset[1].set_format("pt", columns=["image"], output_all_columns=True)
            else:
                local_dataset[0] = local_dataset[0].with_transform(self.transform_func)
                local_dataset[1] = local_dataset[1].with_transform(self.transform_func)
            self.trainloader = torch.utils.data.DataLoader(
                local_dataset[0],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
            )
            self.testloader = torch.utils.data.DataLoader(
                local_dataset[1],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            if hugging_face_map:
                convert_tensor = transforms.ToTensor()
                local_dataset[0] = local_dataset[0].map(lambda sample: {"image": convert_tensor(sample['image'])})
                local_dataset[0].set_format("pt", columns=["image"], output_all_columns=True)
            else:
                local_dataset[0] = local_dataset[0].with_transform(self.transform_func)
            self.testloader = torch.utils.data.DataLoader(
                local_dataset[0],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )


    # def print_model_footprint(self) -> None:
    #     """Prints all the information about the model..
    #     Args:
    #     """
    #     unique_hash = hash(next(iter(self.trainloader))['image'] + self.node_name)
        
    #     string = f"""
    #     model id: {self.node_name}
    #     device: {self.device},
    #     optimizer: {self.optimizer},
    #     unique hash: {unique_hash}
    #     """
    #     return (self.node_name, self.device, self.optimizer, unique_hash)
        
    #     # num_examples = {
    #     #     "trainset": len(self.training_set),
    #     #     "testset": len(self.test_set),
    #     # }
    #     # targets = []
    #     # for _, data in enumerate(trainloader, 0):
    #     #     targets.append(data[1])
    #     # targets = [item.item() for sublist in targets for item in sublist]
    #     # model_logger.info(f"{self.node_name}, {Counter(targets)}")
    #     # model_logger.info(f"{self.node_name}: Training set size: {num_examples['trainset']}")
    #     # model_logger.info(f"{self.node_name}: Test set size: {num_examples['testset']}")


    # def get_weights_list(self) -> list[float]:
    #     """Get the parameters of the network.
        
    #     Parameters
    #     ----------
        
    #     Returns
    #     -------
    #     List[float]: parameters of the network
    #     """
    #     return [val.cpu().numpy() for _, val in self.net.state_dict().items()]


    def get_weights(self) -> None:
        """Get the weights of the network.
        
        Parameters
        ----------
        
        Raises
        -------------
            Exception: if the model is not initialized it raises an exception.
        
        Returns
        -------------
            _type_: weights of the network
        """
        self.net.to(self.cpu) # Dupming weights on cpu.
        return copy.deepcopy(self.net.state_dict())
    
    
    def get_gradients(self) -> None:
        """Get the gradients of the network (differences between received and trained model)
        
        Parameters
        ----------
        
        Raises
        -------------
            Exception: if the original model was not preserved.
        
        Returns
        -------------
            Oredered_Dict: Gradients of the network.
        """
        assert self.initial_model != None, "Computing gradients require saving initial model first!"
        self.net.to(self.cpu) # Dupming weights on cpu.
        self.initial_model.to(self.cpu)
        weights_trained = self.net.state_dict()
        weights_initial = self.initial_model.state_dict()
        
        self.gradients = OrderedDict.fromkeys(weights_trained.keys(), 0)
        for key in weights_trained:
            self.gradients[key] =  weights_trained[key] - weights_initial[key]

        return self.gradients # Try: to provide original weights, no copies


    def update_weights(
        self, 
        avg_tensors
        ) -> None:
        """Updates the weights of the network stored on client with passed tensors.
        
        Parameters
        ----------
        avg_tensors: Ordered_Dict
            An Ordered Dictionary containing a averaged tensors
        
        Raises
        ------
        Exception: _description_
       
        Returns
        -------
        None
        """
        self.net.load_state_dict(copy.deepcopy(avg_tensors), strict=True)


    def store_model_on_disk(
        self,
        iteration: int,
        path: str
        ) -> None:
        """Saves local model in a .pt format.
        Parameters
        ----------
        Iteration: int
            Current iteration
        Path: str
            Path to the saved repository
        
        Returns: 
        -------
        None
        
        Raises
        -------
            Exception if the model is not initialized it raises an exception
        """
        name = f"node_{self.node_name}_iteration_{iteration}.pt"
        save_path = os.path.join(path, name)
        torch.save(
            self.net.state_dict(),
            save_path,
        )


    def preserve_initial_model(self) -> None:
        """Preserve the initial model provided at the
        end of the turn (necessary for computing gradients,
        when using aggregating methods such as FedOpt).
        
        Parameters
        ----------
        
        Returns
        -------
            Tuple[float, float]: Loss and accuracy on the training set.
        """
        self.initial_model = copy.deepcopy(self.net)


    def train(
        self,
        iteration: int,
        epoch: int
        ) -> tuple[float, torch.tensor]:
        """Train the network and computes loss and accuracy.
        
        Parameters
        ----------
        iterations: int 
            Current iteration
        epoch: int
            Current (local) epoch
        
        Returns
        -------
        None
        """
        criterion = torch.nn.CrossEntropyLoss()
        train_loss = 0
        correct = 0
        total = 0
        # Try: to place a net on the device during the training stage
        self.net.to(self.device)
        self.net.train()
        for _, dic in enumerate(self.trainloader):
            inputs = dic['image']
            targets = dic['label']
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.net.zero_grad() # Zero grading the network                        
            # forward pass, backward pass and optimization
            outputs = self.net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
                    
            # Emptying the cuda_cache
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

        loss = train_loss / len(self.trainloader)
        accuracy = correct / total
        model_logger.info(f"[ITERATION {iteration} | EPOCH {epoch} | NODE {self.node_name}] Training on {self.node_name} results: loss: {loss}, accuracy: {accuracy}")
        
        return (loss, 
                accuracy)
    
    
    def evaluate_model(self) -> tuple[float, float, float, float, float, list]:
        """Validate the network on the local test set.
        
        Parameters
        ----------
        
        Returns
        -------
            Tuple[float, float]: loss and accuracy on the test set.
        """
        # Try: to place net on device directly during the evaluation stage.
        self.net.to(self.device)
        self.net.eval()
        criterion = torch.nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        total = 0
        y_pred = []
        y_true = []
        losses = []
        
        with torch.no_grad():
            for _, dic in enumerate(self.testloader):
                inputs = dic['image']
                targets = dic['label']
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = self.net(inputs)
                
                total += targets.size(0)
                test_loss = criterion(output, targets).item()
                losses.append(test_loss)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.view_as(pred)).sum().item()
                y_pred.append(pred)
                y_true.append(targets)

        test_loss = np.mean(test_loss)
        accuracy = correct / total

        y_true = [item.item() for sublist in y_true for item in sublist]
        y_pred = [item.item() for sublist in y_pred for item in sublist]

        f1score = f1_score(y_true, y_pred, average="macro")
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")

        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        accuracy_per_class = cm.diagonal()

        true_positives = np.diag(cm)
        num_classes = len(list(set(y_true)))

        false_positives = []
        for i in range(num_classes):
            false_positives.append(sum(cm[:,i]) - cm[i,i])

        false_negatives = []
        for i in range(num_classes):
            false_negatives.append(sum(cm[i,:]) - cm[i,i])

        true_negatives = []
        for i in range(num_classes):
            temp = np.delete(cm, i, 0)   # delete ith row
            temp = np.delete(temp, i, 1)  # delete ith column
            true_negatives.append(sum(sum(temp)))

        denominator = [sum(x) for x in zip(false_positives, true_negatives)]
        false_positive_rate = [num/den for num, den in zip(false_positives, denominator)]

        denominator = [sum(x) for x in zip(true_positives, false_negatives)]
        true_positive_rate = [num/den for num, den in zip(true_positives, denominator)]

        # # Emptying the cuda_cache
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        return (
                test_loss,
                accuracy,
                f1score,
                precision,
                recall,
                accuracy_per_class,
                true_positive_rate,
                false_positive_rate
                )


    # def quick_evaluate(self) -> tuple[float, float]:
    #     """Quicker version of the evaluate_model(function) 
    #     Validate the network on the local test set returning only the loss and accuracy.
        
    #     Parameters
    #     ----------
        
    #     Returns
    #     -------
    #         Tuple[float, float]: loss and accuracy on the test set.
    #     """
    #     # Try: to place net on device directly during the evaluation stage.
    #     self.net.to(self.device)
    #     criterion = nn.CrossEntropyLoss()
    #     test_loss = 0
    #     correct = 0
    #     total = 0
        
    #     with torch.no_grad():
    #         for _, dic in enumerate(self.testloader):
    #             inputs = dic['image']
    #             targets = dic['label']
    #             inputs, targets = inputs.to(self.device), targets.to(self.device)
    #             outputs = self.net(inputs)
    #             loss = criterion(outputs, targets)
                
    #             test_loss += loss.item()
    #             _, predicted = outputs.max(1)
    #             total += targets.size(0)
    #             correct += predicted.eq(targets).sum().item()
        
    #     test_loss = test_loss / len(self.testloader)
    #     accuracy = correct / total
                
    #     # # Emptying the cuda_cache
    #     # if torch.cuda.is_available():
    #     #     torch.cuda.empty_cache()

    #     return (
    #         test_loss,
    #         accuracy
    #         )


    def transform_func(
        self,
        data: datasets.arrow_dataset.Dataset
        ) -> None:
        """ Convers datasets.arrow_dataset.Dataset into a PyTorch Tensor
        Parameters
        ----------
        local_dataset: datasets.arrow_dataset.Dataset
            A local dataset that should be loaded into DataLoader
        only_test: bool [default to False]: 
            If true, only a test set will be returned
        
        Returns
        -------------
        None"""
        convert_tensor = transforms.ToTensor()
        data['image'] = [convert_tensor(img) for img in data['image']]
        return data