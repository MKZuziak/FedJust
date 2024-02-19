import csv
import os

from FedJust.model.federated_model import FederatedModel
from FedJust.node.federated_node import FederatedNode


def evaluate_model(
    iteration: int, 
    model: FederatedModel,
    save_path: str,
    logger = None,
    log_to_screen: bool = False
    ) -> None:
    """Used to save the model metrics.
    
    Parameters
    ----------
    iteration: int 
        Current iteration of the training.
    model: FederatedModel
        FederatedModel to be evaluated
    saving_path: str (default to None)
        The saving path of the csv file, must end with the .csv extension.
    logger: Logger (default to None)
        Logger object that we want to use to handle the logs.
    log_to_screen: bool (default to False)
        Boolean flag whether we want to log the results to the screen.
    
    Returns
    -------
        None"""
    try:
        (
            loss,
            accuracy,
            fscore,
            precision,
            recall,
            test_accuracy_per_class,
            true_positive_rate,
            false_positive_rate
        ) = model.evaluate_model()
        metrics = {"epoch": iteration,
                    "node": model.node_name,
                    "loss":loss, 
                    "accuracy": accuracy, 
                    "fscore": fscore, 
                    "precision": precision,
                    "recall": recall, 
                    "test_accuracy_per_class": test_accuracy_per_class, 
                    "true_positive_rate": true_positive_rate,
                    "false_positive_rate": false_positive_rate,
                    "epoch": iteration}
        if log_to_screen == True:
            pass
            #logger.info(f"Evaluating model after iteration {iteration} on node {model.node_name}. Results: {metrics}")
    except Exception as e:
        logger.warning(f"Unable to compute metrics. {e}")
    path = os.path.join(save_path)
    with open(path, 'a+', newline='') as saved_file:
            writer = csv.DictWriter(saved_file, list(metrics.keys()))
            # If the file does not exist, it will create it and write the header.
            if os.path.getsize(path) == 0:
                writer.writeheader()
            writer.writerow(metrics)


def automatic_node_evaluation(
    iteration: int, 
    nodes: dict[int: FederatedNode],
    save_path: str,
    logger = None,
    log_to_screen: bool = False
    ) -> None:
    """Used to automatically evaluate a set of provided node and preserve metrics in the indicated 
    directory.
    
    Parameters
    ----------
    iteration: int 
        Current iteration of the training.
    nodes: dict[int: FederatedNode]
        Dictionary containing nodes to be evaluated.
    saving_path: str (default to None)
        The saving path of the csv file, must end with the .csv extension.
    logger: Logger (default to None)
        Logger object that we want to use to handle the logs.
    log_to_screen: bool (default to False)
        Boolean flag whether we want to log the results to the screen.
    
    Returns
    -------
        None"""
    for node in nodes.values():
        evaluate_model(
            iteration=iteration,
            model=node.model,
            save_path=save_path,
            logger=logger,
            log_to_screen=log_to_screen
        )