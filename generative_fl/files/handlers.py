import csv
import os


def save_dict_ascsv(
    data: dict,
    save_path: str):
    """Attaches models of the nodes to the simulation instance.
        
    Parameters
    ----------
    data: dict
        Dictionary containing the data. Data in the dictionary will be transformed
        and saved under fieldnames formed by keys.
    save_path: str
        A full save path for preserving the dictionary. Must end with *.csv file extension.
    Returns
    -------
    None
    """
    with open(save_path, "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=next(iter(data.values())).keys())
        # Checks if the file already exists, otherwise writer header
        if os.path.getsize(save_path) == 0:
            writer.writeheader()
        for row in data.values():
            writer.writerow(row)


def save_nested_dict_ascsv(
    data: dict[dict],
    save_path: str):
    """Attaches models of the nodes to the simulation instance.
        
    Parameters
    ----------
    data: dict[dict]
        Nested dict containing the data. Note that the external key will not be saved, only the 
        data in the internal dictionary will be transformed to fieldnames and saved.
    save_path: str
        A full save path for preserving the dictionary. Must end with *.csv file extension.
    Returns
    -------
    None
    """
    with open(save_path, "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=next(iter(data.values())).keys())
        # Checks if the file already exists, otherwise writer header
        if os.path.getsize(save_path) == 0:
            writer.writeheader()
        for row in data.values():
            writer.writerow(row)