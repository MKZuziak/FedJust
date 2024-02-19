import logging

def orchestrator_logger():
    # Creating a head logger for the orchestrator
    # Level and format
    orchestrator_logger = logging.getLogger("orchestrator_logger")
    orchestrator_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Print-out to console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    # Save to file handler
    chw = logging.FileHandler("orchestrator_logs.txt")
    chw.setLevel(logging.DEBUG)
    chw.setFormatter(formatter)
    # Attaching handlers to the logger
    orchestrator_logger.addHandler(ch)
    orchestrator_logger.addHandler(chw)
    orchestrator_logger.propagate = False
    return orchestrator_logger


def node_logger():
    # Creating a head logger for the nodes
    # Level and format
    node_logger = logging.getLogger("node_logger")
    node_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Print-out to console handler
    zh = logging.StreamHandler()
    zh.setLevel(logging.DEBUG)
    zh.setFormatter(formatter)
    # Save to file handler
    zhw = logging.FileHandler("nodes_logs.txt")
    zhw.setLevel(logging.DEBUG)
    zhw.setFormatter(formatter)
    # Attaching handlers to the logger
    node_logger.addHandler(zh)
    node_logger.addHandler(zhw)
    node_logger.propagate = False
    return node_logger


def model_logger():
    # Creating a head loger for the models
    # Level and format
    model_logger = logging.getLogger("model_logger")
    model_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Print-out to console handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    # Save to file handler
    shw = logging.FileHandler("model_logs.txt")
    shw.setLevel(logging.DEBUG)
    shw.setFormatter(formatter)
    # Attaching handlers to the logger
    model_logger.addHandler(sh)
    model_logger.addHandler(shw)
    model_logger.propagate = False
    return model_logger