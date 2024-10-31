# What is FedJust?

FedJust was projected to be a fully adjustable code base that takes off the need to write boilerplate code for Federated Learning. Its components are fully modifiable building blocks that can be adjusted to your needs in a few lines of code. FedJust was designed to be a minimalist Python boilerplate for running Federated simulations (on a single or multiple GPUs at the same time). It does not aim to work as a Federated framework for actual communication between the devices, as there are already frameworks for doing that, e.g. [Flower](https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html) or part of [PySyft](https://github.com/OpenMined/PySyft). However, the niche of a well-documented and easily adjustable framework for the customization of FL experiments is still open - hence a FedJust tries to deliver the most straightforward boilerplate ecosystem that you may need for performing decentralised simulations in a controlled environment.

# How to use FedJust?

As FedJust was designed to be an easy-to-use boilerplate, it comes in several different ways to use it.
**Python pip**
As FedJust is registered in the Python Package Index (PyPI), it can be downloaded using the pip install command. Simply run:
``pip install fedjust``
**Template Cloning**
This GitHub repository is set up as a public template repository. It means, that it is possible to simply clone it and use it as a standalone repository without actually forking it. The package is built with the help of [Poetry](https://python-poetry.org/). To install the project together with a virtual environment, navigate to the main folder and run:
``poetry install``.
The virtual environment will be installed together with the required dependencies.

## Project layout

The documentation follows the best practice for project documentation as described by Daniele Procida in the [Diátaxis documentation framework](https://diataxis.fr/) and consists of four separate parts:

1. [Tutorials](tutorials.md)
2. [How-To Guides](how-to-guides.md)
3. [Reference](reference.md)
4. [Examples](examples.md)
