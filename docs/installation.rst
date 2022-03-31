Installation
============

Prerequisites
~~~~~~~~~~~~~~

my_package can be installed via PyPI.

conda prerequisites
###################

1. Install Conda. We typically use the Miniconda_ Python distribution. Use Python version >=3.7.

2. Create a new conda environment::

    conda create -n scvi-env python=3.7

3. Activate your environment::

    source activate scvi-env

pip prerequisites:
##################

1. Install Python_, we prefer the `pyenv <https://github.com/pyenv/pyenv/>`_ version management system, along with `pyenv-virtualenv <https://github.com/pyenv/pyenv-virtualenv/>`_.

.. _Miniconda: https://conda.io/miniconda.html
.. _Python: https://www.python.org/downloads/
.. _PyTorch: http://pytorch.org

destvi-utils installation
~~~~~~~~~~~~~~~~~~~~~~~

Install destvi-utils in one of the following ways:

Through **pip**::

    pip install destvi-utils

Through pip with packages to run notebooks. This installs scanpy, etc.::

    pip install destvi-utils[tutorials]

Nightly version - clone this repo and run::

    pip install .

For development - clone this repo and run::

    pip install -e .[dev,docs]
