Installation
============

Prerequisites
~~~~~~~~~~~~~~

destvi_utils can be installed via PyPI.

conda prerequisites
###################

1. Install Conda. We typically use the Miniconda_ Python distribution. Use Python version >=3.11.

2. Create a new conda environment::

    conda create -n destvi-env python=3.11

3. Activate your environment::

    source activate destvi-env

pip prerequisites:
##################

1. Install Python_, we prefer the `pyenv <https://github.com/pyenv/pyenv/>`_ version management system, along with `pyenv-virtualenv <https://github.com/pyenv/pyenv-virtualenv/>`_.

.. _Miniconda: https://conda.io/miniconda.html
.. _Python: https://www.python.org/downloads/
.. _PyTorch: http://pytorch.org

destvi_utils installation
~~~~~~~~~~~~~~~~~~~~~~~

Install destvi_utils in one of the following ways:

Through **pip**::

    pip install git+https://github.com/yoseflab/destvi_utils.git

Nightly version - clone this repo and run::

    pip install .

For development - clone this repo and run::

    pip install -e .[dev,docs]
