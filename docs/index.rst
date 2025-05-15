PINNICLE
========

.. image:: https://github.com/ISSMteam/PINNICLE/actions/workflows/CI.yml/badge.svg
  :target: https://github.com/ISSMteam/PINNICLE/actions/workflows/CI.yml
.. image:: https://codecov.io/gh/ISSMteam/PINNICLE/graph/badge.svg?token=S7REK0IKJH
  :target: https://codecov.io/gh/ISSMteam/PINNICLE
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.15178900.svg
   :target: https://doi.org/10.5281/zenodo.15178900
.. image:: https://badge.fury.io/py/PINNICLE.svg
    :target: https://badge.fury.io/py/PINNICLE

`PINNICLE <https://github.com/ISSMteam/PINNICLE>`_ (Physics-Informed Neural Networks for Ice and CLimatE) is a Python library for solving ice sheet modeling problems using a unified framework with Physics Informed Neural Networks.
It is designed to integrate physical laws with observational data to solve both forward and inverse problems in glaciology.
The library currently supports stress balance approximations, mass conservation, and time-dependent simulations, etc. Built on top of `DeepXDE <https://github.com/lululxvi/deepxde>`_, it supports TensorFlow, PyTorch, and JAX backends.


.. note::

   This project is under active development.

.. image:: images/pinn.png


.. toctree::
   :maxdepth: 1
   :caption: Physics

   physics/mass
   physics/momentum


.. toctree::
   :maxdepth: 1
   :caption: Data

   data/issm
   data/scatter
   data/h5


.. toctree::
   :maxdepth: 1
   :caption: Training

   training/nn
   training/fft
   training/lossfunctions
   training/weights
   training/learningratedecay


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation

.. toctree::
   :maxdepth: 1

   advanced
   pinnicle_examples


.. toctree::
   :maxdepth: 2
   :caption: API

   api/pinnicle
   api/pinnicle.domain
   api/pinnicle.modeldata
   api/pinnicle.nn
   api/pinnicle.physics
   api/pinnicle.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
