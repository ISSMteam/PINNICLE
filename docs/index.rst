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

`PINNICLE <https://github.com/ISSMteam/PINNICLE>`_ ((Physics-Informed Neural Networks for Ice and CLimatE) is a Python library for solving ice sheet modeling problems using a unified framework with Physics Informed Neural Networks.
It is designed to integrate physical laws with observational data to solve both forward and inverse problems in glaciology.
The library currently supports stress balance approximations (SSA, MOLHO), mass conservation, and time-dependent simulations, etc. Built on top of `DeepXDE <https://github.com/lululxvi/deepxde>`_, it supports TensorFlow, PyTorch, and JAX backends.


.. note::

   This project is under active development.

.. image:: images/pinn.png

Overview
--------






Physics
-------

- Momentum Conservation (stress balance):
  - Shelfy Stream Approximation (SSA)
  - MOno-Layer Higher-Order (MOLHO) ice flow model  

- Mass Conservation (mass balance):
  - Thickness evolution

- Coupuling:
  - stress balance + mass balance


Data format
-----------

- `ISSM <https://issm.jpl.nasa.gov>`_ ``model()`` type, directly saved from ISSM by ``saveasstruct(md, filename)``
- Scattered data 


Check out the :doc:`Usage` section for further information, including how to install the project.

User guide
----------

.. toctree::
   :maxdepth: 2
   :caption: Usage

   installation

.. toctree::
   :maxdepth: 1

   data
   advanced
   pinnicle_examples


API reference
-------------

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
