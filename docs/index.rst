PINNICLE
========

`PINNICLE <https://github.com/ISSMteam/PINNICLE>`_ is a Python library for solving ice sheet modeling problems using a unified framework with Physics Informed Neural Networks


.. note::

   This project is under active development.


.. image:: images/pinn.png


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
