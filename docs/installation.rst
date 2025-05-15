.. _installation:

Install and Setup
=================

Installation
------------

The easiest way to install PINNICLE is directly from PyPI:

.. code-block:: bash

   pip install pinnicle

This will install the core package and automatically resolve most dependencies.

Alternatively, if you'd like to clone and modify the source code:

.. code-block:: bash

   git clone https://github.com/ISSMteam/PINNICLE.git
   cd PINNICLE
   pip install -e .

PINNICLE requires **Python ≥ 3.9** and supports one of the following machine learning backends:

- **TensorFlow** ≥ 2.11.0  
  (Optionally with `tensorflow-probability` ≥ 0.19.0)

- **PyTorch** ≥ 1.9.0

- **JAX** (plus `Flax`, `Optax`)

Required Dependencies
---------------------

If you install via pip, most dependencies are handled automatically. For manual setup, make sure the following libraries are installed:

- `DeepXDE <https://github.com/lululxvi/deepxde>`_
- `mat7.3 <https://github.com/skjerns/mat7.3>`_
- `Matplotlib <https://matplotlib.org>`_
- `NumPy <http://www.numpy.org>`_
- `pandas <https://pandas.pydata.org>`_
- `scikit-learn <https://scikit-learn.org>`_
- `SciPy <https://www.scipy.org>`_


.. _backends:

Working with Different Backends
-------------------------------

PINNICLE relies on `DeepXDE <https://github.com/lululxvi/deepxde>`_ to interface with your preferred machine learning backend.

Option 1: Environment Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run PINNICLE with your selected backend:

.. code-block:: bash

   DDE_BACKEND=tensorflow python your_script.py

You can also export the backend globally (Linux/macOS):

.. code-block:: bash

   export DDE_BACKEND=pytorch

Valid options: `tensorflow`, `pytorch`, `jax`

Option 2: Config File
~~~~~~~~~~~~~~~~~~~~~

Create or edit ``~/.deepxde/config.json``:

.. code-block:: json

   {
     "backend": "tensorflow"
   }

This sets the default backend for all runs.

Optional: GPU Setup
-------------------

To run large models efficiently, install the GPU-enabled version of your backend.

- `TensorFlow GPU Guide <https://www.tensorflow.org/install/gpu>`_
- `PyTorch CUDA Installation <https://pytorch.org/get-started/locally/>`_
- `JAX GPU Setup <https://github.com/google/jax#installation>`_
