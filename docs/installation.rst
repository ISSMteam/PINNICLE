.. _installation:

Install and Setup
=================

Preinstall ML packages
----------------------

PINNICLE requires **Python ≥ 3.9** and supports several different  machine learning backends. In order to install PINNICLE properly, you will need to install one of the following packages:

- **TensorFlow** ≥ 2.11.0  
  (Optionally with `tensorflow-probability` ≥ 0.19.0)

.. code-block:: bash

   pip install tensorflow>=2.11.0 tensorflow-probability[tf]>=0.19.0

- **PyTorch** ≥ 1.9.0

.. code-block:: bash

  pip install torch torchvision torchaudio

- **JAX** (plus `Flax`, `Optax`)

.. code-block:: bash

  pip install jax flax optax


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

Valid options: ``tensorflow``, ``pytorch``, ``jax``

Option 2: Config File
~~~~~~~~~~~~~~~~~~~~~

Create or edit ``~/.deepxde/config.json``:

.. code-block:: json

   {
     "backend": "tensorflow"
   }

This sets the default backend for all runs.

Or, you can use ``python -m deepxde.backend.set_default_backend BACKEND`` to set the default backend.


Run with Docker
---------------

PINNICLE can also be run in a fully containerized environment using Docker. This is ideal for avoiding dependency conflicts or running the software in a reproducible HPC/cloud environment.
The `PINNICLE Docker image <https://hub.docker.com/r/chenggongdartmouth/pinnicle>`_ contains all the required packages to run PINNICLE with GPU support. 

.. code-block:: bash

   apptainer build --nv set_your_own_name docker://chenggongdartmouth/pinnicle:v0.3

See the details instructions about `how to set up PINNICLE Docker <https://holly-riverbed-43f.notion.site>`_

Optional: GPU Setup
-------------------

To run large models efficiently, install the GPU-enabled version of your backend.

- `TensorFlow GPU Guide <https://www.tensorflow.org/install/gpu>`_
- `PyTorch CUDA Installation <https://pytorch.org/get-started/locally/>`_
- `JAX GPU Setup <https://github.com/google/jax#installation>`_
