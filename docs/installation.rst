Install and Setup
=================


Installation
------------

PINNICLE support one of the following backends

- TensorFlow 2.x: `TensorFlow <https://www.tensorflow.org>`_>=2.11.0, `TensorFlow Probability <https://www.tensorflow.org/probability>`_>=0.19.0

In the future, PINNICLE will support the following backends:

- TensorFlow 1.x: `TensorFlow <https://www.tensorflow.org>`_>=2.7.0
- PyTorch: `PyTorch <https://pytorch.org>`_>=1.9.0
- JAX: `JAX <https://jax.readthedocs.io>`_, `Flax <https://flax.readthedocs.io>`_, `Optax <https://optax.readthedocs.io>`_
- PaddlePaddle: `PaddlePaddle <https://www.paddlepaddle.org.cn/en>`_>=2.6.0

- To install and use the package, you should clone the folder to your local machine and put it along with your project scripts::

    $ git clone https://github.com/ISSMteam/PINNICLE.git

* Other dependencies

    - `DeepXDE <https://github.com/lululxvi/deepxde>`_
    - `mat7.3 <https://github.com/skjerns/mat7.3>`_
    - `Matplotlib <https://matplotlib.org>`_
    - `NumPy <http://www.numpy.org>`_
    - `pandas <https://pandas.pydata.org>`_
    - `scikit-learn <https://scikit-learn.org>`_
    - `SciPy <https://www.scipy.org>`_


Working with different backends
-------------------------------

PINNICLE uses `DeepXDE <https://github.com/lululxvi/deepxde>`_ to support different backends. To choose the backend, use the following options as in DeepXDE

* Use the ``DDE_BACKEND`` environment variable:

    - Use  ``DDE_BACKEND=tensorflow python pinn.py`` to specify the backend

    - Or set the global environment variable ``DDE_BACKEND``. For instance, in Linux, use ``export DDE_BACKEND=BACKEND``

* Modify the ``config.json`` file under "~/.deepxde" to be ``{"backend": "tensorflow"}``


