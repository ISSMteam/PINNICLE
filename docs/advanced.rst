Advanced Features
=================

.. note::

   PINNICLE is under active development, and several **advanced**/**experimental** features are listed here:

Mini-batch
----------

PINNICLE supports mini-batch training when using the ``pytorch`` backend. In the current implementation, mini-batches are resampled at every epoch to enhance convergence stability.
To activate mini-batch training, simply add the following setting to the hyperparameter dictionary. PINNICLE will automatically generate mini-batches from the data specified in ``hp["data"]``.

.. code-block:: python
   
   hp["mini_batch"] = mini_batch

- Avoid setting ``mini_batch`` too large, as this can lead to inefficiencies and excessive memory usage.
- Avoid setting ``mini_batch`` too small, as this can result in wasted computational resources due to the increased overhead of frequent resampling.

Based on our numerical experiments, we recommend choosing a batch size in the range of 500 to 10,000.


Resample collocation points
---------------------------

PINNICLE supports the resampling of collocation points during training. To activate this feature, set the following in your hyperparameter dictionary:

.. code-block:: python

   hp["period"] = 100

This will instruct PINNICLE to resample the collocation points every 100 epochs.



Load Settings from Previous Experiment
--------------------------------------

Each PINNICLE experiment will save the entire settings to the folder defined in ``hp["save_path"]`` if ``hp["is_save"]=True``. The settings are saved into a ``params.json`` file containing all the non-default settings from ``hp``.

To repeat a new experiment with the same settings as in some ``folder_path``, you can easily load the settings by:

.. code-block:: python

   experiment = pinnicle.PINN(loadFrom=folder_path)

However, if you want to run the experiment and save the results in a new folder, you will need to set the ``"save_path"`` and update the experiment:

.. code-block:: python

   experiment.update_parameters({"save_path": new_folder})

Then, compile and train the model.


Load Weights from Previously Trained Neural Network
---------------------------------------------------

PINNICLE can also load the weights from a previous experiment. You can either load the settings from a saved experiment or set up a completely new one. After that, you can load the weights of the neural network from the ``folder_path`` which contains the entire experiment (including ``params.json``, a folder called ``pinn``, and some figures).

.. code-block:: python

   experiment.load_model(path=folder_path)

After loading the weights, compiling and training are the same as for other experiments.


Dummy Equations
---------------

PINNICLE provides :py:mod:`pinnicle.physics.dummy` physics, which allows you to train the neural network using data only. Here is an example:

.. code-block:: python

    dummy = {}
    dummy["output"] = ['u', 's', 'C']
    hp["equations"] = {"DUMMY": dummy}

In this example, we define a ``dict`` with a key ``output``, where the value is a list of three output variables. Then, we add this ``dict`` to ``hp['equations']`` with the key ``DUMMY`` (all uppercase). Additionally, you need to provide the data for ``u``, ``s``, and ``C`` in the ``data`` section, similar to other examples. The neural network will then be trained solely with the provided data.

By default, the ``Dummy`` physics already has ``x`` and ``y`` as ``input``. If there is no need to change this, only the ``output`` needs to be defined.

