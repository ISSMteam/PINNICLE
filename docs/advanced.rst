Advanced Features
=================

PINNICLE is under active development, and several **advanced**/**experimental** features are listed here in alphabetical order:

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


Additional Loss Functions
-------------------------

It is possible to add additional loss functions to the total loss. Here is an example:

.. code-block:: python

   vel_loss = {}
   vel_loss['name'] = "vel MAPE"
   vel_loss['function'] = "MAPE"
   vel_loss['weight'] = 1.0e-6
   hp["additional_loss"] = {"vel":vel_loss}

In this example, we define a ``dict`` which contains:
- ``name``: A user-defined name for the loss function.
- ``function``: A function ID from ``LOSS_DICT`` in `deepxde.losses <https://deepxde.readthedocs.io/en/latest/_modules/deepxde/losses.html#get>`_ or `pinnicle.utils.data_misfit <https://pinnicle.readthedocs.io/en/latest/_modules/pinnicle/utils/data_misfit.html#get>`_. These lists include most commonly used loss functions, such as ``"mean"``, ``"MAE"``, ``"MSE"``, ``"MAPE"``, ``"zero"``, ``"VEL_LOG"``, ``"MEAN_SQUARE_LOG"``, etc. Before writing your own loss function, refer to these lists, as these functions are optimized with the backends.
- ``weight``: the weight of this loss function. 

Finally, add the new ``dict`` to ``hp["additional_loss"]`` with the key indicating the variable to which this loss function should be applied. In the above example, we are adding the mean absolute percentage error of the velocity magnitude to the total loss.

Dummy Equations
---------------

PINNICLE provides `Dummy <https://pinnicle.readthedocs.io/en/latest/api/pinnicle.physics.html#module-pinnicle.physics.dummy>`_ physics, which allows you to train the neural network using data only. Here is an example:

.. code-block:: python

    dummy = {}
    dummy["output"] = ['u', 's', 'C']
    hp["equations"] = {"DUMMY": dummy}

In this example, we define a ``dict`` with a key ``output``, where the value is a list of three output variables. Then, we add this ``dict`` to ``hp['equations']`` with the key ``DUMMY`` (all uppercase). Additionally, you need to provide the data for ``u``, ``s``, and ``C`` in the ``data`` section, similar to other examples. The neural network will then be trained solely with the provided data.

By default, the ``Dummy`` physics already has ``x`` and ``y`` as ``input``. If there is no need to change this, only the ``output`` needs to be defined.


Architecture of the Neural Network
----------------------------------

PINNICLE supports both fully connected neural networks and parallel neural networks. To choose one, simply set ``hp["is_parallel"]`` to ``False`` or ``True``. Currently, PINNICLE only supports parallel networks for each individual output, and all these networks are of the same size.

Another feature of the neural network architecture is that you can set the number of neurons and layers as follows:

.. code-block:: python

   hp["num_neurons"] = 20
   hp["num_layers"] = 4

This configuration creates a network with 4 layers, each containing 20 neurons. Alternatively, you can define the number of neurons in each layer using a list:

.. code-block:: python

   hp["num_neurons"] = [128, 64, 32, 16]

In this case, the number of layers is inferred from the length of the list, creating a network with 4 layers having 128, 64, 32, and 16 neurons respectively.

