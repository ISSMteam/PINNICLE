.. _loss_functions:

Loss Functions
==============

In PINNICLE, the loss function is the core of the training process. It integrates both **data misfit** and **physical residuals**, enabling the neural network to satisfy observational constraints and underlying governing equations simultaneously.

General Form
------------

The total loss function is composed of two main parts:

.. math::

   L = L_\phi + L_d

where:

- :math:`L_\phi` is the physics loss, computed from the residuals of the governing PDEs (e.g., momentum, mass conservation).
- :math:`L_d` is the data misfit term, measuring the error between predicted and observed data.

Both components can include multiple variables, each with a customizable weight.

Physics-Based Loss
------------------

This term ensures that the neural network outputs satisfy the governing PDEs. It is computed at a set of randomly choosen collocation points in the domain defined in ``hp["shapefile"]``:

.. math::

   L_\phi = \frac{1}{N_\phi} \sum_{i=1}^{N_\phi}|\mathcal{R}|^2

where:

- :math:`N_\phi` is the number of collocation points defined in ``hp["num_collocation_points"]``
- :math:`\mathcal{R}` is the PDE residual (e.g., :ref:`momentum_conservation`, :ref:`mass_conservation`)

Each PDE residual is evaluated using automatic differentiation on the network outputs.

Data Misfit
-----------

This term compares predicted quantities to ground-truth data using predefined error metrics. PINNICLE supports the following data misfit functions:

.. list-table::
   :widths: 30 65
   :header-rows: 1

   * - **Key**
     - **Formula**
   * - ``"MAE"``
     - :math:`\frac{1}{n}\sum_{i=1}^N|d_i-\hat{d}_i|`
   * - ``"MSE"``
     - :math:`\frac{1}{n} \sum_{i=1}^N (d_i - \hat{d}_i)^2`
   * - ``"MAPE"``
     - :math:`\frac{100}{n} \sum_{i=1}^N \left| \frac{d_i - \hat{d}_i}{d_i} \right|`
   * - ``"MEAN_SQUARE_LOG"``
     - :math:`\frac{1}{n} \sum_{i=1}^N \left( \log(|d_i| + 1) - \log(|\hat{d}_i| + 1) \right)^2`
   * - ``"VEL_LOG"``
     - :math:`\frac{1}{n} \sum_{i=1}^N \log(|\hat{u}_i| + \epsilon) / \log(|u_i| + \epsilon)`

Here, :math:`d_i` is the ground truth, :math:`\hat{d}_i` is the predicted value, and :math:`\epsilon` is a small constant to avoid divide-by-zero.

These metrics are selected automatically based on variable type or can be customized by the user.

Weighting
---------

Different terms in the total loss function are scaled to ensure balanced contribution. Default weights are described in :ref:`weights`.

You can override these defaults in your hyper-parameter configuration by setting it in each equation, for example:

.. code::

   hp["equations"] = {
      "SSA": {
         "data_weight": [1.0e-8*31536000**2.0, 1.0e-8*31536000**2.0, 1.0e-6, 1.0e-6, 1.0e-8],
         "pde_weights": [1e-8, 1e-8]
      }
   }

- ``"data_weight"`` follows the same order as the ``output`` defined in the PDEs
- ``"pde_weights"`` has the same order as the ``residuals``


There is also a global level parameter ``hp["loss_weights"]``, this contains all the weights following the order: first all PDE residualsm and then the outputs.

Loss Function Example
---------------------

A composite loss function may look like:

.. math::

   L = \gamma_u \cdot L_u + \gamma_H \cdot L_H + \gamma_s \cdot L_s + \gamma_C \cdot L_C + \gamma_\tau \cdot L_\phi

Each term corresponds to:

- :math:`L_u`: Velocity misfit
- :math:`L_H`: Ice thickness misfit
- :math:`L_s`: Surface elevation misfit
- :math:`L_C`: Friction coefficient constraint
- :math:`L_\phi`: Momentum PDE residual

Additional Loss Functions
-------------------------

It is also possible to add additional loss functions to the total loss, other than the two types described above. One example is to use velocity magnitude in the data misfit:

.. code-block:: python

   vel_loss = {}
   vel_loss['name'] = "vel MAPE"
   vel_loss['function'] = "MAPE"
   vel_loss['weight'] = 1.0e-6
   hp["additional_loss"] = {"vel":vel_loss}

- ``name``: A user-defined name for the loss function.
- ``function``: A function ID from ``LOSS_DICT`` in `deepxde.losses <https://deepxde.readthedocs.io/en/latest/_modules/deepxde/losses.html#get>`_ or :py:mod:`pinnicle.utils.data_misfit`. 
  These lists include most commonly used loss functions, such as ``"mean"``, ``"MAE"``, ``"MSE"``, ``"MAPE"``, ``"zero"``, ``"VEL_LOG"``, ``"MEAN_SQUARE_LOG"``, etc. Before writing your own loss function, refer to these lists, as these functions are optimized with the backends.
- ``weight``: the weight of this loss function.

Finally, add the new ``dict`` to ``hp["additional_loss"]`` with the key indicating the variable to which this loss function should be applied. In the above example, we are adding the mean absolute percentage error of the velocity magnitude to the total loss.


Applications
------------

See the :ref:`examples` section for usage in full experiments.

