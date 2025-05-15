.. _loss_functions:

Loss Functions
==============

In PINNICLE, the loss function is the core of the training process. It integrates both **data misfit** and **physical residuals**, enabling the neural network to satisfy observational constraints and underlying governing equations simultaneously.

General Form
------------

The total loss function is composed of two main parts:

.. math::

   L = L_d + L_\phi

where:

- :math:`L_d` is the data misfit term, measuring the error between predicted and observed data.
- :math:`L_\phi` is the physics loss, computed from the residuals of the governing PDEs (e.g., momentum, mass conservation).

Both components can include multiple variables, each with a customizable weight.

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

Physics-Based Loss
------------------

This term ensures that the neural network outputs satisfy the governing PDEs. It is computed at a set of randomly choosen collocation points in the domain defined in ``hp["shapefile"]``:

.. math::

   L_\phi = \frac{1}{N_\phi} \sum_{i=1}^{N_\phi}|\mathcal{R}(\hat{u})|^2

where:

- :math:`N_\phi` is the number of collocation points defined in ``hp["num_collocation_points"]``
- :math:`\mathcal{R}` is the PDE residual (e.g., :ref:`momentum_conservation`, :ref:`mass_conservation`)

Each PDE residual is evaluated using automatic differentiation on the network outputs.

Weighting
---------

Different terms in the total loss function are scaled to ensure balanced contribution. Default weights are based on empirical values from over 15,000 experiments and are listed below:

:math:`T = 31536000` is the number of seconds in a year.

You can override these defaults in your configuration dictionary or by modifying the model class.

Loss Function Example
---------------------

A composite loss function in an inverse problem may look like:

.. math::

   L = \gamma_u \cdot L_u + \gamma_H \cdot L_H + \gamma_s \cdot L_s + \gamma_C \cdot L_C + \gamma_\tau \cdot L_\phi

Each term corresponds to:
- :math:`L_u`: Velocity misfit
- :math:`L_H`: Ice thickness misfit
- :math:`L_s`: Surface elevation misfit
- :math:`L_C`: Friction coefficient constraint
- :math:`L_\phi`: Momentum PDE residual

Custom Loss Functions
---------------------

Advanced users can define custom data misfit functions by extending the loss module:

.. code-block:: python

   def my_custom_loss(true, pred):
       return tf.reduce_mean(tf.abs(true - pred) ** 1.5)

You can then assign this to a specific variable using:

.. code-block:: python

   hp["custom_loss"] = {"s": my_custom_loss}

Applications
------------

- **Example 1**: Combines velocity, elevation, and thickness misfits with SSA residuals
- **Example 2**: Adds constraints for both basal friction and rheology
- **Example 3**: Includes time-dependent mass transport and dynamic thinning terms

See the `Examples <examples.html>`_ section for usage in full experiments.

