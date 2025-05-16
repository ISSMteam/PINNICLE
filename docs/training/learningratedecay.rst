.. _learning_rate_decay:

Learning Rate Decay
====================

The learning rate is a critical hyperparameter that controls how quickly a neural network updates its weights during training. In PINNICLE, learning rate decay can be used to **improve convergence**, **stabilize optimization**, and **avoid overshooting** when solving complex PDE-constrained problems.


A high initial learning rate helps the model make fast progress during early training, but can cause instability later. Learning rate decay reduces the learning rate as training proceeds, allowing the model to fine-tune its solution.

Decay Strategy
--------------

PINNICLE uses **inverse time decay** as its default learning rate scheduling strategy through the DeepXDE backend. This method gradually decreases the learning rate over time to ensure stable convergence, especially during the later stages of training.


The decay follows the formula:


.. math::

   \eta(k) = \frac{\eta_0}{1 + \text{decay_rate} \cdot \left\lfloor \frac{k}{\text{decay_step}} \right\rfloor}

where:

- :math:`\eta_0` is the initial learning rate
- :math:`k` is the current training step
- :math:`\text{decay_rate}` is a small positive constant
- :math:`\text{decay_step}` is the number of steps after which decay is applied

Configuration Example
---------------------

To enable learning rate decay in PINNICLE, you can add the following keys to your configuration dictionary:

.. code-block:: python

   hp["learning_rate"] = 1e-3
   hp["decaay_steps"] = 10000
   hp["decay_rate"] = 0.5


At step 10,000, the learning rate would be:

.. math::

   \eta(10000) = \frac{10^{-3}}{1 + 0.5 \cdot \left\lfloor \frac{10000}{10000} \right\rfloor} = 5\times 10^{-4}

At step 20,000:

.. math::

   \eta(20000) = \frac{10^{-3}}{1 + 0.5 \cdot 2} = 3.33\times10^{-4}

References
----------

- Bengio, Y. (2012). "Practical recommendations for gradient-based training of deep architectures"
- Karniadakis et al. (2021). "Physics-informed machine learning"
- DeepXDE Docs: https://deepxde.readthedocs.io/en/latest/modules/deepxde.html#deepxde.model.Model
