.. _learning_rate_decay:

Learning Rate Decay
====================

The learning rate is a critical hyperparameter that controls how quickly a neural network updates its weights during training. In PINNICLE, learning rate decay can be used to **improve convergence**, **stabilize optimization**, and **avoid overshooting** when solving complex PDE-constrained problems.

Overview
--------

A high initial learning rate helps the model make fast progress during early training, but can cause instability later. Learning rate decay reduces the learning rate as training proceeds, allowing the model to fine-tune its solution.

PINNICLE supports a variety of learning rate scheduling strategies through the backend (e.g., TensorFlow, PyTorch), and provides built-in options for step-based decay and exponential decay.

Built-In Scheduler
------------------

To enable learning rate decay in PINNICLE, you can add the following keys to your configuration dictionary:

.. code-block:: python

   hp["lr"] = 1e-3                    # Initial learning rate
   hp["lr_decay"] = True             # Enable learning rate decay
   hp["lr_decay_type"] = "exponential"  # or "step"
   hp["lr_decay_steps"] = 20000      # Steps after which to apply decay
   hp["lr_decay_rate"] = 0.9         # Decay multiplier

Decay Types
-----------

1. **Exponential Decay**

   Reduces the learning rate by a fixed fraction every `lr_decay_steps`:

   .. math::

      \eta_k = \eta_0 \cdot \text{decay\_rate}^{\left\lfloor \frac{k}{\text{decay\_steps}} \right\rfloor}

   where:
   - :math:`\eta_0` is the initial learning rate
   - :math:`\eta_k` is the learning rate at step :math:`k`

2. **Step Decay**

   Drops the learning rate at fixed intervals:

   .. code-block:: python

      if step % decay_steps == 0:
          lr *= decay_rate

   This results in a staircase-style schedule, often used in computer vision or when training plateaus.

Manual Adjustment
-----------------

You can also manually schedule learning rate changes during long training runs. For example:

.. code-block:: python

   if step == 100000:
       optimizer.lr.assign(1e-4)
   elif step == 200000:
       optimizer.lr.assign(1e-5)

This is useful for inverse problems where fine-tuning is needed in later stages.

Visualization
-------------

To monitor learning rate evolution, log or plot the current learning rate at each epoch:

.. code-block:: python

   print("Epoch:", epoch, "Learning rate:", optimizer.lr.numpy())

When to Use
-----------

Use learning rate decay when:
- Training plateaus or diverges in later epochs
- The loss decreases quickly at first, then oscillates
- You're solving complex inverse problems (e.g., friction + rheology)

Best Practices
--------------

- Start with a relatively large initial learning rate (e.g., 1e-3)
- Apply exponential decay for smoother convergence
- Decay slower (e.g., every 20,000–50,000 steps) for forward problems
- Decay faster for unstable or ill-posed inverse problems

Example: Helheim Inverse Problem
--------------------------------

In **Example 1**, PINNICLE uses:

.. code-block:: python

   hp["lr"] = 1e-3
   hp["lr_decay"] = True
   hp["lr_decay_type"] = "exponential"
   hp["lr_decay_steps"] = 20000
   hp["lr_decay_rate"] = 0.9

This setup allows rapid convergence early on, followed by gradual refinement of the basal friction field.

Advanced Options
----------------

You can also integrate external schedulers from PyTorch (`torch.optim.lr_scheduler`) or TensorFlow (`tf.keras.optimizers.schedules`) by customizing the training loop.

For JAX backends, custom decay functions can be passed through the optimizer builder.

References
----------

- Bengio, Y. (2012). "Practical recommendations for gradient-based training of deep architectures"
- Karniadakis et al. (2021). "Physics-informed machine learning"
- TensorFlow Docs: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules


