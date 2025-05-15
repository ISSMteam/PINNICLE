.. _weights:

Loss Function Weights
=====================

Weights in PINNICLE control the relative contribution of each term in the composite loss function. Because physical variables in glaciology differ in units and magnitudes (e.g., velocity vs. stress vs. elevation), assigning appropriate weights is critical to ensure balanced training and model convergence.

Overview
--------

The total loss in PINNICLE is composed of multiple terms:

.. math::

   L = \sum_i \gamma_i \cdot L_i

Where:
- :math:`L_i` is the loss for a specific variable or physics term
- :math:`\gamma_i` is the corresponding weight

These weights scale the magnitude of each loss component so that no single variable dominates or becomes numerically insignificant.

Default Weights
---------------

PINNICLE uses empirically derived default weights that normalize each term to be approximately order 1 based on typical values in glaciological applications:

.. list-table:: Default Weights and Typical Values
   :widths: 30 30 30 40
   :header-rows: 1

   * - **Component**
     - **Weight Symbol**
     - **Default Value**
     - **Typical Variable Value**
   * - Ice velocity
     - :math:`\gamma_u`
     - :math:`10^{-8} \cdot T^2`
     - :math:`10^4\ \text{m/yr}`
   * - Ice thickness
     - :math:`\gamma_H`
     - :math:`10^{-6}`
     - :math:`10^3\ \text{m}`
   * - Surface elevation
     - :math:`\gamma_s`
     - :math:`10^{-6}`
     - :math:`10^3\ \text{m}`
   * - Mass balance
     - :math:`\gamma_a`
     - :math:`T^2`
     - :math:`1\ \text{m/yr}`
   * - Friction coefficient
     - :math:`\gamma_C`
     - :math:`10^{-8}`
     - :math:`10^4\ \text{Pa}^{1/2}\ \text{m}^{-1/6}\ \text{s}^{1/6}`
   * - Rheology pre-factor
     - :math:`\gamma_B`
     - :math:`10^{-18}`
     - :math:`10^9\ \text{Pa·s}^{1/3}`
   * - Driving stress (PDE)
     - :math:`\gamma_{\tau}`
     - :math:`10^{-10}`
     - :math:`10^5\ \text{Pa}`
   * - Dynamic thinning
     - :math:`\gamma_{H/t}`
     - :math:`10^{10}`
     - :math:`10^3\ \text{m/year}`

:math:`T = 31536000` seconds in one year, used to convert time-based quantities to SI units.

Customization
-------------

You can override any default weight in your configuration dictionary:

.. code-block:: python

   hp["weights"] = {
       "u": 1e-8 * (31536000 ** 2),
       "H": 1e-6,
       "C": 1e-9,
       "tau": 1e-10
   }

Each key corresponds to the variable name or PDE term used in the experiment.

When to Adjust Weights
-----------------------

Consider tuning weights if:

- One loss term dominates training (e.g., velocity converges but thickness doesn't)
- A variable is noisy or has sparse data (downweight it)
- You’re adding a new physical constraint and need to balance it with existing terms

Use lower weights to **reduce influence**, higher weights to **emphasize** a particular term.

Tips
----

- Always keep weights dimensionally consistent
- Try matching the order of magnitude of each loss component
- Monitor individual loss terms during training for imbalance

Advanced: Auto-balancing
------------------------

Though PINNICLE uses fixed weights by default, you can implement custom logic for **dynamic reweighting**:

- Normalize loss terms during training
- Use adaptive schemes (e.g., SoftAdapt, uncertainty-based weighting)
- Log loss values and manually adjust over multiple runs

Applications
------------

- **Example 1**: Uses default weights for velocity, elevation, and friction inversion
- **Example 2**: Adds custom weight for ice rheology pre-factor
- **Example 3**: Applies a large weight to dynamic thinning to prioritize PDE accuracy

See the `Examples <examples.html>`_ page to view how weights influence different training scenarios.

