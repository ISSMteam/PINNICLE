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
     - :math:`10^{-8} \cdot 31536000^2`
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


Customization
-------------

You can override these defaults in your hyper-parameter configuration by setting it in each equation, for example:

.. code::

   hp["equations"] = {
      "SSA": {
         "pde_weight": [1.0e-8, 1.0e-8];
      }
   }


or
.. code::

   hp["equations"] = {
      "SSA": {
         "data_weight": [1.0e-8*31536000**2.0, 1.0e-8*31536000**2.0, 1.0e-6, 1.0e-6, 1.0e-8],
      }
   }


- ``"pde_weight"`` follows the same order as the ``residuals`` defined in the PDEs
- ``"data_weight"`` follows the same order as the ``output`` defined in the PDEs

Use lower weights to **reduce influence**, higher weights to **emphasize** a particular term.


See the :ref:`examples` page to view how weights influence different training scenarios.
