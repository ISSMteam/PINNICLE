.. _mass_conservation:

Mass Conservation
=================

Mass conservation is a fundamental principle in ice sheet modeling. In PINNICLE, this equation governs how ice thickness evolves over time in response to ice flow and surface mass balance processes.

Equation
--------

PINNICLE assumes that ice behaves as an incompressible fluid. The depth-averaged form of the mass conservation equation is:

.. math::

    \frac{\partial H}{\partial t} + \nabla \cdot (\bar{\mathbf{u}} H) = a

where:

- :math:`H` is the ice thickness,
- :math:`\bar{\mathbf{u}} = (u, v)^T` is the depth-averaged horizontal velocity,
- :math:`a` is the net surface mass balance, representing the difference between accumulation (e.g., snowfall) and ablation (e.g., surface melting or basal melt).

This formulation allows PINNICLE to support both steady-state and transient (time-dependent) simulations.

Time-Dependent Modeling
-----------------------

To simulate transient ice sheet behavior, PINNICLE expands the network input to include time :math:`t` along with spatial coordinates :math:`x, y`. The model predicts the evolution of ice thickness over time by enforcing the mass conservation equation at a set of spatio-temporal collocation points.

This is useful for:

- Simulating seasonal or interannual ice sheet changes
- Predicting future glacier evolution
- Assimilating time series data into forward models

Loss Function Contribution
--------------------------

The mass conservation equation contributes to the physical loss term :math:`L_\phi` in the total loss function:

.. math::

    L_\phi = \frac{\gamma_{H/t}}{N_\phi} \sum_{i=1}^{N_\phi} \left| \frac{\partial H}{\partial t} + \nabla \cdot (\bar{\mathbf{u}} H) - a \right|^2

where:

- :math:`N_\phi` is the number of collocation points,
- :math:`\gamma_{H/t}` is a weighting parameter used to balance the influence of this term in the loss function.

Implementation Notes
--------------------

- The user must set `"Mass transport"` as the equation in the configuration file:
  .. code-block:: python

     hp["equations"] = {"Mass transport": {}}

- Data inputs may include time series of:
  - Ice thickness
  - Horizontal velocity components
  - Surface mass balance

- Initial ice thickness must be provided at :math:`t = t_0`.

Applications
------------

This equation is demonstrated in:

- **Example 3** (Time-dependent modeling of Helheim Glacier, 2008–2009):
  A forward simulation using real velocity and mass balance time series to track ice thickness evolution.

For more details, see the `Examples <examples.html>`_ section.


