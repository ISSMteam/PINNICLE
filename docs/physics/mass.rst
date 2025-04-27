.. _mass_conservation:

Mass Conservation
=================

Mass conservation is a fundamental principle in ice sheet modeling. In PINNICLE, this equation governs how ice thickness evolves over time in response to ice flow and surface mass balance processes.

PINNICLE supports both steady-state and transient form mass conservation. The key words are as following:

- :code:`"MC"`: steady-state
- :code:`"Mass transport"`: time-dependent

Equation
--------

Ice, in large scale, behaves as an incompressible fluid. The mass conservation equation is written as:

.. math::

    \frac{\partial H}{\partial t} + \nabla \cdot (\bar{\mathbf{u}} H) = a

where:

- :math:`H` is the ice thickness,
- :math:`\bar{\mathbf{u}} = (u, v)^T` is the depth-averaged horizontal velocity,
- :math:`a` is the net surface mass balance, representing the difference between accumulation (e.g., snowfall) and ablation (e.g., surface melting or basal melt).

Time-Dependent Modeling
-----------------------

To simulate transient behavior, PINNICLE will automatically expands the network input to include time :math:`t` along with spatial coordinates :math:`x, y`. This equation describes the evolution of ice thickness over time, and will be evaluated at a set of spatio-temporal collocation points.

This is useful for:

- Simulating seasonal or interannual ice sheet changes
- Predicting future glacier evolution
- Assimilating time series data into models

Loss Function Contribution
--------------------------

The mass conservation equation contributes to the physical loss term :math:`L_\phi` in the total loss function:

.. math::

    L_{\phi(MC)} = \frac{\gamma_{H/t}}{N_\phi} \sum^{N_\phi}_{i=1} \left| \frac{\partial H}{\partial t} + \nabla \cdot (\bar{\mathbf{u}} H) - a \right|^2

where:

- :math:`N_\phi` is the number of collocation points,
- :math:`\gamma_{H/t}` is a weighting parameter used to balance the influence of this term in the loss function.

Implementation Notes
--------------------

- The user must set :code:`"Mass transport"` or :code:`"MC"` as the equation in the configuration file:

  .. code-block:: python

     hp["equations"] = {"Mass transport": {}}

- Data inputs may include time series of:

  - :code:`"u", "v"`: Horizontal velocity components
  - :code:`"a"`: Surface mass balance
  - :code:`"H"`: Ice thickness, only initial ice thickness need to be provided if sovling a forward problem.

Applications
------------

This equation is demonstrated in:

.. toctree::
   :maxdepth: 1

   ../examples/Helheim_Transient.rst
..
   A forward simulation using real velocity and mass balance time series to track ice thickness evolution.

For more details, see the `Examples <../pinnicle_examples.html>`_ section.


