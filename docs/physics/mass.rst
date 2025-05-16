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

Steady-State
~~~~~~~~~~~~

The steady-state simulate assumes no time dependent in the mass conservation equation by simply moving :math:`\frac{\partial H}{\partial t}` to the right-hand-side of the equation, and considering that as a forcing term (dynamical thinning).


Time-Dependent Modeling
~~~~~~~~~~~~~~~~~~~~~~~

To simulate transient behavior, PINNICLE will automatically expands the network input to include time :math:`t` along with spatial coordinates :math:`x, y`. This equation describes the evolution of ice thickness over time, and will be evaluated at a set of spatio-temporal collocation points.

This is useful for:

- Simulating seasonal or interannual ice sheet changes
- Predicting future glacier evolution
- Assimilating time series data into models

To enable transient simulations, set the following keys in your configuration dictionary:

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

To activate mass conservation models in PINNICLE, use one of the following in the hyper-parameter dictionary:

.. code-block:: python

   # For steady-state
   hp["equations"] = {"MC": {}}

.. code-block:: python

   # For time dependent
   hp["equations"] = {"Mass transport": {}}

Time-Dependent Problems
~~~~~~~~~~~~~~~~~~~~~~~

To solve time dependent problems in PINNICLE, just set the following in the hyper-parameter dictionary:

.. code-block:: python

   hp["time_dependent"] = True
   hp["start_time"] = 2008
   hp["end_time"] = 2009

And, the time series data can be added as:

.. code-block:: python

   for t in np.linspace(2008, 2009, 11):
       issm = {}
       if t == 2008:
           issm["data_size"] = {"u": 3000, "v": 3000, "a": 3000, "H": 3000}
       else:
           issm["data_size"] = {"u": 3000, "v": 3000, "a": 3000, "H": None}

       issm["data_path"] = f"Helheim_Transient_{t}.mat"
       issm["default_time"] = t
       issm["source"] = "ISSM"
       hp["data"][f"ISSM{t}"] = issm

Applications
------------

This equation is demonstrated in:

- :ref:`example3`
..
   A forward simulation using real velocity and mass balance time series to track ice thickness evolution.

For more details, see the :ref:`examples` section.


