.. _time_dependent:

Time-Dependent Modeling
=======================

PINNICLE supports time-dependent (transient) modeling by solving partial differential equations (PDEs) that evolve over both space and time. This functionality enables simulations of ice sheet changes over months to years, using observational time series of variables such as ice velocity, surface mass balance, and ice thickness.

Motivation
----------

Glaciers and ice sheets exhibit strong temporal variability, driven by seasonal climate cycles, oceanic forcing, and long-term changes. Time-dependent models are essential for:

- Simulating glacier evolution across multiple years
- Integrating time series observational datasets
- Forecasting future changes in ice dynamics

Formulation
-----------

PINNICLE solves time-dependent PDEs by incorporating time :math:`t` as an additional input variable to the neural network. One typical application is the mass conservation equation:

.. math::

   \frac{\partial H}{\partial t} + \nabla \cdot (\bar{\mathbf{u}} H) = a

where:
- :math:`H` is ice thickness,
- :math:`\bar{\mathbf{u}} = (u, v)^T` is depth-averaged velocity,
- :math:`a` is surface mass balance.

This formulation enables PINNICLE to simulate the evolution of ice thickness over time by incorporating both spatial and temporal derivatives into the loss function.

Neural Network Design
---------------------

For time-dependent problems:
- Inputs: :math:`x, y, t` (spatial and temporal coordinates)
- Outputs: predicted values of variables such as :math:`H`, :math:`u`, :math:`v`, and :math:`a`
- Collocation points are generated in both space and time

Time-dependent training data can be provided at multiple time slices. PINNICLE handles this with built-in temporal alignment and batch processing.

Loss Function
-------------

The loss function integrates misfit and physics-based terms across all time steps. For example:

.. math::

   L = L_u + L_H + L_a + L_\phi

The physical residual term enforces the time-dependent PDE:

.. math::

   L_\phi = \frac{\gamma_{H/t}}{N_\phi} \sum_{i=1}^{N_\phi} \left| \frac{\partial H}{\partial t} + \nabla \cdot (\bar{\mathbf{u}} H) - a \right|^2

Implementation
--------------

To enable transient simulations, set the following keys in your configuration dictionary:

.. code-block:: python

   hp["time_dependent"] = True
   hp["start_time"] = 2008
   hp["end_time"] = 2009

Time series data can be added as:

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

PINNICLE will automatically:
- Construct the spatiotemporal domain
- Align time-stamped data
- Generate collocation points across space and time

Applications
------------

- **Example 3:** Forward modeling of Helheim Glacier from 2008 to 2009 using real-world transient data

See the `Examples <examples.html>`_ section for more information on how to use time-dependent simulations in practice.

