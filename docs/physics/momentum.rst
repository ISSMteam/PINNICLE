.. _momentum_conservation:

Momentum Conservation
=====================

The conservation of momentum governs the mechanical behavior of ice flow under the influence of gravity, internal deformation, and basal resistance. PINNICLE currently supports two widely used approximations of the momentum balance equations:

1. **Shelfy-Stream Approximation (SSA)**
2. **Mono-Layer Higher-Order (MOLHO) Model**

Other momentum conservation equations can be implemented in the similar way as these two equations.

Shelfy-Stream Approximation (SSA)
---------------------------------

The SSA simplifies the full-Stokes equations by neglecting vertical shear stresses, making it particularly suitable for modeling fast-flowing regions like ice streams and ice shelves.
The SSA equations are:

.. math::

   \nabla \cdot \sigma_{\text{SSA}} - {\tau}_b = \rho_i g H \nabla s

where:

- :math:`\sigma_{\text{SSA}}` is the stress tensor,
- :math:`{\tau}_b` is the basal shear stress,
- :math:`\rho_i` is the ice density,
- :math:`g` is gravitational acceleration,
- :math:`H` is ice thickness,
- :math:`s` is surface elevation.


Particularly, the stress tensor is

.. math::

   \sigma_{\text{SSA}} = \mu H\begin{bmatrix}
   4 \frac{\partial u}{\partial x} + 2\frac{\partial v}{\partial y} &  \frac{\partial u}{\partial y} +  \frac{\partial v}{\partial x} \\
     \frac{\partial u}{\partial y} +  \frac{\partial v}{\partial x} & 2\frac{\partial u}{\partial x} + 4\frac{\partial v}{\partial y}
   \end{bmatrix}
   
with the viscosity :math:`\mu` follows Glen's flow law:

.. math::

   \mu = \frac{B}{2}
   \left[
   \left( \frac{\partial u}{\partial x} \right)^2
   +
   \left( \frac{\partial v}{\partial y} \right)^2
   +
   \frac{1}{4} \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right)^2
   +
   \frac{\partial u}{\partial x} \frac{\partial v}{\partial y}
   \right]^{\frac{1-n}{2n}}

where :math:`B` is a temperature-dependent rheological parameter and :math:`n` is the flow-law exponent (typically 3).

The basal shear stress can be modeled, for example, using Weertman's friction law:

.. math::

   {\tau}_b = C^2 {|\mathbf{u}|}^{m-1} \cdot {\mathbf{u}}

where:

- :math:`C` is the friction coefficient,
- :math:`m` is the sliding exponent,
- :math:`{\mathbf{u}}` is the basal velocity

Mono-Layer Higher-Order (MOLHO) Model
-------------------------------------

The MOLHO model extends SSA by including vertical shear deformation in a depth-integrated form, improving accuracy for inland ice or thick ice sheets.

The velocity field is decomposed as:

.. math::

   \mathbf{u}(z) = \mathbf{u}_b + \mathbf{u}_{sh} (1 - \zeta^{n+1})

where:

- :math:`\mathbf{u}_b` is basal velocity,
- :math:`\mathbf{u}_{sh}` is shear velocity component,
- :math:`\zeta = \frac{s - z}{H}` is the normalized vertical coordinate.

The MOLHO equations introduce additional vertically integrated viscosity terms :math:`\bar{\mu}_1, \bar{\mu}_2, \bar{\mu}_3, \bar{\mu}_4` to capture depth-dependent stress and strain.

These extensions allow for:

- Capturing deformation-dominated flow
- Improved modeling of interior flow regimes
- More accurate inverse problems involving ice rheology

Loss Function Contribution
--------------------------

Momentum conservation contributes to the total loss function by evaluating the residual of the PDEs at some randomly select collocation points within the domain of interests.
An example of the SSA residual is as follows:

.. math::

   L_\phi = \frac{\gamma_{\tau}}{N_\phi} \sum_{i=1}^{N_\phi} \left| \nabla \cdot \sigma - {\tau}_b - \rho_i g H \nabla s \right|^2

where:

- :math:`N_\phi` is the number of collocation points,
- :math:`\gamma_{\tau}` is the weight for the momentum residual term.

Implementation Notes
--------------------

To activate momentum conservation models in PINNICLE, use one of the following in the hyper-parameter dictionary:

.. code-block:: python

   # For SSA with a constant B
   hp["equations"] = {"SSA": {}}

.. code-block:: python

   # For SSA with spatially varying rheology B 
   hp["equations"] = {"SSA_VB": {}}

.. code-block:: python

   # For MOLHO
   hp["equations"] = {"MOLHO": {}}


..
   TODO: input data 

Applications
------------

This equation is demonstrated in:

- :ref:`example1`
- :ref:`example2`

For more details, see the :ref:`examples` section.

References
----------

- Cheng et al. (2024). "Forward and Inverse Modeling of Ice Sheet Flow Using Physics-Informed Neural Networks"
- dos Santos et al. (2022). "A new vertically integrated, Mono-Layer Higher-Order ice flow model"
