.. _momentum_conservation:

Momentum Conservation
=====================

In glaciology, the conservation of momentum governs the mechanical behavior of ice flow under the influence of gravity, internal deformation, and basal resistance. PINNICLE supports two widely used approximations of the momentum equations:

1. **Shelfy-Stream Approximation (SSA)**
2. **Mono-Layer Higher-Order (MOLHO) Model**

These models allow users to simulate fast-flowing ice streams, floating ice shelves, and inland deformation zones.

Shelfy-Stream Approximation (SSA)
---------------------------------

The SSA simplifies the full-Stokes equations by neglecting vertical shear stresses, making it particularly suitable for modeling fast-sliding regions like ice streams and ice shelves.

The SSA momentum balance equations are:

.. math::

   \nabla \cdot \sigma_{\text{SSA}} - \boldsymbol{\tau}_b = \rho_i g H \nabla s

where:

- :math:`\sigma_{\text{SSA}}` is the vertically integrated stress tensor,
- :math:`\boldsymbol{\tau}_b` is the basal shear stress,
- :math:`\rho_i` is the ice density,
- :math:`g` is gravitational acceleration,
- :math:`H` is ice thickness,
- :math:`s` is surface elevation.

The viscosity :math:`\mu` follows Glen's flow law:

.. math::

   \mu = \frac{B}{2} \left( \dot{\varepsilon}_{\text{eff}} \right)^{\frac{1-n}{n}}

where :math:`B` is a temperature-dependent rheological parameter and :math:`n` is the flow-law exponent (typically 3).

The basal shear stress is modeled using Weertman's friction law:

.. math::

   \boldsymbol{\tau}_b = C^2 |\mathbf{u}|^m \cdot \hat{\mathbf{u}}

where:
- :math:`C` is the friction coefficient,
- :math:`m` is the sliding exponent,
- :math:`\hat{\mathbf{u}}` is the unit vector of velocity direction.

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

Momentum conservation contributes to the physical loss term in the total loss function:

.. math::

   L_\phi = \frac{\gamma_{\tau}}{N_\phi} \sum_{i=1}^{N_\phi} \left| \nabla \cdot \sigma - \boldsymbol{\tau}_b - \rho_i g H \nabla s \right|^2

where:
- :math:`N_\phi` is the number of collocation points,
- :math:`\gamma_{\tau}` is the weight for the momentum residual term.

Implementation Notes
--------------------

To activate momentum conservation models in PINNICLE, use one of the following in the configuration dictionary:

.. code-block:: python

   # For SSA
   hp["equations"] = {"SSA": {}}

   # For SSA with rheology (SSA_VB)
   hp["equations"] = {"SSA_VB": {}}

   # For MOLHO
   hp["equations"] = {"MOLHO": {}}

These models can be customized and combined with different data inputs and inverse targets (e.g., basal friction, viscosity).

Applications
------------

- **Example 1:** SSA inverse problem on Helheim Glacier
- **Example 2:** SSA with spatially varying rheology and basal drag on Pine Island Glacier

See the `Examples <examples.html>`_ page for full details.


