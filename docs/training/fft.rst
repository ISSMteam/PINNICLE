.. _fourier_features:

Fourier Feature Transform
=========================

PINNICLE includes an optional Fourier Feature Transform (FFT) module to improve the neural network’s ability to learn complex, high-frequency patterns, such as sharp surface variations, abrupt ice front transitions, or localized velocity gradients. This technique helps the neural network overcome limitations associated with traditional fully connected layers.


.. note::
   The Fourier Feature Transform is still an experimental feature in PINNICLE, use with cautions.

Mathematical Formulation
-------------------------

Let :math:`\mathbf{x} \in \mathbb{R}^d` be the input coordinates (e.g., :math:`x, y, t`). The Fourier feature transform maps these into a higher-dimensional space using random projections:

.. math::

   \gamma(\mathbf{x}) = [\cos(2\pi B \mathbf{x}), \sin(2\pi B \mathbf{x})]^T

where:

- :math:`B \in \mathbb{R}^{m \times d}` is a random matrix sampled from a Gaussian distribution:
  :math:`B \sim \mathcal{N}(0, \sigma^2)`
- :math:`m` is the number of Fourier features
- :math:`\sigma` controls the frequency bandwidth

The transformed vector :math:`\gamma(\mathbf{x})` is used as the new input to the neural network, replacing or augmenting the original coordinates.

Configuration in PINNICLE
--------------------------

To activate Fourier features in your model, modify the neural network section of the configuration:

.. code-block:: python

   hp["fft"] = True                 # Enable Fourier Feature Transform
   hp["sigma"] = 10                 # Standard deviation of Gaussian projection
   hp["num_fourier_feature"] = 30   # Number of frequency components (m)

PINNICLE will automatically embed the input coordinates using the specified settings before passing them to the first layer of the network.

Typical Parameter Guidelines
----------------------------

- **sigma**: A larger :math:`\sigma` increases the frequency range. Common values for glaciology application range from 5 to 30.
- **num_fourier_feature**: Use 10–100 depending on problem size. More features capture finer details but increase computation.
- Inputs are automatically normalized using min–max scaling before Fourier embedding.

Performance Considerations
---------------------------

- Increases model input dimensionality (from :math:`d` to :math:`2m`)
- May increase training time slightly
- Improves convergence for hard-to-fit functions
- Reduces aliasing and helps overcome neural network spectral bias

Example
-------

In :ref:`example2`, Fourier features are used to infer both basal friction and spatially varying rheology. The configuration included:

.. code-block:: python

   hp["fft"] = True
   hp["sigma"] = 10
   hp["num_fourier_feature"] = 30

References
----------

- Tancik et al., 2020: "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
- Wang et al., 2021: "Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks"

