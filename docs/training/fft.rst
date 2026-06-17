.. _fourier_features:

Fourier Feature Transform
=========================

PINNICLE includes an optional Fourier Feature Transform (FFT) module to improve the neural network’s ability to learn complex, high-frequency patterns, such as sharp surface variations, abrupt ice front transitions, or localized velocity gradients. This technique helps the neural network overcome limitations associated with traditional fully connected layers.


.. note::
   The Fourier Feature Transform is still an experimental feature in PINNICLE, use it with caution and check sensitivity to the chosen frequency scales.

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

Before applying this transform, PINNICLE min-max scales input coordinates to :math:`[-1, 1]`. The transformed vector :math:`\gamma(\mathbf{x})` is then used as the input to the neural network.


Multiscale Fourier Features
---------------------------

For multiscale problems, a single value of :math:`\sigma` may emphasize one preferred frequency range. A small :math:`\sigma` tends to favor smoother, larger-scale variation, while a larger :math:`\sigma` helps expose finer oscillatory structure. When a solution contains both scales, one Fourier embedding can fit one part of the signal while converging slowly on the other.

PINNICLE addresses this by allowing :math:`\sigma` to be a list:

.. math::

   \Gamma(\mathbf{x}) =
   \left[
   \gamma_{\sigma_1}(\mathbf{x}),
   \gamma_{\sigma_2}(\mathbf{x}),
   \ldots,
   \gamma_{\sigma_M}(\mathbf{x})
   \right],

where each :math:`\gamma_{\sigma_i}` uses its own random Gaussian projection matrix. The resulting input dimension is
:math:`2mM`, where :math:`M` is the number of scales in ``hp["sigma"]``.

This implementation is inspired by the multiscale Fourier feature architecture proposed for PINNs by Wang, Wang, and Perdikaris (2020). PINNICLE currently applies the multiscale transform as a concatenated input embedding across all configured coordinates; it does not separate spatial and temporal embeddings into different branches.


Configuration in PINNICLE
--------------------------

To activate Fourier features in your model, modify the neural network section of the configuration:

.. code-block:: python

   hp["fft"] = True                 # Enable Fourier Feature Transform
   hp["sigma"] = [1, 10, 100]       # List of standard deviation of Gaussian projection
   hp["num_fourier_feature"] = 30   # Number of frequency components (m)

PINNICLE will automatically embed the input coordinates using the specified settings before passing them to the first layer of the network. If ``hp["sigma"]`` is a scalar, PINNICLE uses a standard single-scale FFT. If it is a list, PINNICLE uses the multiscale FFT.

When using FFT, PINNICLE will save the randomly generated values of ``B`` to the ``param.json`` file, so that after training, this can be recovered.

Typical Parameter Guidelines
----------------------------

- ``sigma``: Controls the frequency scale of the random projection. Use a scalar for a single scale, or a list such as ``[1, 10]`` or ``[1, 10, 100]`` for multiscale behavior.
- ``num_fourier_feature``: Use 10-100 depending on problem size. A common choice is to use a value comparable to one half of the hidden-layer width. More features can capture finer details but increase memory use and training cost.
- ``B``: Optional fixed projection matrix. Leave this unset unless you need reproducible or externally designed Fourier features.
- Inputs are automatically normalized using min-max scaling before Fourier embedding.
- FFT currently does not support parallel neural networks in PINNICLE.


Performance Considerations
---------------------------

- Increases model input dimensionality from :math:`d` to :math:`2mM`, where :math:`M = 1` for single-scale FFT and :math:`M = \mathrm{len}(\sigma)` for multiscale FFT.
- May increase training time and memory use, especially with many scales or many Fourier features.
- Can improve convergence for hard-to-fit functions with localized or oscillatory structure.
- Can reduce spectral-bias effects by making multiple frequency ranges visible to the network.
- Very large ``sigma`` values can overfit or make optimization harder; tune the list of scales for the data and physics being modeled.

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
- Wang et al., 2021: "On the Eigenvector Bias of Fourier Feature Networks: From Regression to Solving Multi-scale PDEs with Physics-Informed Neural Networks"

