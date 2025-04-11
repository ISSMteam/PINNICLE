.. _neural_network:

Neural Network Architecture
===========================

PINNICLE is built on fully connected neural networks (FNNs) tailored for solving partial differential equations (PDEs) with physics-informed learning. Its architecture is modular and highly configurable to support both simple and complex glaciological modeling tasks.

Architecture Overview
---------------------

PINNICLE supports two types of neural network architectures:

1. **Fully Connected Neural Network (FNN)**  
   Default architecture used in most problems. All output variables are predicted jointly by a single neural network.

2. **Parallel Fully Connected Neural Networks (PFNN)**  
   Each dependent variable is predicted by its own sub-network. Useful when the relationships among variables are too complex to be learned jointly.

FNN (Default)
~~~~~~~~~~~~~

.. image:: _static/fnn_architecture.png
   :align: center
   :width: 60%

- One input layer (e.g., :math:`x, y, t`)
- Multiple hidden layers with activation functions
- One output layer (e.g., :math:`u, v, H, C`)

PFNN
~~~~

- One network per output variable
- Shared inputs across all sub-networks
- Useful for ill-conditioned or weakly coupled systems

Configuration
-------------

The neural network is configured using the following parameters in the `hp` dictionary:

.. code-block:: python

   hp["num_layers"] = 6             # Number of hidden layers
   hp["num_neurons"] = 32           # Number of neurons per layer
   hp["activation"] = "tanh"        # Activation function (default is tanh)
   hp["architecture"] = "FNN"       # or "PFNN"

You can also set:

- `fft = True` to activate Fourier Feature Transform
- `input_normalization = True` (enabled by default) to apply min–max scaling

Activation Functions
--------------------

The default activation is the hyperbolic tangent (`tanh`), which works well in many physical applications. You can change it to:

- `relu`
- `sine`
- `sigmoid`
- `swish`
- or custom activations from TensorFlow, PyTorch, or JAX

Input/Output Mapping
--------------------

- **Inputs**: spatial (:math:`x, y`) and temporal (:math:`t`) coordinates
- **Outputs**: dependent PDE variables (e.g., :math:`u, v, H, s, C, B`)

PINNICLE automatically constructs the mapping between inputs and outputs based on the specified physics model and user data.

Normalization
-------------

By default:
- Inputs are min–max normalized
- Outputs are de-normalized to original units
- This avoids scaling issues in PDE residuals and improves training stability

You do not need to manually scale physical quantities; PINNICLE handles this automatically.

Parallelization and Backends
----------------------------

PINNICLE is built on **DeepXDE**, which supports the following ML frameworks:

- **TensorFlow**
- **PyTorch**
- **JAX**

You can choose a backend by setting the environment variable `BACKEND` or directly in code.

Performance Tips
----------------

- For complex inverse problems, increase `num_neurons` to 64 or 128
- Use `PFNN` when solving multiple loosely coupled variables (e.g., friction + rheology)
- Deeper networks may require more epochs to converge
- Use FFT if training stagnates or if the solution contains high-frequency features

Example
-------

.. code-block:: python

   hp["num_layers"] = 6
   hp["num_neurons"] = 40
   hp["architecture"] = "PFNN"
   hp["fft"] = True
   hp["sigma"] = 10
   hp["num_fourier_feature"] = 30

This setup was used successfully in Example 2 (Pine Island Glacier) for joint inversion.


