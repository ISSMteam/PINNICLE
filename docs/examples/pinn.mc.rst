A simple PINN example to infer the basal friction coefficent
============================================================

Problem setup
-------------

We use  Shelfy Stream Approximation (SSA): 

.. math:: \nabla\cdot\boldsymbol{\sigma}_{SSA}+\boldsymbol{\tau}_b=\rho_i g H \nabla s, \quad \text{in } \Omega

where `\boldsymbol{\tau}_b` is the basal shear stress, determined by a friction law

.. math:: \boldsymbol{\tau}_b = -C^2|\boldsymbol{u}|^{m-1}\boldsymbol{u}

with the a calving front boundary condition

.. math:: \boldsymbol{n}\cdot\boldsymbol{\sigma}=(\bar{p}_i-\bar{p}_w)\boldsymbol{n} \quad \text{on } \Gamma


Implementation
--------------

This description goes through the implementation of a PINN solver for inverting basal friction coefficient step-by-step.

First, import the necessary modules:

.. code-block:: python

   import pinnicle as pinn
   import numpy as np
   import deepxde as dde


We setup some configurations in DeepXDE:

.. code-block:: python

   dde.config.set_default_float('float64')
   dde.config.disable_xla_jit()
   dde.config.set_random_seed(1234)


The whole setup of the PINNICLE is stored in a ``dict`` with specific names. We begin with the general parameters:

.. code-block:: python

   hp = {}
   hp["epochs"] = 1000
   hp["learning_rate"] = 0.001
   hp["loss_function"] = "MSE"
   hp["save_path"] = "./Models/Helheim_test"
   hp["is_save"] = False
   hp["is_plot"] = True


Next, we set the nerual network architecture:

.. code-block:: python

   hp["activation"] = "tanh"
   hp["initializer"] = "Glorot uniform"
   hp["num_neurons"] = 20
   hp["num_layers"] = 6


Then, we define the domain of the computation:

.. code-block:: python

   hp["shapefile"] = "./dataset/Helheim_Big.exp"
   hp["num_collocation_points"] = 5000


We add physics, SSA, to the PINN by:

.. code-block:: python

   SSA = {}
   SSA["scalar_variables"] = {"B":1.26802073401e+08}
   hp["equations"] = {"SSA":SSA}


There are several default setting in `SSAEquationParameter <https://pinnicle.readthedocs.io/en/add_example/_modules/pinnicle/physics/stressbalance.html#SSAEquationParameter.set_default>`_ such as:

.. code-block:: python

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u', 'v', 's', 'H', 'C']
        self.output_lb = [-1.0e4/self.yts, -1.0e4/self.yts, -1.0e3, 10.0, 0.01]
        self.output_ub = [ 1.0e4/self.yts,  1.0e4/self.yts,  2.5e3, 2000.0, 1.0e4]
        self.data_weights = [1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-6, 1.0e-6, 1.0e-8]
        self.residuals = ["fSSA1", "fSSA2"]
        self.pde_weights = [1.0e-10, 1.0e-10]

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                'B':1.26802073401e+08   # -8 degree C, cuffey
                }


This includes the ``key`` names of the input and output variables of the PINN, scaling factors, weights in the loss functions, ``key`` name of the residuals and weights, etc.


After that, we assign the data used for training:

.. code-block:: python

   issm = {}
   issm["data_size"] = {"u":1000, "v":1000, "s":1000, "H":1000, "C":None, "vel":1000}
   issm["data_path"] = "./dataset/Helheim_fastflow.mat"
   hp["data"] = {"ISSM":issm}

In ``data_size``, each ``key``:``value`` pair defines a variable in the training. 
If the ``key`` is not redefined in ``name_map``, then it will be used as default in the physics. 
The ``value`` associated with the ``key`` gives the number of data points used for training. 
If, the ``value`` is set to ``None``, then only Dirichlet boundary condition around the domain boundary will be used for the corresponding ``key``. 
If the variables is included in the training, but not gaven in ``data_size``, then there will be no data for this variable in the training.


Last, add an additional loss function

.. code-block:: python

   vel_loss = {}
   vel_loss['name'] = "vel log"
   vel_loss['function'] = "VEL_LOG"
   vel_loss['weight'] = 1.0e-5
   hp["additional_loss"] = {"vel":vel_loss}


Now, we can run the PINN model: 

.. code-block:: python

   experiment = pinn.PINN(hp)
   experiment.compile()
   experiment.train()



Complete code
-------------

.. literalinclude:: ../../examples/pinn_ssa.py
  :language: python
