import PINN_ICE as pinn
import os
import numpy as np
from datetime import datetime
import deepxde as dde
dde.config.set_default_float('float64')
dde.config.disable_xla_jit()

weights = [7, 7, 5, 5, 3, 3, 5];

inputFileName="Helheim_Weertman_iT080_PINN_fastflow_CF"
expFileName = "fastflow_CF.exp"

# path for loading data and saving models
repoPath = os.path.dirname(__file__) + "/../examples/"
appDataPath = os.path.join(repoPath, "dataset")
path = os.path.join(appDataPath, inputFileName)
yts =3600*24*365
loss_weights = [10**(-w) for w in weights]
loss_weights[2] = loss_weights[2] * yts*yts
loss_weights[3] = loss_weights[3] * yts*yts

hp = {}
# General parameters
hp["epochs"] = 10
hp["loss_weights"] = loss_weights
hp["learning_rate"] = 0.001
hp["loss_function"] = "MSE"

# NN
hp["input_variables"] = ["x","y"]
hp["output_variables"] = ["u", "v", "s", "H", "C"]
hp["activation"] = "tanh"
hp["initializer"] = "Glorot uniform"
hp["num_neurons"] = 20
hp["num_layers"] = 6
hp["output_lb"] = [-200/yts, -5000/yts, -900, 10, 0.01]
hp["output_ub"] = [7500/yts, 800/yts, 1600, 1700, 8500]

# data
hp["datasize"] = {"u":4000, "v":4000, "s":4000, "H":4000, "C":None}

# domain
hp["shapefile"] = os.path.join(repoPath, "dataset", expFileName)
hp["num_collocation_points"] = 9000

# physics
hp["equations"] = ["SSA"]
hp["mu"] = 1.26802073401e+08

def test_loaddata():
    # load the data
    X_star, u_star, X_train, u_train, X_bc, u_bc, X_cf, n_cf, uub, ulb, mu = \
    pinn.modeldata.prep_2D_data(path, hp["datasize"])
    data = pinn.modeldata.Data(X=X_train, sol=u_train)

def test_pinn():
    experiment = pinn.PINN(hp)
    experiment.compile()
