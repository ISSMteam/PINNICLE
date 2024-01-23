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
hp["epochs"] = 2
hp["loss_weights"] = loss_weights
hp["learning_rate"] = 0.001
hp["loss_function"] = "MSE"
hp["is_save"] = False

# NN
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
hp["scalar_variables"] = {"B":1.26802073401e+08}

def test_loaddata():
    # load the data
    X_star, u_star, X_train, u_train, X_bc, u_bc, X_cf, n_cf, uub, ulb, B = \
    pinn.utils.prep_2D_data(path, hp["datasize"])
    data = pinn.modeldata.Data(X=X_train, sol=u_train)

def test_compile():
    experiment = pinn.PINN(hp)
    experiment.compile()
    assert experiment.loss_names == ['fSSA1', 'fSSA2']

def test_save_and_load_setting(tmp_path):
    experiment = pinn.PINN(hp)
    experiment.save_setting(path=tmp_path)
    assert experiment.param.param_dict == experiment.load_setting(path=tmp_path)

def test_train(tmp_path):
    hp["save_path"] = str(tmp_path)
    hp["is_save"] = True
    X_star, u_star, X_train, u_train, X_bc, u_bc, X_cf, n_cf, uub, ulb, mu = \
    pinn.utils.prep_2D_data(path, hp["datasize"])
    data = pinn.modeldata.Data(X=X_train, sol=u_train)
    experiment = pinn.PINN(hp, training_data=data)
    experiment.compile()
    experiment.train()
    assert experiment.loss_names == ['fSSA1', 'fSSA2', 'u', 'v', 's', 'H', 'C']
