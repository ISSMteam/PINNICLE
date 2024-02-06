import PINN_ICE as pinn
import os
import numpy as np
from datetime import datetime
import deepxde as dde

dde.config.set_default_float('float64')
dde.config.disable_xla_jit()

datestr = datetime.now().strftime("%Y%m%d_%H%M%S")
inputFileName="Helheim_fastflow.mat"
expFileName = "fastflow_CF.exp"
outputFileName="SSA_2D_test_deepxde"
modelFolder = "./Models/" + outputFileName + "_" + datestr  + "/"


# path for loading data and saving models
repoPath = "./"
appDataPath = os.path.join(repoPath, "dataset")
path = os.path.join(appDataPath, inputFileName)
# create output folder
yts =3600*24*365

# Hyper parameters
hp = {}
# General parameters
hp["epochs"] = 10
hp["learning_rate"] = 0.001
hp["loss_function"] = "MSE"
hp["save_path"] = modelFolder
hp["is_save"] = False
hp["is_plot"] = True

# NN
hp["activation"] = "tanh"
hp["initializer"] = "Glorot uniform"
hp["num_neurons"] = 20
hp["num_layers"] = 6

# data
hp["data_size"] = {"u":4000, "v":4000, "s":4000, "H":4000, "C":None}
hp["data_path"] = path

# domain
hp["shapefile"] = os.path.join(repoPath, "dataset", expFileName)
hp["num_collocation_points"] = 9000

# physics
SSA = {}
# SSA["input"] = ["x", "y"]
# SSA["output"] = ["u", "v", "s", "H", "C"]
# SSA["output_lb"] = [-1.0e4/yts, -1.0e4/yts, -1.0e3,  10.0, 0.01]
# SSA["output_ub"] = [ 1.0e4/yts,  1.0e4/yts,  2.5e3, 2.0e3, 1.0e4]
# SSA["data_weights"] = [1.0e-8*yts**2.0, 1.0e-8*yts**2.0, 1.0e-6, 1.0e-6, 1.0e-8]
# SSA["pde_weights"] = [1.0e-10, 1.0e-10 ]
SSA["scalar_variables"] = {"B":1.26802073401e+08}
hp["equations"] = {"SSA": SSA}

# create exp
experiment = pinn.PINN(hp)
print(experiment.param)
experiment.compile()
experiment.train()
