import PINN_ICE as pinn
import os
import numpy as np
from datetime import datetime
import deepxde as dde

dde.config.set_default_float('float64')
dde.config.disable_xla_jit()
dde.config.set_random_seed(1234)

# create experiments
datestr = datetime.now().strftime("%Y%m%d_%H%M%S")

# data file and path
inputFileName="Helheim_basin.mat"
expFileName = "Helheim_Big.exp"
repoPath = "./"
appDataPath = os.path.join(repoPath, "DATA")
data_path = os.path.join(appDataPath, inputFileName)

# path for saving results and figures
outputFileName="Helheim_test"
modelFolder = "./Models/" + outputFileName + "_" + datestr  + "/"

# General parameters
hp = {}
hp["epochs"] = 1000
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
hp["data_size"] = {"u":1000, "v":1000, "s":1000, "H":1000, "C":None, "vel":1000}
hp["data_path"] = data_path

# domain
hp["shapefile"] = os.path.join(repoPath, "DATA", expFileName)
hp["num_collocation_points"] = 5000

# additional loss function
vel_loss = {}
vel_loss['name'] = "vel log"
vel_loss['function'] = "VEL_LOG"
vel_loss['weight'] = 1.0e-5
hp["additional_loss"] = {"vel":vel_loss}

# physics
SSA = {}
SSA["scalar_variables"] = {"B":1.26802073401e+08}
hp["equations"] = {"SSA":SSA}

# create experiment
experiment = pinn.PINN(hp)
print(experiment.params)
experiment.compile()

# Train
experiment.train()
# show results
experiment.plot_predictions(X_ref=experiment.model_data.X_dict, sol_ref=experiment.model_data.data_dict)
