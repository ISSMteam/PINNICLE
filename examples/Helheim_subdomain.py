import pinnicle as pinn
import os
import numpy as np
from datetime import datetime
import deepxde as dde
import sys

dde.config.set_default_float('float64')
dde.config.disable_xla_jit()
dde.config.set_random_seed(1234)

# load arguments from command line
print ('argument list', sys.argv)
subId = int(sys.argv[1]) if len(sys.argv) > 1 else 6
sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 30
num_fourier_feature = int(sys.argv[3]) if len(sys.argv) > 3 else 10
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 400000

# path for loading data and saving models
datestr = datetime.now().strftime("%Y%m%d_%H%M%S")

# input data
repoPath = "./"
glacier = "Helheim_Big"
subdomain = f"sub_3_2_{subId}"

# data files
filename = glacier + "_" + subdomain
inputFileName = filename + ".mat"
inputPath = os.path.join(repoPath, "DATA/Helheim_subdomain/submodels/", inputFileName)
expFileName =  filename + ".exp"
expPath = os.path.join(repoPath, "DATA/Helheim_subdomain/Exp/", expFileName)

# output folder
outputFileName=f"{filename}_epochs{epochs}_sigma{sigma:.0f}_nff{num_fourier_feature}"
outputFolder = os.path.join(repoPath, "Models", outputFileName + "_" + datestr + "/")

hp = {}
# General parameters
hp["epochs"] = epochs
hp["learning_rate"] = 0.001
hp["loss_function"] = "MSE"
hp["save_path"] = outputFolder
hp["is_save"] = True
hp["is_plot"] = True
# callbacks
# resample
hp["period"] = 0.2*hp["epochs"]
# early stop
hp["min_delta"] = 1e-5
hp["patience"] = 0.8*hp["epochs"]
# checkpoints
hp["checkpoint"] = True

# NN
hp["activation"] = "tanh"
hp["initializer"] = "Glorot uniform"
hp["num_neurons"] = 32
hp["num_layers"] = 6
hp["is_parallel"] = False
hp['fft'] = (sigma > 0)
hp['sigma'] = sigma
hp['num_fourier_feature'] = num_fourier_feature

# data
issm = {}
issm["data_size"] = {"u":10000, "v":10000, "s":10000, "C":None, "H":10000, "B":10000, 'vel':10000}
issm["data_path"] = inputPath
issm["source"] = "ISSM"

hp["data"] = {"ISSM":issm}

# domain
hp["shapefile"] = expPath
hp["num_collocation_points"] = 10000

# additional loss function
vel_loss = {}
vel_loss['name'] = "vel MAPE"
vel_loss['function'] = "MAPE"
# vel_loss['name'] = "vel L2"
# vel_loss['function'] = "MEAN_SQUARE_LOG"
vel_loss['weight'] = 1.0e-6
hp["additional_loss"] = {"vel":vel_loss}

# physics
SSA = {}
SSA["scalar_variables"] = {}
#SSA["pde_weights"] = [1e-10,1e-10]

hp["equations"] = {"SSA_VB":SSA}

# create exp
experiment = pinn.PINN(hp)
print(experiment.params)
experiment.compile()

experiment.train()
experiment.plot_predictions(filename="2D_adam.png", X_ref=experiment.model_data.data["ISSM"].X_dict, sol_ref=experiment.model_data.data["ISSM"].data_dict, absvariable=['C'])
