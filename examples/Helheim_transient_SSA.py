# %load test.py
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
sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 5
num_fourier_feature = int(sys.argv[3]) if len(sys.argv) > 3 else 30
pdeweights = int(sys.argv[4]) if len(sys.argv) > 4 else 10
dt0weights = int(sys.argv[5]) if len(sys.argv) > 5 else 12
dCweights = int(sys.argv[6]) if len(sys.argv) > 6 else 10
epochs = int(sys.argv[7]) if len(sys.argv) > 7 else 400000

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
outputFileName=f"{filename}_epochs{epochs}_sigma{sigma:.0f}_w1e{pdeweights}_dt1e{dt0weights}"
outputFolder = os.path.join(repoPath, "Models", outputFileName + "_" + datestr + "/")

yts = pinn.physics.constants.Constants().yts

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
# time dependent problem
hp["time_dependent"] = True
hp["start_time"] = 2008*yts
hp["end_time"] = 2009*yts

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
timeList = np.linspace(2008,2009,11)
hp["data"] = {}
for t in timeList:
    issm = {}
    issm["data_size"] = {"u":3000, "v":3000, "a":3000, "H":None, "s":3000, "C":None}
    inputFileName = filename + "_Transient_" + "%g"%t + ".mat"
    inputPath = os.path.join(repoPath, "DATA/Helheim_subdomain/submodels/", inputFileName)
    issm["data_path"] = inputPath
    issm["default_time"] = t*yts
    issm["source"] = "ISSM"
    hp["data"]["ISSM"+"%g"%t] = issm

# physics
Time_Invariant = {}
Time_Invariant["pde_weights"] = [10**dt0weights, 10**dCweights]
Thickness = {}
Thickness["pde_weights"] = [10**pdeweights]
hp["equations"] = {"Thickness": Thickness, "Time_Invariant": Time_Invariant, "SSA":{}}

# domain
hp["shapefile"] = expPath
hp["num_collocation_points"] = 30000

# create exp
experiment = pinn.PINN(hp)
print(experiment.params)
experiment.compile()

experiment.train()
experiment.plot_predictions(filename="2D_2009.png", X_ref=experiment.model_data.data["ISSM2009"].X_dict, sol_ref=experiment.model_data.data["ISSM2009"].data_dict, default_time=2009*yts)

