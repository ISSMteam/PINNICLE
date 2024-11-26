import pinnicle as pinn
import os
import numpy as np
import deepxde as dde

dde.config.set_default_float('float64')
dde.config.disable_xla_jit()
dde.config.set_random_seed(1234)

# General parameters
hp = {}
hp["epochs"] = 800000
hp["learning_rate"] = 0.001
hp["loss_function"] = "MSE"
hp["save_path"] = "./Models/Helheim_test/"
hp["is_save"] = True
hp["is_plot"] = True

# NN
hp["activation"] = "tanh"
hp["initializer"] = "Glorot uniform"
hp["num_neurons"] = 20
hp["num_layers"] = 6

# domain
hp["shapefile"] = "./dataset/fastflow_CF.exp"
hp["num_collocation_points"] = 9000

# physics
SSA = {}
SSA["scalar_variables"] = {"B":1.26802073401e+08}
hp["equations"] = {"SSA":SSA}

# data
issm = {}
issm["data_size"] = {"u":4000, "v":4000, "s":4000, "H":4000, "C":None, "vel":4000}
issm["data_path"] = "./dataset/Helheim_fastflow.mat"
hp["data"] = {"ISSM":issm}

# additional loss function
vel_loss = {}
vel_loss['name'] = "vel log"
vel_loss['function'] = "VEL_LOG"
vel_loss['weight'] = 1.0e-5
hp["additional_loss"] = {"vel":vel_loss}

# create experiment
experiment = pinn.PINN(hp)
print(experiment.params)
experiment.compile()

# Train
experiment.train()
# show results
experiment.plot_predictions(X_ref=experiment.model_data.data["ISSM"].X_dict, 
                            sol_ref=experiment.model_data.data["ISSM"].data_dict)

