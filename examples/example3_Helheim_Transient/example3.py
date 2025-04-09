import pinnicle
import numpy as np

# hyperparameters
hp = {}
hp["epochs"] = 800000

# time dependent problem
hp["time_dependent"] = True
hp["start_time"]     = 2008
hp["end_time"]       = 2009

# NN
hp["num_neurons"] = 32
hp["num_layers"]  = 6

# domain
hp["shapefile"] = "Helheim_Basin.exp"
hp["num_collocation_points"] = 10000

# physics
hp["equations"] = {"Mass transport":{}}

# data
hp["data"] = {}
for t in np.linspace(2008,2009,11):
    issm = {}
    if t == 2008:
        issm["data_size"] = {"u":3000, "v":3000, "a":3000, "H":3000}
    else:
        issm["data_size"] = {"u":3000, "v":3000, "a":3000, "H":None}
        
    issm["data_path"]         = "Helheim_Transient_" + "%g"%t + ".mat"
    issm["default_time"]      = t
    issm["source"]            = "ISSM"
    hp["data"]["ISSM"+"%g"%t] = issm

# create experiment
experiment = pinnicle.PINN(hp)
experiment.compile()

# Train
experiment.train()
