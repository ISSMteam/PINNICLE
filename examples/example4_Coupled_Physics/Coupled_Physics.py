# PINN experiment for Deception
# Mass Conservation + Momentum Conservation
# ------------------------------------------
# Author: Mansa Krishna
# ------------------------------------------
import pinnicle as pinn
# -----------------------------------------
# general parameters
hp = {}
hp["epochs"] = 700000
hp["learning_rate"] = 0.0001
hp["loss_function"] = "MSE"

# neural network
hp["activation"] = "tanh"
hp["initializer"] = "Glorot uniform"
hp["num_neurons"] = 128
hp["num_layers"] = 6

# data
issm = {}
issm["data_size"] = {'u':4000, 'v':4000, 'a':4000, 's':4000, 'vel':4000}

mat = {}
mat["data_size"] = {"H":4000}
mat["data_path"] = "ProcessedTracks.mat"
mat["name_map"] = {"H":"thickness"}
mat["source"] = "mat"

hp["data"] = {"ISSM":issm, "MAT":mat}

# domain
hp["shapefile"] = "Narssap.exp"
hp["num_collocation_points"] = 9000

# additional loss function
hp["additional_loss"] = {}
# vel log
vel_loss = {}
vel_loss['name'] = "vel MAPE"
vel_loss['function'] = "MAPE"
vel_loss['weight'] = 1.0e-6
hp["additional_loss"]["vel"] = vel_loss

# physics
hp["equations"] = {"MC":{}, "SSA":{}}

# create experiment
md = pinn.PINN(hp)
print(md.params)
md.compile()

# train the model
md.train()
