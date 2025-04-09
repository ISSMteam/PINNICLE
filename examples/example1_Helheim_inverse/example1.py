import pinnicle

# General parameters
hp = {}
hp["epochs"] = 100000

# NN
hp["num_neurons"] = 20
hp["num_layers"] = 6

# domain
hp["shapefile"] = "Helheim.exp"
hp["num_collocation_points"] = 9000

# physics
hp["equations"] = {"SSA":{}}

# data
issm = {}
issm["data_size"] = {"u":4000, "v":4000, "s":4000, "H":4000, "C":None}
issm["data_path"] = "Helheim.mat"
hp["data"] = {"ISSM":issm}

# create experiment
experiment = pinnicle.PINN(hp)
experiment.compile()

# Train
experiment.train()
