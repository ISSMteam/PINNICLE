import pinnicle

# hyperparameters
hp = {}
hp["epochs"] = 1000000

# NN
hp["num_neurons"] = 40
hp["num_layers"]  = 6
hp["fft"] = True
hp['sigma'] = 10
hp['num_fourier_feature'] = 30

# domain
hp["shapefile"] = "PIG.exp"
hp["num_collocation_points"] = 18000

# physics
hp["equations"] = {"SSA_VB": {}}

# data
issm = {}
issm["data_size"] = {"u":8000, "v":8000, "s":8000, "H":8000}
issm["data_path"] = "PIG.mat"
B = {"data_size":{"B":4000}, "data_path":"B.mat", "source":"mat"}
C = {"data_size":{"C":4000}, "data_path":"C.mat", "source":"mat"}

hp["data"] = {"ISSM":issm, "B":B, "C":C}

# create experiment
experiment = pinnicle.PINN(hp)
experiment.compile()

# Train
experiment.train()
