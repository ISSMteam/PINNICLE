import pinnicle

# hyperparameters
hp = {}
hp["epochs"] = [100000, 10000]
hp["optimizers"] = ["adam","L-BFGS"]

# NN
hp["num_neurons"] = 40
hp["num_layers"]  = 6
hp["fft"] = True
hp['sigma'] = [1, 10]
hp['num_fourier_feature'] = 20

# domain
hp["shapefile"] = "PIG.exp"
hp["num_collocation_points"] = 4500
hp["period"] = 100
hp["mini_batch"] = 4000

# physics
hp["equations"] = {"SSA_VB": {}}

# data
issm = {}
issm["data_size"] = {"u":"MAX", "v":"MAX", "s":"MAX", "H":"MAX"}
issm["data_path"] = "PIG.mat"
B = {"data_size":{"B":"MAX"}, "data_path":"B.mat", "source":"mat"}
C = {"data_size":{"C":"MAX"}, "data_path":"C.mat", "source":"mat"}

hp["data"] = {"ISSM":issm, "B":B, "C":C}

# create experiment
experiment = pinnicle.PINN(hp)
experiment.compile()

# Train
experiment.train()
