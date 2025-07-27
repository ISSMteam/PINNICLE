import os
import pinnicle as pinn
import numpy as np
import deepxde as dde
import matplotlib as mpl
import matplotlib.pyplot as plt
from pinnicle.utils import *
import pytest

dde.config.set_default_float('float64')

weights = [7, 7, 5, 5, 3, 3, 5]

inputFileName="Helheim_fastflow.mat"
expFileName = "fastflow_CF.exp"
radarFileName = "flightTracks.mat"

# path for loading data and saving models
repoPath = os.path.dirname(__file__) + "/../examples/"
appDataPath = os.path.join(repoPath, "dataset")
path = os.path.join(appDataPath, inputFileName)
rpath = os.path.join(appDataPath, radarFileName)
yts =3600*24*365
loss_weights = [10**(-w) for w in weights]
loss_weights[2] = loss_weights[2] * yts*yts
loss_weights[3] = loss_weights[3] * yts*yts

hp = {}
# General parameters
hp["epochs"] = 10
hp["loss_weights"] = loss_weights
hp["learning_rate"] = 0.001
hp["loss_functions"] = "MSE"
hp["is_save"] = False

# NN
hp["activation"] = "tanh"
hp["initializer"] = "Glorot uniform"
hp["num_neurons"] = 10
hp["num_layers"] = 2

# data
issm = {}
issm["data_path"] = path

# domain
hp["shapefile"] = os.path.join(repoPath, "dataset", expFileName)
hp["num_collocation_points"] = 100

# physics
SSA = {}
SSA["scalar_variables"] = {"B":1.26802073401e+08}
hp["equations"] = {"SSA":SSA}

x = np.array(range(10))
y = np.array(range(10))
[X,Y]= np.meshgrid(x,y)
X = X.flatten()
Y = Y.flatten()
data = X*2+Y/2+X*Y
fig, axs = plt.subplots(3,2, figsize=(8,8))

def test_plot2d():
    assert plot2d(axs[0][0], X, Y, data)

def test_plottriangle():
    assert plottriangle(axs[0][0], mpl.tri.Triangulation(X, Y), data)

def test_plotscatter():
    assert plot2d(axs[0][0], X, Y, data)
