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
issm["data_size"] = {"u":10,"s":10,"H":10}
hp["data"] = {"issm":issm}

# domain
hp["shapefile"] = os.path.join(repoPath, "dataset", expFileName)
hp["num_collocation_points"] = 100

# physics
SSA = {}
SSA["scalar_variables"] = {"B":1.26802073401e+08}
hp["equations"] = {"SSA":SSA}
model = pinn.PINN(params=hp)
model.compile()

x = np.array(range(10))
y = np.array(range(10))
[X,Y]= np.meshgrid(x,y)
X = X.flatten()
Y = Y.flatten()
data = X*2+Y/2+X*Y
fig, axs = plt.subplots(3,2, figsize=(8,8))

def test_plotmodelcompare():
    assert all(plotmodelcompare(model, "issm", "u"))
    assert all(plotmodelcompare(model, "issm", "u", scaling=2))
    assert all(plotmodelcompare(model, "issm", "u", iscatter=True))
    assert len(plotmodelcompare(model, "issm", "u", diffrange=1))==3

def test_plotpredict():
    assert plotprediction(axs[0][0], model, "u")
    assert plotprediction(axs[0][0], model, "u", X=X, Y=Y, scaling=2)
    assert plotprediction(axs[0][0], model, "bed")
    with pytest.raises(Exception):
        plotprediction(axs[0][0], model, "invalid_key")
    assert plotprediction(axs[0][0], model, "u", operator=lambda x: x**2)
    assert plotprediction(axs[0][0], model, "u", operator=np.abs)

def test_plotdiff():
    assert plotdiff(axs[0][0], model, X, Y, data, "u")
    assert plotdiff(axs[0][0], model, X, Y, data, "u", scaling=2, iscatter=True)

def test_plot2d():
    assert plot2d(axs[0][0], X, Y, data)
    mask = np.ones(X.shape, dtype=bool)
    data[mask] = np.nan
    assert plot2d(axs[0][0], X, Y, data)
    assert plot2d(axs[0][0], X, Y, data, mask=mask)

def test_plottriangle():
    assert plottriangle(axs[0][0], mpl.tri.Triangulation(X, Y), data)

def test_plotscatter():
    assert plot2d(axs[0][0], X, Y, data)
