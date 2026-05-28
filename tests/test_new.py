import os
import importlib
import pinnicle as pinn
import numpy as np
import deepxde as dde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.collections import PathCollection, PolyCollection, QuadMesh

from pinnicle.utils import *
import pytest

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

pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")

def test_plotmodelcompare():
    axs = pm.plotmodelcompare(model, "issm", "u")
    assert len(axs) == 3
    assert [ax.get_title() for ax in axs] == ["Data: u", "Prediction", "Difference"]

def test_plotmodelcompare1():
    assert all(plotmodelcompare(model, "issm", "u", batch_size=10))
    assert all(plotmodelcompare(model, "issm", "u", max_points=4))
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

pm = importlib.import_module('pinnicle.utils.plotmodel')
@pytest.fixture
def regular_grid():
    x = np.linspace(0.0, 2.0, 3)
    y = np.linspace(0.0, 1.0, 2)
    Xg, Yg = np.meshgrid(x, y)
    return Xg.ravel(), Yg.ravel()


def test_as_1d_float_array_converts_masked_values_to_nan():
    arr = np.ma.array([[1.0, 2.0], [3.0, 4.0]], mask=[[False, True], [False, False]])

    result = pm._as_1d_float_array(arr)

    assert result.shape == (4,)
    assert result[0] == 1.0
    assert np.isnan(result[1])
    assert np.array_equal(result[[2, 3]], np.array([3.0, 4.0]))

def test_get_nan_mask_handles_masked_nan_and_inf_values():
    arr = np.ma.array([1.0, np.nan, np.inf, 4.0], mask=[False, False, False, True])

    mask = pm._get_nan_mask(arr)

    assert mask.dtype == bool
    assert np.array_equal(mask, np.array([False, True, True, True]))

def test_predict_in_batches_without_batching_calls_predict_once():
    X_nn = np.column_stack((np.arange(5.0), np.arange(5.0)))
    result = pm._predict_in_batches(model, X_nn, batch_size=None)

    assert result.shape == (5, 5)

def test_predict_in_batches_chunks_large_inputs():
    X_nn = np.column_stack((np.arange(10.0), np.arange(10.0)))

    result = pm._predict_in_batches(model, X_nn, batch_size=5)
    assert result.shape == (10, 5)

def test_extract_prediction_returns_named_output_column():
    sol_pred = np.array([[1.0, 2.0, 10.0, 3.0], [4.0, 5.0, 20.0, 8.0]])
    keylist = ["u", "v", "s", "H"]

    result = pm._extract_prediction(sol_pred, keylist, "v")

    assert np.array_equal(result, np.array([2.0, 5.0]))

def test_extract_prediction_computes_bed_from_surface_minus_thickness():
    sol_pred = np.array([[1.0, 2.0, 10.0, 3.0], [4.0, 5.0, 20.0, 8.0]])
    keylist = ["u", "v", "s", "H"]

    result = pm._extract_prediction(sol_pred, keylist, "bed")

    assert np.array_equal(result, np.array([7.0, 12.0]))

def test_extract_prediction_raises_for_unknown_output():
    sol_pred = np.zeros((2, 2))

    with pytest.raises(ValueError, match="Key missing"):
        pm._extract_prediction(sol_pred, ["u", "v"], "missing")

def test_build_plot_cache_detects_regular_grid(regular_grid):
    X, Y = regular_grid

    cache = pm._build_plot_cache(X, Y, prefer_grid=True)

    assert cache["is_grid"] is True
    assert cache["shape"] == (2, 3)
    assert cache["triangles"] is None
    assert cache["Xg"].shape == (2, 3)
    assert cache["Yg"].shape == (2, 3)

def test_build_plot_cache_falls_back_to_triangulation_for_scattered_points():
    X = np.array([0.0, 1.0, 0.0, 1.0, 0.4])
    Y = np.array([0.0, 0.0, 1.0, 1.0, 0.6])

    cache = pm._build_plot_cache(X, Y, prefer_grid=True)

    assert cache["is_grid"] is False
    assert isinstance(cache["triangles"], Triangulation)

def test_build_plot_cache_can_force_triangulation_even_for_regular_grid(regular_grid):
    X, Y = regular_grid

    cache = pm._build_plot_cache(X, Y, prefer_grid=False)

    assert cache["is_grid"] is False
    assert isinstance(cache["triangles"], Triangulation)

def test_plot_from_cache_uses_pcolormesh_for_regular_grid(regular_grid):
    X, Y = regular_grid
    cache = pm._build_plot_cache(X, Y, prefer_grid=True)
    fig, ax = plt.subplots()

    artist = pm._plot_from_cache(ax, cache, np.arange(X.size, dtype=float))

    assert isinstance(artist, QuadMesh)

def test_plot_from_cache_uses_tripcolor_for_scattered_points():
    X = np.array([0.0, 1.0, 0.0, 1.0, 0.4])
    Y = np.array([0.0, 0.0, 1.0, 1.0, 0.6])
    cache = pm._build_plot_cache(X, Y, prefer_grid=True)
    fig, ax = plt.subplots()

    artist = pm._plot_from_cache(ax, cache, np.arange(X.size, dtype=float))

    assert isinstance(artist, PolyCollection)

def test_plot_from_cache_scatter_drops_masked_points(regular_grid):
    X, Y = regular_grid
    cache = pm._build_plot_cache(X, Y, prefer_grid=True)
    fig, ax = plt.subplots()
    data = np.arange(X.size, dtype=float)
    mask = np.array([False, True, False, True, False, False])

    artist = pm._plot_from_cache(ax, cache, data, mask=mask, iscatter=True)

    assert isinstance(artist, PathCollection)
    assert artist.get_offsets().shape[0] == np.count_nonzero(~mask)

def test_plot2d_returns_quadmesh_for_regular_grid(regular_grid):
    X, Y = regular_grid
    fig, ax = plt.subplots()

    artist = pm.plot2d(ax, X, Y, np.arange(X.size, dtype=float))

    assert isinstance(artist, QuadMesh)
