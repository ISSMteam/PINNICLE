import pytest
import deepxde as dde
import os
import numpy as np
from deepxde import backend
from deepxde.backend import backend_name
from pinnicle.utils import save_dict_to_json, load_dict_from_json, data_misfit, load_mat, down_sample_core, down_sample

data = {"s":1, "v":[1, 2, 3]}


def test_save_and_load_dict(tmp_path):
    save_dict_to_json(data, tmp_path, "temp.json")
    save_dict_to_json(data, tmp_path, "noextension")
    assert data == load_dict_from_json(tmp_path, "temp.json")
    assert data == load_dict_from_json(tmp_path, "temp")
    assert data == load_dict_from_json(tmp_path, "noextension.json")

def test_data_misfit():
    with pytest.raises(Exception):
        data_misfit.get("not defined")
    assert data_misfit.get("MSE") != None
    assert data_misfit.get("VEL_LOG") != None
    assert data_misfit.get("MEAN_SQUARE_LOG") != None
    if backend_name == "tensorflow":
        assert data_misfit.get("MAPE") != None

def test_data_misfit_functions():
    assert  data_misfit.get("MSE")(dde.backend.as_tensor([1.0]),dde.backend.as_tensor([1.0])) == 0.0
    assert  data_misfit.get("VEL_LOG")(dde.backend.as_tensor([1.0]),dde.backend.as_tensor([1.0])) == 0.0
    assert  data_misfit.get("MEAN_SQUARE_LOG")(dde.backend.as_tensor([1.0]),dde.backend.as_tensor([1.0])) == 0.0
    if backend_name == "tensorflow":
        assert  data_misfit.get("MAPE")(dde.backend.as_tensor([1.0]),dde.backend.as_tensor([1.0])) == 0.0

def test_loadmat():
    filename = "flightTracks.mat"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, filename)
    assert load_mat(path)

    filename = "flightTracks73.mat"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, filename)
    assert load_mat(path)

def test_down_sample_core():
    filename = "flightTracks.mat"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, filename)
    data = load_mat(path)
    points = np.hstack((data['x'].flatten()[:,None],
                        data['y'].flatten()[:,None]))

    ind = down_sample_core(points)
    assert ind.shape == (2966,)

def test_down_sample():
    filename = "flightTracks.mat"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, filename)
    data = load_mat(path)
    points = np.hstack((data['x'].flatten()[:,None],
                        data['y'].flatten()[:,None]))

    sizeList = [100,200,500,2000]
    for size in sizeList:
        ind = down_sample(points, size)
        assert ind.shape == (size,)

    ind = down_sample(points, 4000)
    assert ind.shape == (3129,)
