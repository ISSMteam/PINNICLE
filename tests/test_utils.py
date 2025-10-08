import pytest
import os
import numpy as np
from deepxde import backend
import pinnicle
from pinnicle.utils import *

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
    dde_loss = ["mean absolute error", "MAE", "mae", "mean squared error", "mse", "mean l2 relative error", "softmax cross entropy", "zero"]
    for l in dde_loss:
        assert data_misfit.get(l) == l

def test_data_misfit_functions():
    assert data_misfit.get("VEL_LOG") != None
    assert data_misfit.get("MEAN_SQUARE_LOG") != None
    assert data_misfit.get("MAPE") != None
    assert data_misfit.get("VEL_LOG")(backend.as_tensor([1.0]),backend.as_tensor([1.0])) == 0.0
    assert data_misfit.get("MEAN_SQUARE_LOG")(backend.as_tensor([1.0]),backend.as_tensor([1.0])) == 0.0
    assert data_misfit.get("MAPE")(backend.as_tensor([1.0]),backend.as_tensor([1.0])) == 0.0

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
    ind = down_sample(points, "MAX")
    assert ind.shape == (3129,)

def test_slice_column():
    a = np.array([[1,2],[3,4]])
    c = pinnicle.utils.backends_specified.slice_column_tf(a, 1)
    assert c.shape == (2, 1)
    assert c[1] == 4
    c = pinnicle.utils.backends_specified.slice_column_jax(a, 1)
    assert c.shape == (1,)
    assert c[0] == 2

def test_ppow():
    a = backend.as_tensor([2.0])
    c = pinnicle.utils.backends_specified.ppow(a, 2.0)
    assert c == 4.0


def test_interpfrombedmachine():
    x = np.array([300025,301025,302025])
    y = np.array([-2579975, -2578975, -2577975])
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, "subdomain_bed.nc")
    var_interp = interpfrombedmachine(x, y, "thickness", path)
    assert var_interp.shape == (3,)
    x = np.array([330025])
    y = np.array([-2579975])
    with pytest.raises(Exception):
        var_interp = interpfrombedmachine(x, y, "bed", path)

def test_createsubdomain():
    xmin = 300000
    ymin = -2580000
    xid = 1
    yid = 1
    dx = 50000
    dy = 50000
    x0, x1, y0, y1 = createsubdomain(xmin, ymin, xid, yid, dx, dy)
    assert x1 == 400000
    assert y1 == -2480000
    assert x0 == 350000
    assert y0 == -2530000
    x0, x1, y0, y1 = createsubdomain(xmin, ymin, xid, yid, dx, dy, 0.1)
    assert x1 == 405000
    assert y1 == -2475000
    assert x0 == 345000
    assert y0 == -2535000
    with pytest.raises(Exception):
        createsubdomain(xmin, ymin, xid, yid, dx, dy, 1.1)
    with pytest.raises(Exception):
        createsubdomain(xmin, ymin, xid, yid, dx, dy, -0.1)

def test_subdomainmask():
    xmin = 300000
    ymin = -2580000
    xmax = 400000
    ymax = -2480000
    subdomain = (xmin, xmax, ymin,ymax)
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, "subdomain_bed.nc")
    assert subdomainmask(subdomain, path) == True

def test_rect_weights():
    X, Y = np.meshgrid(np.linspace(-1, 2, 5), np.linspace(-1, 2, 5))
    W = feathered_rect_weights(X.flatten(), Y.flatten(), rect=(0, 1, 0, 1), width=0.5)
    assert W[1] == 0.0
    assert W[12] == 1.0
    assert W[13] == 0.5
    assert W[24] == 0.0
    
