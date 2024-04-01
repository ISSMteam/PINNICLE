import pytest
import tensorflow as tf
import os
from PINNICLE.utils import save_dict_to_json, load_dict_from_json, data_misfit, load_mat

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
    assert data_misfit.get("MAPE") != None

def test_data_misfit_functions():
    assert  data_misfit.get("MSE")(tf.convert_to_tensor([1.0]),tf.convert_to_tensor([1.0])) == 0.0
    assert  data_misfit.get("VEL_LOG")(tf.convert_to_tensor([1.0]),tf.convert_to_tensor([1.0])) == 0.0
    assert  data_misfit.get("MEAN_SQUARE_LOG")(tf.convert_to_tensor([1.0]),tf.convert_to_tensor([1.0])) == 0.0
    assert  data_misfit.get("MAPE")(tf.convert_to_tensor([1.0]),tf.convert_to_tensor([1.0])) == 0.0

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
