import PINN_ICE as pinn
import pytest
import tensorflow as tf

data = {"s":1, "v":[1, 2, 3]}


def test_save_and_load_dict(tmp_path):
    pinn.utils.save_dict_to_json(data, tmp_path, "temp.json")
    pinn.utils.save_dict_to_json(data, tmp_path, "noextension")
    assert data == pinn.utils.load_dict_from_json(tmp_path, "temp.json")
    assert data == pinn.utils.load_dict_from_json(tmp_path, "temp")
    assert data == pinn.utils.load_dict_from_json(tmp_path, "noextension.json")

def test_data_misfit():
    with pytest.raises(Exception):
        pinn.utils.data_misfit.get("not defined")
    assert pinn.utils.data_misfit.get("MSE") != None
    assert pinn.utils.data_misfit.get("VEL_LOG") != None
    assert pinn.utils.data_misfit.get("MEAN_SQUARE_LOG") != None
    assert pinn.utils.data_misfit.get("MAPE") != None

def test_data_misfit_functions():
    assert  pinn.utils.data_misfit.get("MSE")(tf.convert_to_tensor([1.0]),tf.convert_to_tensor([1.0])) == 0.0
    assert  pinn.utils.data_misfit.get("VEL_LOG")(tf.convert_to_tensor([1.0]),tf.convert_to_tensor([1.0])) == 0.0
    assert  pinn.utils.data_misfit.get("MEAN_SQUARE_LOG")(tf.convert_to_tensor([1.0]),tf.convert_to_tensor([1.0])) == 0.0
    assert  pinn.utils.data_misfit.get("MAPE")(tf.convert_to_tensor([1.0]),tf.convert_to_tensor([1.0])) == 0.0
