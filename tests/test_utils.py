import PINN_ICE as pinn
import pytest

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
    assert pinn.utils.data_misfit.get("VEL_LOG") != None
    assert pinn.utils.data_misfit.get("MAPE") != None
