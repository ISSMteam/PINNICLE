import PINN_ICE as pinn

data = {"s":1, "v":[1, 2, 3]}


def test_save_and_load_dict(tmp_path):
    pinn.utils.save_dict_to_json(data, tmp_path, "temp.json")
    pinn.utils.save_dict_to_json(data, tmp_path, "noextension")
    assert data == pinn.utils.load_dict_from_json(tmp_path, "temp.json")
    assert data == pinn.utils.load_dict_from_json(tmp_path, "temp")
    assert data == pinn.utils.load_dict_from_json(tmp_path, "noextension.json")
