import pytest
import os
from pinnicle.modeldata import ISSMmdData, MatData, Data
from pinnicle.parameter import DataParameter, SingleDataParameter

def test_ISSMmdData():
    filename = "Helheim_fastflow.mat"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, filename)
    
    hp = {}
    hp["data_path"] = path
    hp["data_size"] = {"u":4000, "v":4000, "s":4000, "H":4000, "C":None}
    p = SingleDataParameter(hp)
    data_loader = ISSMmdData(p)
    data_loader.load_data()
    data_loader.prepare_training_data()

    assert(data_loader.sol['u'].shape == (4000,1))
    assert(data_loader.X['v'].shape == (4000,2))
    assert(data_loader.sol['s'].shape == (4000,1))
    assert(data_loader.X['H'].shape == (4000,2))
    assert(data_loader.sol['C'].shape == (564,1))

    iice = data_loader.get_ice_indices()
    assert iice[0].shape == (23049,)
    icoord = data_loader.get_ice_coordinates()
    assert icoord.shape == (23049, 2)

def test_ISSMmdData_plot():
    filename = "Helheim_fastflow.mat"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, filename)

    hp = {}
    hp["data_path"] = path
    p = SingleDataParameter(hp)
    data_loader = ISSMmdData(p)
    data_loader.load_data()
    _, _, _, axs = data_loader.plot(resolution=10)
    assert len(axs) == len(data_loader.data_dict.keys())
    data_names = ['u','v','s']
    X, Y, im_data, axs = data_loader.plot(data_names=data_names,resolution=10)
    assert len(axs) == len(data_names)
    assert X.shape == (10,10)
    assert Y.shape == (10,10)
    assert len(im_data) == len(data_names)
    assert im_data['u'].shape == (10,10)

def test_Data():
    filename = "Helheim_fastflow.mat"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, filename)
    
    issm = {}
    issm["data_path"] = path
    issm["data_size"] = {"u":4000, "v":4000, "s":4000, "H":4000, "C":None}
    hp = {}
    hp['data'] = {"issm":issm}

    p = DataParameter(hp)
    data_loader = Data(p)
    data_loader.load_data()
    data_loader.prepare_training_data()

    assert(data_loader.sol['u'].shape == (4000,1))
    assert(data_loader.X['v'].shape == (4000,2))
    assert(data_loader.sol['s'].shape == (4000,1))
    assert(data_loader.X['H'].shape == (4000,2))
    assert(data_loader.sol['C'].shape == (564,1))

def test_Data_multiple():
    filename = "Helheim_fastflow.mat"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, filename)
    
    issm = {}
    issm["data_path"] = path
    issm["data_size"] = {"u":4000, "v":4000, "s":4000, "H":4000, "C":None}
    issm2 = {}
    issm2["data_path"] = path
    issm2["data_size"] = {"u":400, "v":None, "s":1000, "C":1000}
    issm2["default_time"] = 1

    hp = {}
    hp['data'] = {"issm":issm, "issm2":issm2}

    p = DataParameter(hp)
    data_loader = Data(p)
    data_loader.load_data()
    data_loader.prepare_training_data()

    assert(data_loader.sol['u'].shape == (4400,1))
    assert(data_loader.X['v'].shape == (4564,2))
    assert(data_loader.sol['s'].shape == (5000,1))
    assert(data_loader.X['H'].shape == (4000,2))
    assert(data_loader.sol['C'].shape == (1564,1))

    icoord = data_loader.get_ice_coordinates()
    assert icoord.shape == (23049*2, 2)

    p = DataParameter(hp)
    data_loader = Data(p)
    data_loader.load_data()
    data_loader.prepare_training_data(transient=True, default_time=10)
    assert(data_loader.sol['v'].shape == (4564,1))
    assert(data_loader.X['u'].shape == (4400,3))
    assert(data_loader.X['u'][1,2] == 10)
    assert(data_loader.X['u'][-1,2] == 1)
    icoord = data_loader.get_ice_coordinates()
    assert icoord.shape == (23049*2, 2)

def test_MatData():
    filename = "flightTracks.mat"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, filename)

    hp = {}
    hp["data_path"] = path
    hp["data_size"] = {"H":100}
    hp["name_map"] = {"H":"thickness"}
    hp["source"] = "mat"
    hp["X_map"] = {"x1":"x", "x2":"y"}
    p = SingleDataParameter(hp)
    data_loader = MatData(p)
    data_loader.load_data()
    data_loader.prepare_training_data()

    assert(data_loader.sol['H'].shape == (100,1))
    assert(data_loader.X['H'].shape == (100,2))

    icoord = data_loader.get_ice_coordinates()
    assert icoord.shape == (3192, 2)

    hp["X_map"] = {"x":"x"}
    p = SingleDataParameter(hp)
    data_loader = MatData(p)
    data_loader.load_data()
    data_loader.prepare_training_data()
    assert(data_loader.X['H'].shape[1] == 1)

    hp["X_map"] = {"x":"t", "y":"y"}
    p = SingleDataParameter(hp)
    data_loader = MatData(p)
    data_loader.load_data()
    assert ("y" in data_loader.X_dict)
    assert ("x" not in data_loader.X_dict)
