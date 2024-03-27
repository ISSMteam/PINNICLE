import os
from PINN_ICE.modeldata import ISSMmdData, Data
from PINN_ICE.parameter import DataParameter, SingleDataParameter

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

    iice = data_loader.get_ice_coordinates()
    assert iice[0].shape == (23049,)

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

