import pytest
import os
import pinnicle as pinn
from pinnicle.modeldata import ISSMmdData, MatData, Data, H5Data
from pinnicle.parameter import DataParameter, SingleDataParameter

def test_ISSMmdData():
    filename = "Helheim_fastflow.mat"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, filename)
    
    hp = {}
    hp["data_path"] = path
    hp["data_size"] = {"u":4000, "v":4000, "s":4000, "H":4000, "C":None, "a":500}
    p = SingleDataParameter(hp)
    data_loader = ISSMmdData(p)
    data_loader.load_data()
    data_loader.prepare_training_data()

    assert(data_loader.sol['u'].shape == (4000,1))
    assert(data_loader.X['v'].shape == (4000,2))
    assert(data_loader.sol['s'].shape == (4000,1))
    assert(data_loader.X['H'].shape == (4000,2))
    assert(data_loader.sol['C'].shape == (278,1))
    assert(data_loader.sol['a'].shape == (500,1))

    iice = data_loader.get_ice_indices()
    assert iice[0].shape == (11874,)
    icoord = data_loader.get_ice_coordinates()
    assert icoord.shape == (11874, 2)

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
    assert(data_loader.sol['C'].shape == (278,1))

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
    issm2["data_size"] = {"u":400, "v":None, "s":1000, "C":1000, "a":1000}
    issm2["default_time"] = 1

    hp = {}
    hp['data'] = {"issm":issm, "issm2":issm2}

    p = DataParameter(hp)
    data_loader = Data(p)
    data_loader.load_data()
    data_loader.prepare_training_data()

    assert(data_loader.sol['u'].shape == (4400,1))
    assert(data_loader.X['v'].shape == (4278,2))
    assert(data_loader.sol['s'].shape == (5000,1))
    assert(data_loader.X['H'].shape == (4000,2))
    assert(data_loader.sol['C'].shape == (1278,1))
    assert(data_loader.sol['a'].shape == (1000,1))

    icoord = data_loader.get_ice_coordinates()
    assert icoord.shape == (11874*2, 2)

    p = DataParameter(hp)
    data_loader = Data(p)
    data_loader.load_data()
    data_loader.prepare_training_data(transient=True, default_time=10)
    assert(data_loader.sol['v'].shape == (4278,1))
    assert(data_loader.X['u'].shape == (4400,3))
    assert(data_loader.X['u'][1,2] == 10)
    assert(data_loader.X['u'][-1,2] == 1)
    icoord = data_loader.get_ice_coordinates()
    assert icoord.shape == (11874*2, 2)

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

    hp["X_map"] = {"notload":"t", "y":"y"}
    p = SingleDataParameter(hp)
    data_loader = MatData(p)
    data_loader.load_data()
    assert ("y" in data_loader.X_dict)
    assert ("t" not in data_loader.X_dict)
    assert ("notload" not in data_loader.X_dict)

def test_MatData_domain():
    filename = "flightTracks.mat"
    expFileName = "fastflow_CF.exp"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, filename)
    
    hp = {}
    hp["data_path"] = path
    hp["data_size"] = {"H":100}
    hp["name_map"] = {"H":"thickness"}
    hp["source"] = "mat"
    hp["X_map"] = {"x1":"x", "x2":"y"}
    hp["shapefile"] = os.path.join(repoPath, "dataset", expFileName)

    d = pinn.domain.Domain( pinn.parameter.DomainParameter(hp))
    p = SingleDataParameter(hp)
    data_loader = MatData(p)
    data_loader.load_data(d)
    data_loader.prepare_training_data()
    assert(data_loader.X['H'].shape == (100,2))
    assert(data_loader.sol['H'].shape == (100,1))
    icoord = data_loader.get_ice_coordinates()
    assert icoord.shape == (673,2)

def test_MatData_physics():
    filename = "flightTracks.mat"
    expFileName = "fastflow_CF.exp"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, filename)

    hp = {}
    hp["data_path"] = path
    hp["data_size"] = {"H":100}
    hp["name_map"] = {"H":"thickness"}
    hp["source"] = "mat"
    hp["X_map"] = {"x1":"x", "x2":"y"}
    hp["shapefile"] = os.path.join(repoPath, "dataset", expFileName)
    hp["equations"] = {"SSA":{"input":["x1", "x2"]}}

    phy = pinn.physics.Physics(pinn.parameter.PhysicsParameter(hp))
    p = SingleDataParameter(hp)
    data_loader = MatData(p)
    data_loader.load_data(physics=phy)
    data_loader.prepare_training_data()
    assert(data_loader.X['H'].shape == (100,2))
    assert(data_loader.sol['H'].shape == (100,1))
    icoord = data_loader.get_ice_coordinates()
    assert icoord.shape == (3192,2)

def test_h5Data():
    filename = "subdomain_data.h5"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, filename)

    hp = {}
    hp["data_path"] = path
    hp["data_size"] = {"u":300, "v":100, "s":2, "b":100, "a":10}
    hp["X_map"] = {"x":"surf_x", "y":"surf_y" }
    hp["name_map"] = {"s":"surf_elv", "u":"surf_vx", "v":"surf_vy", "a":"surf_SMB", "b":"bed_BedMachine"}
    hp["source"] = "h5"
    
    p = SingleDataParameter(hp)
    data_loader = H5Data(p)
    data_loader.load_data()
    data_loader.prepare_training_data()
    assert(data_loader.sol['u'].shape == (300,1))
    assert(data_loader.X['v'].shape == (100,2))
    icoord = data_loader.get_ice_coordinates()
    assert icoord.shape == (60000, 2)

def test_h5Data_domain():
    filename = "subdomain_data.h5"
    expFileName = "fastflow_CF.exp"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, filename)

    hp = {}
    hp["data_path"] = path
    hp["data_size"] = {"u":300, "v":100, "s":2, "b":100, "a":10}
    hp["X_map"] = {"x":"surf_x", "y":"surf_y" }
    hp["name_map"] = {"s":"surf_elv", "u":"surf_vx", "v":"surf_vy", "a":"surf_SMB", "b":"bed_BedMachine"}
    hp["source"] = "h5"
    hp["shapefile"] = os.path.join(repoPath, "dataset", expFileName)

    d = pinn.domain.Domain( pinn.parameter.DomainParameter(hp))
    p = SingleDataParameter(hp)
    data_loader = H5Data(p)
    data_loader.load_data(d)
    data_loader.prepare_training_data()
    assert(data_loader.X['b'].shape == (100,2))
    assert(data_loader.sol['a'].shape == (10,1))
    icoord = data_loader.get_ice_coordinates()
    assert icoord.shape == (51800,2)

def test_h5Data_physics():
    filename = "subdomain_data.h5"
    expFileName = "fastflow_CF.exp"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, filename)

    hp = {}
    hp["data_path"] = path
    hp["data_size"] = {"u":300, "v":100, "s":2, "b":100, "a":10}
    hp["X_map"] = {"x":"surf_x", "y":"surf_y" }
    hp["name_map"] = {"s":"surf_elv", "u":"surf_vx", "v":"surf_vy", "a":"surf_SMB", "b":"bed_BedMachine"}
    hp["source"] = "h5"
    hp["shapefile"] = os.path.join(repoPath, "dataset", expFileName)
    hp["equations"] = {"SSA":{}}

    phy = pinn.physics.Physics(pinn.parameter.PhysicsParameter(hp))
    p = SingleDataParameter(hp)
    data_loader = H5Data(p)
    data_loader.load_data(physics=phy)
    data_loader.prepare_training_data()
    assert(data_loader.X['b'].shape == (100,2))
    assert(data_loader.sol['a'].shape == (10,1))
    icoord = data_loader.get_ice_coordinates()
    assert icoord.shape == (60000,2)
