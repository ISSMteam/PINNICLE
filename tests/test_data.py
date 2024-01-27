import os
from PINN_ICE.modeldata import *
from PINN_ICE.parameter import DataParameter

def test_ISSMmdData():
    filename = "Helheim_fastflow.mat"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    appDataPath = os.path.join(repoPath, "dataset")
    path = os.path.join(appDataPath, filename)
    
    hp = {}
    hp["data_path"] = path
    hp["data_size"] = {"u":4000, "v":4000, "s":4000, "H":4000, "C":None}
    p = DataParameter(hp)
    data_loader = ISSMmdData(p)
    data_loader.load_data()
    data_loader.prepare_training_data()

    assert(data_loader.sol['u'].shape == (4000,1))
    assert(data_loader.X['v'].shape == (4000,2))
    assert(data_loader.sol['s'].shape == (4000,1))
    assert(data_loader.X['H'].shape == (4000,2))
    assert(data_loader.sol['C'].shape == (564,1))
