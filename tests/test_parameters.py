import pytest
import numpy as np
import pinnicle as pinn
from pinnicle.parameter import DataParameter, SingleDataParameter, NNParameter, DomainParameter, PhysicsParameter, Parameters, EquationParameter, TrainingParameter
from pinnicle.physics import SSAEquationParameter, DummyEquationParameter

yts = 3600*24*365.0

def test_domain_parameter():
    d = DomainParameter()
    assert hasattr(d, "param_dict"), "Default attribute 'param_dict' not found"

    newat = {"feature_not_exist_1":1, "feature_not_exist_2": [2,3,4]}
    d.set_parameters(newat)
    assert not d.has_keys(newat)

    d._add_parameters(newat)
    assert d.has_keys(newat)

    timedep = {'time_dependent':True, 'start_time':0, 'end_time':10}
    d = DomainParameter(timedep)

    assert d.end_time == 10
    with pytest.raises(Exception):
        timedep = {'time_dependent':True, 'start_time':10, 'end_time':1}
        d = DomainParameter(timedep)

def test_single_data_parameter():
    issm = {"data_path":"./", "data_size":{"u":4000, "v":None}}
    d = SingleDataParameter(issm)
    assert hasattr(d, "param_dict"), "Default attribute 'param_dict' not found"
    assert d.name_map["u"] == "u"
    assert d.name_map["v"] == "v"
    assert d.source == "ISSM"
    assert [d.X_map[k] == k for k in ["x","y","t"]]
    assert d.default_time is None

    mat = {"data_path":"./", "data_size":{"u":4000, "v":None}, "source":"mat", "X_map":{"x":"y"}, "default_time":1}
    d = SingleDataParameter(mat)
    assert d.name_map["u"] == "u"
    assert d.name_map["v"] == "v"
    assert d.source == "mat"
    assert d.X_map["x"] == "y"
    assert "y" not in d.X_map
    assert d.default_time == 1

    with pytest.raises(Exception):
        unknown = {"source": "unknown"}
        d = SingleDataParameter(unknown)

def test_data_parameter():
    hp = {}
    issm = {"data_path":"./", "data_size":{"u":4000, "v":None}}
    mat = {"data_path":"./", "data_size":{"u":4000, "v":None}, "source":"mat"}
    hp["data"] = {"mymat": mat, "myISSM": issm}

    d = DataParameter(hp)
    assert hasattr(d, "param_dict"), "Default attribute 'param_dict' not found"
    assert hasattr(d, "data"), "attribute 'data' not found" 
    assert d.data["myISSM"].source == "ISSM"
    assert d.data["mymat"].source == "mat"

def test_nn_parameter():
    d = NNParameter()
    assert hasattr(d, "param_dict"), "Default attribute 'param_dict' not found"

    assert not d.is_input_scaling()
    d.input_lb = 1
    d.input_ub = 10
    assert d.is_input_scaling()
    assert not d.is_output_scaling()

    d.output_lb = 1
    d.output_ub = 10
    assert d.is_output_scaling()

    d = NNParameter({"num_neurons":[1,2,3]})
    assert d.num_layers == 3
    assert d.input_size == 0

    d = NNParameter({"fft":True})
    assert d.input_size == 2*d.num_fourier_feature
    assert d.is_input_scaling()
    assert d.B is None

    d = NNParameter({"fft":True, "num_fourier_feature":4, "B":[[1,2,3,4]]})
    assert d.B is not None
    with pytest.raises(Exception):
        d = NNParameter({"fft":True, "num_fourier_feature":4, "B":1})
        d = NNParameter({"fft":True, "num_fourier_feature":4, "B":[[1,2]]})

def test_parameters():
    p = Parameters()
    domain = DomainParameter()
    data = DataParameter()
    nn = NNParameter()
    physics = PhysicsParameter()
    assert p.domain.__dict__ == domain.__dict__
    assert p.data.__dict__ == data.__dict__
    assert p.nn.__dict__ == nn.__dict__
    assert p.physics.__dict__ == physics.__dict__

def test_equation_parameters():
    SSA = {}
    SSA["input"] = ["x", "y"]
    SSA["output"] = ["u", "v", "s", "H", "C"]
    SSA["output_lb"] = [-1.0e4/yts, -1.0e4/yts, -1.0e3,  10.0, 0.01]
    SSA["output_ub"] = [ 1.0e4/yts,  1.0e4/yts,  2.5e3, 2.0e3, 1.0e4]
    SSA["data_weights"] = [1.0e-8*yts**2.0, 1.0e-8*yts**2.0, 1.0e-6, 1.0e-6, 1.0e-8]
    p = EquationParameter(SSA)
    assert p.input == SSA["input"]
    assert p.output == SSA["output"]
    assert p.data_weights == SSA["data_weights"]

    p = SSAEquationParameter(SSA)
    assert p.scalar_variables['n'] == 3.0
    assert p.scalar_variables['B'] == 1.26802073401e+08

    SSA['scalar_variables'] = {'n':4.0}
    p = SSAEquationParameter(SSA)
    assert p.scalar_variables['n'] == 4.0
    assert p.scalar_variables['B'] == 1.26802073401e+08

    SSA["output_lb"] = [1.0e4/yts, -1.0e4/yts]
    with pytest.raises(Exception):
        p = EquationParameter(SSA)

    SSA["output_lb"] = [1.0e4/yts, -1.0e4/yts, -1.0e3,  10.0, 0.01]
    SSA["output_ub"] = [1.0e4/yts,  1.0e4/yts,  2.5e3, 2.0e3, 1.0e4]
    with pytest.raises(Exception):
        p = EquationParameter(SSA)

    hp = {}
    hp['equations'] = {'SSA': {}}
    p = Parameters(hp)

    hp['equations'] = {'NOT DEFINED': {}}
    with pytest.raises(Exception):
        p = Parameters(hp)

def test_dummy_equation_parameters():
    DUMMY = {}
    DUMMY["input"] = ["x", "y"]
    DUMMY["output"] = ["u", "v", "s", "H", "C"]
    DUMMY["output_lb"] = [-1.0e4/yts, -1.0e4/yts, -1.0e3,  10.0, 0.01]
    DUMMY["output_ub"] = [ 1.0e4/yts,  1.0e4/yts,  2.5e3, 2.0e3, 1.0e4]

    p = DummyEquationParameter(DUMMY)
    assert p.input == DUMMY["input"]
    assert p.output == DUMMY["output"]
    assert p.data_weights == [1.0]*5

    DUMMY["data_weights"] = [1.0e-8*yts**2.0, 1.0e-8*yts**2.0, 1.0e-6, 1.0e-6, 1.0e-8]
    p = DummyEquationParameter(DUMMY)
    assert p.data_weights == DUMMY["data_weights"]

def test_training_parameters():
    hp =  {}
    hp['decay_steps'] = 10
    hp['decay_rate'] = 0.3
    p = TrainingParameter(hp)
    assert p.decay_steps == 10
    assert p.decay_rate == 0.3
    assert p.additional_loss == {}
    u_loss = {}
    u_loss['name'] = "vel log"
    u_loss['function'] = "VEL_LOG"
    u_loss['weight'] = 1.0
    hp['additional_loss'] = {"u": u_loss}
    p = TrainingParameter(hp)
    assert p.additional_loss["u"].name == u_loss['name']

def test_training_callbacks():
    hp = {}
    p = TrainingParameter(hp)
    assert p.has_callbacks == False

def test_training_callbacks_EarlyStopping():
    hp = {}
    hp["min_delta"] = 1
    p = TrainingParameter(hp)
    assert p.has_EarlyStopping() == True
    assert p.patience == 0
    assert p.has_callbacks == True
    hp = {}
    hp["patience"] = 1
    p = TrainingParameter(hp)
    assert p.has_EarlyStopping() == True
    assert p.has_callbacks == True

def test_training_callbacks_Resampler():
    hp = {}
    hp["period"] = 1
    p = TrainingParameter(hp)
    assert p.has_callbacks == True
    assert p.has_PDEPointResampler() == True

def test_training_callbacks_Checkpoint():
    hp = {}
    hp["checkpoint"] = True
    p = TrainingParameter(hp)
    assert p.has_callbacks == True
    assert p.has_ModelCheckpoint() == True

def test_print_parameters(capsys):
    hp = {}
    p = Parameters(hp)
    print(p) 
    captured = capsys.readouterr()
    assert "TrainingParameter" in captured.out
    assert "DomainParameter" in captured.out
    assert "DataParameter" in captured.out
    assert "NNParameter" in captured.out
    assert "PhysicsParameter" in captured.out

