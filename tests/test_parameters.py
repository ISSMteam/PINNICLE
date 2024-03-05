import pytest
import PINN_ICE as pinn
from PINN_ICE.parameter import DataParameter, NNParameter, DomainParameter, PhysicsParameter, Parameters, EquationParameter, TrainingParameter
from PINN_ICE.physics import SSAEquationParameter

yts = 3600*24*365.0

def test_domain_parameter():
    d = DomainParameter()
    assert hasattr(d, "param_dict"), "Default attribute 'param_dict' not found"

    newat = {"feature_not_exist_1":1, "feature_not_exist_2": [2,3,4]}
    d.set_parameters(newat)
    assert not d.has_keys(newat)

    d._add_parameters(newat)
    assert d.has_keys(newat)

def test_data_parameter():
    d = DataParameter({"dataname":['u', 'v'], "datasize":[4000, 4000]})
    assert hasattr(d, "param_dict"), "Default attribute 'param_dict' not found"

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

def test_training_parameters():
    hp =  {}
    p = TrainingParameter(hp)
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
    assert p.has_callbacks == True
    hp = {}
    hp["patience"] = 1
    p = TrainingParameter(hp)
    assert p.has_callbacks == True

def test_training_callbacks_Resampler():
    hp = {}
    hp["period"] = 1
    p = TrainingParameter(hp)
    assert p.has_callbacks == True

def test_training_callbacks_Checkpoint():
    hp = {}
    hp["checkpoint"] = True
    p = TrainingParameter(hp)
    assert p.has_callbacks == True
