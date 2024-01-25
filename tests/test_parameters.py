import PINN_ICE as pinn
from PINN_ICE.parameter import *
import pytest
from PINN_ICE.physics import SSAEquationParameter

yts = 3600*24*365.0

def test_domain_parameter():
    d = DomainParameter()
    assert hasattr(d, "param_dict"), "Default attribute 'param_dict' not found"

    newat = {"feature_not_exist_1":1, "feature_not_exist_2": [2,3,4]}
    d.set_parameters(newat)
    assert d.has_keys(newat) == False

    d._add_parameters(newat)
    assert d.has_keys(newat) == True

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
    SSA["output_ub"] = [ 1.0e4/yts,  1.0e4/yts,  2.5e3, 2.0e3, 1.0e4]
    with pytest.raises(Exception):
        p = EquationParameter(SSA)

