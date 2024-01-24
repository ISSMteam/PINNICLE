import PINN_ICE as pinn
import pytest

yts = 3600*24*365.0

def test_domain_parameter():
    d = pinn.parameters.DomainParameter()
    assert hasattr(d, "param_dict"), "Default attribute 'param_dict' not found"

    newat = {"feature_not_exist_1":1, "feature_not_exist_2": [2,3,4]}
    d.set_parameters(newat)
    assert d.has_keys(newat) == False

    d._add_parameters(newat)
    assert d.has_keys(newat) == True

def test_data_parameter():
    d = pinn.parameters.DataParameter({"dataname":['u', 'v'], "datasize":[4000, 4000]})
    assert hasattr(d, "param_dict"), "Default attribute 'param_dict' not found"

def test_nn_parameter():
    d = pinn.parameters.NNParameter()
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
    p = pinn.Parameters()
    domain = pinn.parameters.DomainParameter()
    data = pinn.parameters.DataParameter()
    nn = pinn.parameters.NNParameter()
    physics = pinn.parameters.PhysicsParameter()
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
    p = pinn.parameters.EquationParameter(SSA)
    assert p.input == SSA["input"]
    assert p.output == SSA["output"]

    SSA["output_lb"] = [1.0e4/yts, -1.0e4/yts]
    with pytest.raises(Exception):
        p = pinn.parameters.EquationParameter(SSA)

    SSA["output_lb"] = [1.0e4/yts, -1.0e4/yts, -1.0e3,  10.0, 0.01]
    SSA["output_ub"] = [ 1.0e4/yts,  1.0e4/yts,  2.5e3, 2.0e3, 1.0e4]
    with pytest.raises(Exception):
        p = pinn.parameters.EquationParameter(SSA)
