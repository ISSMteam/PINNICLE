import PINN_ICE as pinn
import pytest

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

#def test_parameters_variable_match_data():
#    with pytest.raises(Exception):
#        d = pinn.Parameters({"datasize":{"u":100}, "output_variables":["v"], "output_size":1})
