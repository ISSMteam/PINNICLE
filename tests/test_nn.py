import PINN_ICE as pinn
from PINN_ICE.nn.helper import minmax_scale, up_scale
from PINN_ICE.parameter import NNParameter
import numpy as np

def test_minmax_scale():
    lb = 1.0
    ub = 10.0
    x = np.linspace(lb, ub, 100)
    y = minmax_scale(x, lb, ub)
    assert np.all(abs(y- np.linspace(-1.0, 1.0, 100)) < np.finfo(float).eps*ub)

def test_upscale():
    lb = 1.0
    ub = 10.0
    x = np.linspace(-1.0, 1.0, 100)
    y = up_scale(x, lb, ub)
    assert np.all(abs(y- np.linspace(lb, ub, 100)) < np.finfo(float).eps*ub)

def test_new_nn():
    p = pinn.nn.FNN()
    d = NNParameter()
    assert (p.parameters.__dict__ == d.__dict__)

def test_input_scale_nn():
    d = NNParameter()
    d.input_lb = 1.0
    d.input_ub = 10.0
    p = pinn.nn.FNN(d)
    x = np.linspace(d.input_lb, d.input_ub, 100)
    assert np.all(abs(p.net._input_transform(x)) < 1.0+np.finfo(float).eps)

def test_output_scale_nn():
    d = NNParameter()
    d.output_lb = 1.0
    d.output_ub = 10.0
    p = pinn.nn.FNN(d)
    x = np.linspace(-1.0, 1.0, 100)
    assert np.all(p.net._output_transform(0, x) > d.output_lb - d.output_lb*np.finfo(float).eps) 
    assert np.all(p.net._output_transform(0, x) < d.output_ub + d.output_ub*np.finfo(float).eps) 
