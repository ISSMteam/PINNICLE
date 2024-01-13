from PINN_ICE.nn.helper import minmax_scale, up_scale
import numpy as np

def test_minmax_scale():
    lb = 1
    ub = 10
    x = np.linspace(lb, ub, 100)
    y = minmax_scale(x, lb, ub)
    assert np.all(abs(y- np.linspace(-1, 1, 100)) < np.finfo(float).eps*ub)

def test_upscale():
    lb = 1
    ub = 10
    x = np.linspace(-1, 1, 100)
    y = up_scale(x, lb, ub)
    assert np.all(abs(y- np.linspace(lb, ub, 100)) < np.finfo(float).eps*ub)
