import pinnicle as pinn
from pinnicle.nn.helper import minmax_scale, up_scale, fourier_feature, default_float_type
from pinnicle.parameter import NNParameter
import deepxde as dde
import deepxde.backend as bkd
from deepxde.backend import backend_name
import pytest
import numpy as np

dde.config.set_default_float('float64')

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

@pytest.mark.skipif(backend_name=="jax", reason="bkd.matmul is not implemented for jax")
def test_fourier_feature():
    x = bkd.reshape(bkd.as_tensor((np.linspace(1,100, 100)), dtype=default_float_type()), [50,2])
    B = bkd.as_tensor(np.random.normal(0.0, 10.0, [x.shape[1], 2]), dtype=default_float_type())
    y = bkd.to_numpy(fourier_feature(x, B))
    z = y**2
    assert np.all((z[:,1]+z[:,3]) < 1.0+100**np.finfo(float).eps)

def test_default_float_type():
    assert default_float_type() is not None
    assert default_float_type() in bkd.data_type_dict.values()
    assert default_float_type() == bkd.data_type_dict['float64']

def test_new_nn():
    hp={}
    hp['input_variables'] = ['x']
    hp['output_variables'] = ['u']
    hp['num_neurons'] = 1
    hp['num_layers'] = 1
    d = NNParameter(hp)
    p = pinn.nn.FNN(d)
    assert (p.parameters.__dict__ == d.__dict__)

@pytest.mark.skipif(backend_name=="jax", reason="bkd.matmul is not implemented for jax")
def test_input_fft_nn():
    hp={}
    hp['input_variables'] = ['x']
    hp['output_variables'] = ['u']
    hp['num_neurons'] = 1
    hp['num_layers'] = 1
    hp['fft'] = True
    d = NNParameter(hp)
    d.input_lb = 1.0
    d.input_ub = 10.0
    p = pinn.nn.FNN(d)
    x = bkd.reshape(bkd.as_tensor(np.linspace(d.input_lb, d.input_ub, 100), dtype=default_float_type()), [100,1])
    y = bkd.to_numpy(p.net._input_transform(x))
    z = y**2
    assert np.all(abs(z[:,1:10]+z[:,11:20]) <= 1.0+np.finfo(float).eps)

    hp['B'] = [[1,2,3]]
    hp['num_fourier_feature'] = 3
    d = NNParameter(hp)
    p = pinn.nn.FNN(d)
    assert np.all(hp['B'] == bkd.to_numpy(p.B))

def test_input_scale_nn():
    hp={}
    hp['input_variables'] = ['x']
    hp['output_variables'] = ['u']
    hp['num_neurons'] = 1
    hp['num_layers'] = 1
    d = NNParameter(hp)
    d.input_lb = 1.0
    d.input_ub = 10.0
    p = pinn.nn.FNN(d)
    x = bkd.as_tensor(np.linspace(d.input_lb, d.input_ub, 100), dtype=default_float_type())
    y = bkd.to_numpy(p.net._input_transform(x))
    assert np.all(abs(y) <= 1.0+np.finfo(float).eps)

def test_output_scale_nn():
    hp={}
    hp['input_variables'] = ['x']
    hp['output_variables'] = ['u']
    hp['num_neurons'] = 1
    hp['num_layers'] = 1
    d = NNParameter(hp)
    d.output_lb = 1.0
    d.output_ub = 10.0
    p = pinn.nn.FNN(d)
    x = bkd.as_tensor(np.linspace(-1.0, 1.0, 100), dtype=default_float_type())
    y = [0.0]
    out = bkd.to_numpy(p.net._output_transform(y,x))
    assert np.all(out >= 1.0 - 1.0*np.finfo(float).eps) 
    assert np.all(out <= 10.0 + 10.0*np.finfo(float).eps) 

def test_pfnn():
    hp={}
    hp['input_variables'] = ['x','y']
    hp['output_variables'] = ['u', 'v','s']
    hp['num_neurons'] = 4
    hp['num_layers'] = 5
    hp['is_parallel'] = False
    d = NNParameter(hp)
    p = pinn.nn.FNN(d)
    if backend_name == "jax":
        assert p.net.layer_sizes == [2, 4, 4, 4, 4, 4, 3]
    elif backend_name == "pytorch":
        assert [k.in_features for k in p.net.linears] == [2, 4, 4, 4, 4, 4]
    else:
        assert len(p.net.layers) == 6
    hp['is_parallel'] = True
    d = NNParameter(hp)
    p = pinn.nn.FNN(d)
    if backend_name == "jax":
        assert p.net.layer_sizes == [2, [4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4], 3]
    elif backend_name == "pytorch":
        assert [[i.in_features for i in k] for k in p.net.layers] == [[2, 2, 2], [4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4]]
    else:
        assert len(p.net.layers) == 18

def test_pfnn_list_neuron():
    hp={}
    hp['input_variables'] = ['x','y']
    hp['output_variables'] = ['u', 'v','s']
    hp['num_neurons'] = [3,4,5]
    hp['num_layers'] = 5
    hp['is_parallel'] = False
    d = NNParameter(hp)
    p = pinn.nn.FNN(d)
    if backend_name == "jax":
        assert p.net.layer_sizes == [2, 3, 4, 5, 3]
    elif backend_name == "pytorch":
        assert [k.in_features for k in p.net.linears] == [2, 3, 4, 5]
    else:
        assert len(p.net.layers) == 4
    hp['is_parallel'] = True
    d = NNParameter(hp)
    p = pinn.nn.FNN(d)
    if backend_name == "jax":
        assert p.net.layer_sizes == [2, [3, 3, 3], [4, 4, 4], [5, 5, 5], 3]
    elif backend_name == "pytorch":
        assert [[i.in_features for i in k] for k in p.net.layers] == [[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]
    else:
        assert len(p.net.layers) == 12
