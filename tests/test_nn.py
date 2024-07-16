import pinnicle as pinn
from pinnicle.nn.helper import minmax_scale, up_scale
from pinnicle.parameter import NNParameter
from deepxde import backend
from deepxde.backend import backend_name
import numpy as np

def test_minmax_scale():
    lb = 1.0
    ub = 10.0
    x = backend.as_tensor(np.linspace(lb, ub, 100))
    y = backend.to_numpy(minmax_scale(x, backend.as_tensor(lb), backend.as_tensor(ub)))
    assert np.all(abs(y- np.linspace(-1.0, 1.0, 100)) < np.finfo(float).eps*ub)

def test_upscale():
    lb = 1.0
    ub = 10.0
    x = backend.as_tensor(np.linspace(-1.0, 1.0, 100))
    y = backend.to_numpy(up_scale(x, backend.as_tensor(lb), backend.as_tensor(ub)))
    assert np.all(abs(y- np.linspace(lb, ub, 100)) < np.finfo(float).eps*ub)

def test_new_nn():
    hp={}
    hp['input_variables'] = ['x']
    hp['output_variables'] = ['u']
    hp['num_neurons'] = 1
    hp['num_layers'] = 1
    d = NNParameter(hp)
    p = pinn.nn.FNN(d)
    assert (p.parameters.__dict__ == d.__dict__)

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
    x = backend.as_tensor(np.linspace(d.input_lb, d.input_ub, 100))
    assert np.all(abs(backend.to_numpy(p.net._input_transform(x))) < 1.0+np.finfo(float).eps)

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
    x = backend.as_tensor(np.linspace(-1.0, 1.0, 100))
    y = backend.as_tensor([0])
    assert np.all(backend.to_numpy(p.net._output_transform(y, x)) > 1.0 - 1.0*np.finfo(float).eps) 
    assert np.all(backend.to_numpy(p.net._output_transform(y, x)) < 10.0 + 10.0*np.finfo(float).eps) 

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
