import deepxde as dde
import deepxde.backend as bkd
import numpy as np
from deepxde.backend import tf
from .helper import minmax_scale, up_scale, fourier_feature, default_float_type
from ..parameter import NNParameter

class FNN:
    def __init__(self, parameters=NNParameter()):
        """
        general class for constructing nerual network
        """
        self.parameters = parameters

        # create new NN
        if self.parameters.is_parallel:
            self.net = self.createPFNN()
        else:
            self.net = self.createFNN()

        # by default, use min-max scale for the input
        if self.parameters.is_input_scaling():
            if self.parameters.fft :
                print(f"add Fourier feature transform to input transform")
                if self.parameters.B is not None: 
                    self.B = bkd.as_tensor(self.parameters.B, dtype=default_float_type())
                else:
                    self.B = bkd.as_tensor(np.random.normal(0.0, self.parameters.sigma, [len(self.parameters.input_variables), self.parameters.num_fourier_feature]), dtype=default_float_type())
                def wrapper(x):
                    """a wrapper function to add fourier feature transform to the input
                    """
                    return fourier_feature(minmax_scale(x, self.parameters.input_lb, self.parameters.input_ub), self.B)
                # add to input transform
                self.net.apply_feature_transform(wrapper)
            else: 
                print(f"add input transform with {self.parameters.input_lb} and {self.parameters.input_ub}")
                # force the input and output lb and ub to be tensors
                if bkd.backend_name == "pytorch":
                    self.parameters.input_lb = bkd.as_tensor(self.parameters.input_lb, dtype=default_float_type())
                    self.parameters.input_ub = bkd.as_tensor(self.parameters.input_ub, dtype=default_float_type())
                # add input transform
                self._add_input_transform(minmax_scale)

        # upscale the output by min-max
        if self.parameters.is_output_scaling():
            print(f"add output transform with {self.parameters.output_lb} and {self.parameters.output_ub}")
            # force the input and output lb and ub to be tensors
            if bkd.backend_name == "pytorch":
                self.parameters.output_lb = bkd.as_tensor(self.parameters.output_lb, dtype=default_float_type())
                self.parameters.output_ub = bkd.as_tensor(self.parameters.output_ub, dtype=default_float_type())
            # add output transform
            self._add_output_transform(up_scale)

    def createFNN(self):
        """
        create a fully connected neural network
        """
        if isinstance(self.parameters.num_neurons, list):
            # directly use the given list of num_neurons
            layer_size = [self.parameters.input_size] + \
                        self.parameters.num_neurons + \
                        [self.parameters.output_size]
        else:
            # repeat num_layers times
            layer_size = [self.parameters.input_size] + \
                        [self.parameters.num_neurons] * self.parameters.num_layers + \
                        [self.parameters.output_size]

        return dde.nn.FNN(layer_size, self.parameters.activation, self.parameters.initializer)

    def createPFNN(self):
        """
        create a parallel fully connected neural network
        """
        if isinstance(self.parameters.num_neurons, list):
            layer_size = [self.parameters.input_size] + \
                        [[n]*self.parameters.output_size for n in self.parameters.num_neurons] + \
                        [self.parameters.output_size]
        else:
            layer_size = [self.parameters.input_size] + \
                        [[self.parameters.num_neurons]*self.parameters.output_size] * self.parameters.num_layers + \
                        [self.parameters.output_size]
        return dde.nn.PFNN(layer_size, self.parameters.activation, self.parameters.initializer)
        
    def _add_input_transform(self, func):
        """
        a wrapper function to add scaling at the input
        """
        def _wrapper(x):
            return func(x, self.parameters.input_lb, self.parameters.input_ub)
        self.net.apply_feature_transform(_wrapper)

    def _add_output_transform(self, func):
        """
        a wrapper function to add scaling at the output
        """
        def _wrapper(dummy, x):
            return  func(x, self.parameters.output_lb, self.parameters.output_ub)
        self.net.apply_output_transform(_wrapper)

