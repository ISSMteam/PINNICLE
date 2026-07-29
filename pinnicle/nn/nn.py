import deepxde as dde
import deepxde.backend as bkd
import numpy as np
from deepxde.backend import tf
from .helper import minmax_scale, up_scale, fourier_feature
from ..utils import default_float_type
from ..parameter import NNParameter

class FNN:
    def __init__(self, parameters=NNParameter()):
        """
        general class for constructing nerual network
        """
        self.parameters = parameters
        # update necesarry parameters for fourier feature transform
        # NOTE: these changes will not be saved to the param file, 
        # so that the change will not accumulate and loading the previous param file will create the same nn
        if self.parameters.fft and not self.parameters.time_dependent:
            # Then add an additional layer before the output node
            self.num_neurons = parameters.num_neurons + [parameters.num_fourier_feature*parameters.sigma_size]
            self.num_layers = len(self.num_neurons)
            # append linear transform for the output
            self.activation = self.parameters.activation + [None]

        # Merge space and time-dependent Fourier features in second-to-last layer
        elif self.parameters.fft and self.parameters.time_dependent:
            # TODO: Point-wise multiplication of Fourier features to merge in second-to-last layer
            
            # Add layer before output node
            self.num_neurons = parameters.num_neurons + [parameters.num_fourier_feature*parameters.sigma_size + parameters.num_time_fourier_feature*parameters.time_sigma_size]
            self.num_layers = len(self.num_neurons)
            # append linear transform for the output
            self.activation = self.parameters.activation + [None]
        else:
            # just to avoid modify parameters
            self.num_neurons = self.parameters.num_neurons
            self.num_layers = self.parameters.num_layers
            self.activation = self.parameters.activation

        # create new NN
        if self.parameters.is_parallel:
            self.net = self.createPFNN()
        else:
            self.net = self.createFNN()

        # by default, use min-max scale for the input
        if self.parameters.is_input_scaling():
            # force the input and output lb and ub to be tensors
            if bkd.backend_name == "pytorch" or bkd.backend_name == "paddle":
                self.parameters.input_lb = bkd.as_tensor(self.parameters.input_lb, dtype=default_float_type())
                self.parameters.input_ub = bkd.as_tensor(self.parameters.input_ub, dtype=default_float_type())

            if self.parameters.fft and not self.parameters.time_dependent:
                print(f"add Fourier feature transform to spatial input transform")
                if self.parameters.B is not None: 
                    self.B = bkd.as_tensor(self.parameters.B, dtype=default_float_type())
                else:
                    self.B = bkd.as_tensor(
                            np.reshape(np.random.normal(0.0, self.parameters.sigma, [len(self.parameters.input_variables), self.parameters.num_fourier_feature, self.parameters.sigma_size]), [len(self.parameters.input_variables), self.parameters.num_fourier_feature*self.parameters.sigma_size]),
                            dtype=default_float_type())
                def wrapper(x):
                    """a wrapper function to add fourier feature transform to the spatial input
                    """
                    return fourier_feature(minmax_scale(x, self.parameters.input_lb, self.parameters.input_ub), self.B)
                # add to input transform
                self.net.apply_feature_transform(wrapper)
            elif self.parameters.fft and self.parameters.time_dependent:
                print(f"add Fourier feature transform to spatial and temporal input transform")
                # Spatial features
                if self.parameters.B is not None: 
                    self.B = bkd.as_tensor(self.parameters.B, dtype=default_float_type())
                else:
                    space_len = len([var for var in self.parameters.input_variables if var == 'x' or var == 'y'])
                    self.B = bkd.as_tensor(
                            np.reshape(np.random.normal(0.0, self.parameters.sigma, [space_len, self.parameters.num_fourier_feature, self.parameters.sigma_size]), [space_len, self.parameters.num_fourier_feature*self.parameters.sigma_size]),
                            dtype=default_float_type())
                
                # Temporal features
                if self.parameters.time_B is not None: 
                    self.time_B = bkd.as_tensor(self.parameters.time_B, dtype=default_float_type())
                else:
                    time_len = len([var for var in self.parameters.input_variables if var == 't'])
                    self.time_B = bkd.as_tensor(
                            np.reshape(np.random.normal(0.0, self.parameters.time_sigma, [time_len, self.parameters.num_time_fourier_feature, self.parameters.time_sigma_size]), [time_len, self.parameters.num_time_fourier_feature*self.parameters.time_sigma_size]),
                            dtype=default_float_type())
                def wrapper(x):
                    """a wrapper function to add Fourier feature transform to the spatial and temporal inputs separately
                    """
                    x_scaled = minmax_scale(x, self.parameters.input_lb, self.parameters.input_ub)
                    x_space = x_scaled[:, :space_len]
                    x_time = x_scaled[:, space_len:]
                    space_features = fourier_feature(x_space, self.B)
                    time_features = fourier_feature(x_time, self.time_B)
                    return bkd.concat([space_features, time_features], 1)
                # add to input transform
                self.net.apply_feature_transform(wrapper)
            else: 
                print(f"add input transform with {self.parameters.input_lb} and {self.parameters.input_ub}")
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
        if isinstance(self.num_neurons, list):
            # directly use the given list of num_neurons
            layer_size = [self.parameters.input_size] + \
                        self.num_neurons + \
                        [self.parameters.output_size]
        else:
            # repeat num_layers times
            layer_size = [self.parameters.input_size] + \
                        [self.num_neurons] * self.num_layers + \
                        [self.parameters.output_size]

        return dde.nn.FNN(layer_size, self.activation, self.parameters.initializer)

    def createPFNN(self):
        """
        create a parallel fully connected neural network
        """
        if isinstance(self.num_neurons, list):
            layer_size = [self.parameters.input_size] + \
                        [[n]*self.parameters.output_size for n in self.num_neurons] + \
                        [self.parameters.output_size]
        else:
            layer_size = [self.parameters.input_size] + \
                        [[self.num_neurons]*self.parameters.output_size] * self.num_layers + \
                        [self.parameters.output_size]
        return dde.nn.PFNN(layer_size, self.activation, self.parameters.initializer)
        
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
