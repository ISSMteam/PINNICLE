import deepxde as dde
from .helper import minmax_scale, up_scale

class NN:
    def __init__(self, input_size=2, output_size=0, num_neurons=0, num_layers=0, activation='tanh', initializer='Glorot uniform', 
            input_lb=None, input_ub=None, output_lb=None, output_ub=None):
        """
        general class for constructing nerual network
        """
        self.input_size = input_size
        self.output_size = output_size
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.activation = activation
        self.initializer = initializer

        #  scaling parameters
        self.input_lb = input_lb
        self.input_ub = input_ub
        self.output_lb = output_lb
        self.output_ub = output_ub

        # create new NN
        self.net = self.createFNN()

        # apply transform
        # by default, use min-max scale for the input
        if (self.input_lb is not None) and (self.input_ub is not None):
            self._add_input_transform(minmax_scale)

        # upscale the output by min-max
        if (self.output_lb is not None) and (self.output_ub is not None):
            self._add_output_transform(up_scale)

    def createFNN(self):
        """
        create a fully connected neural network
        """
        layer_size = [self.input_size] + [self.num_neurons] * self.num_layers + [self.output_size]
        return dde.nn.FNN(layer_size, self.activation, self.initializer)
        
    def _add_input_transform(self, func):
        """
        a function to add scaling at the input
        """
        def _wrapper(x):
            return func(x,self.input_lb, self.input_ub)
        self.net.apply_feature_transform(_wrapper)

    def _add_output_transform(self, func):
        """
        a function to add scaling at the output
        """
        def _wrapper(dummy, x):
            return  func(x, self.output_lb, self.output_ub)
        self.net.apply_output_transform(_wrapper)

