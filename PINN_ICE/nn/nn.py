import deepxde as dde
from .helper import minmax_scale, up_scale
from ..parameters import NNParameter

class NN:
    def __init__(self, parameters=NNParameter()):
        """
        general class for constructing nerual network
        """
        self.parameters = parameters

        # create new NN
        self.net = self.createFNN()

        # apply transform
        # by default, use min-max scale for the input
        if self.parameters.is_input_scaling():
            print(f"add input transform with {self.parameters.input_lb} and {self.parameters.input_ub}")
            self._add_input_transform(minmax_scale)

        # upscale the output by min-max
        if self.parameters.is_output_scaling():
            print(f"add output transform with {self.parameters.output_lb} and {self.parameters.output_ub}")
            self._add_output_transform(up_scale)

    def createFNN(self):
        """
        create a fully connected neural network
        """
        layer_size = [self.parameters.input_size] + [self.parameters.num_neurons] * self.parameters.num_layers + [self.parameters.output_size]
        return dde.nn.FNN(layer_size, self.parameters.activation, self.parameters.initializer)
        
    def _add_input_transform(self, func):
        """
        a function to add scaling at the input
        """
        def _wrapper(x):
            return func(x,self.parameters.input_lb, self.parameters.input_ub)
        self.net.apply_feature_transform(_wrapper)

    def _add_output_transform(self, func):
        """
        a function to add scaling at the output
        """
        def _wrapper(dummy, x):
            return  func(x, self.parameters.output_lb, self.parameters.output_ub)
        self.net.apply_output_transform(_wrapper)

