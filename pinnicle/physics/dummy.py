import deepxde as dde
from . import EquationBase, Constants
from ..parameter import EquationParameter

class DummyEquationParameter(EquationParameter, Constants):
    """ default parameters for Dummy PDE
    """
    _EQUATION_TYPE = 'DUMMY' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = []
        self.output_lb = []
        self.output_ub = []
        self.data_weights = []
        self.residuals = []
        self.pde_weights = []
        self.scalar_variables = {}

    def update(self):
        """ set all the weights to 1, and load all the lb and ub is not given
        """
        if not self.data_weights:
            self.data_weights = [1.0 for ou in self.output]
        if not self.output_lb:
            self.output_lb = [self.variable_lb[k] for k in self.output]
        if not self.output_ub:
            self.output_ub = [self.variable_ub[k] for k in self.output]

    def check_consistency(self):
        """ output can not be empty
        """
        if len(self.output) != len(self.output_lb):
            raise ValueError("The size of the output is not consistent with size of the lower bound")
        if len(self.output) != len(self.output_ub):
            raise ValueError("The size of the output is not consistent with size of the upper bound")

class Dummy(EquationBase): #{{{
    """ 
    """
    _EQUATION_TYPE = 'DUMMY' 
    def __init__(self, parameters=DummyEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var):
        """ Dummy PDE returns nothing, to train the NN with data only

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return [] 
    def _pde_jax(self, nn_input_var, nn_output_var):
        """ Dummy PDE returns nothing, to train the NN with data only

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return [] #}}}
