import deepxde as dde
from . import EquationBase, Constants
from ..parameter import EquationParameter

class BCEquationParameter(EquationParameter, Constants):
    """ default parameters for Boundary Conditions
    """
    _EQUATION_TYPE = 'BC' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['nx', 'ny']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e1, 1.0e1]
        self.residuals = []
        self.pde_weights = []
        self.scalar_variables = {}

class BC(EquationBase): #{{{
    """ 
    """
    _EQUATION_TYPE = 'BC' 
    def __init__(self, parameters=BCEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var):
        """ BC PDE returns nothing, to train the NN with data only

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return [] 
    def _pde_jax(self, nn_input_var, nn_output_var):
        """ BC PDE returns nothing, to train the NN with data only

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return [] #}}}
