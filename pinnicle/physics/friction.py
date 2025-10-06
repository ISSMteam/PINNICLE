import deepxde as dde
from deepxde.backend import jax
from . import EquationBase, Constants
from ..parameter import EquationParameter
from ..utils import slice_column, jacobian, slice_function_jax

# Weertman {{{
class WeertmanFrictionParameter(EquationParameter, Constants):
    """ parameters for Weertman Friction law
    """
    _EQUATION_TYPE = 'Weertman' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u', 'v', 'C', 'taub']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-8, 1.0e-10]
        self.residuals = ["f"+self._EQUATION_TYPE]
        self.pde_weights = [1.0e-10]

        # scalar variables: name:value
        self.scalar_variables = {
                'm': 3.0,               # exponent of friction law
                }
class WeertmanFriction(EquationBase): #{{{
    """ Weertman Friciton law
    """
    _EQUATION_TYPE = 'Weertman' 
    def __init__(self, parameters=WeertmanFrictionParameter()):
        super().__init__(parameters)
    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ residual of Weertman friction law

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # get the ids
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]

        uid = self.local_output_var["u"]
        vid = self.local_output_var["v"]
        Cid = self.local_output_var["C"]
        tauid = self.local_output_var["taub"]

        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        C = slice_column(nn_output_var, Cid)
        taub = slice_column(nn_output_var, tauid)

        # compute the basal stress
        u_norm = (u**2+v**2+self.eps**2)**0.5
        f1 = C**2*(u_norm)**(1.0/self.m) - taub

        return [f1] #}}}
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of Weertman friction law

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}
