from . import EquationBase, Constants
from ..parameter import EquationParameter
from ..utils import slice_column, jacobian

# static mass conservation {{{
class MCEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'MC' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u', 'v', 'a', 'H']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2, 1.0e-8*self.yts**2, 1.0*self.yts**2, 1.0e-6]
        self.residuals = ["f"+self._EQUATION_TYPE]
        self.pde_weights = [1.0e10]

        # scalar variables: name:value
        self.scalar_variables = {}
class MC(EquationBase): #{{{
    """ MC on 2D problem
    """
    _EQUATION_TYPE = 'MC' 
    def __init__(self, parameters=MCEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ residual of MC 2D PDE

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # get the ids
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]

        uid = self.local_output_var["u"]
        vid = self.local_output_var["v"]
        aid = self.local_output_var["a"]
        Hid = self.local_output_var["H"]

        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        a = slice_column(nn_output_var, aid)
        H = slice_column(nn_output_var, Hid)

        # spatial derivatives
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        H_x = jacobian(nn_output_var, nn_input_var, i=Hid, j=xid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        H_y = jacobian(nn_output_var, nn_input_var, i=Hid, j=yid)

        # residual
        f = H*u_x + H_x*u + H*v_y + H_y*v - a

        return [f] #}}}
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of MC 2D PDE, jax version

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}
# time dependent mass conservation {{{
class ThicknessEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'Mass transport'
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y', 't']
        self.output = ['u', 'v', 'a', 'H']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2, 1.0e-8*self.yts**2, 1.0e-2*self.yts**2, 1.0e-6]
        self.residuals = ["f"+self._EQUATION_TYPE]
        self.pde_weights = [1.0e10]

        # scalar variables: name:value
        self.scalar_variables = {
                }
class Thickness(EquationBase): #{{{
    """ 2D time depenent thickness evolution
    """
    _EQUATION_TYPE = 'Mass transport'
    def __init__(self, parameters=ThicknessEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ residual of 2D thickness evolution PDE

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # get the ids
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]
        tid = self.local_input_var["t"]

        uid = self.local_output_var["u"]
        vid = self.local_output_var["v"]
        aid = self.local_output_var["a"]
        Hid = self.local_output_var["H"]

        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        a = slice_column(nn_output_var, aid)
        H = slice_column(nn_output_var, Hid)

        # spatial derivatives
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        H_x = jacobian(nn_output_var, nn_input_var, i=Hid, j=xid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        H_y = jacobian(nn_output_var, nn_input_var, i=Hid, j=yid)

        # temporal derivative
        H_t = jacobian(nn_output_var, nn_input_var, i=Hid, j=tid)

        # residual
        f = H_t + H*u_x + H_x*u + H*v_y + H_y*v - a

        return [f] #}}}
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of MC 2D PDE, jax version

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}
