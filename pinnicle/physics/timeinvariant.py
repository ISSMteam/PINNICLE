from . import EquationBase, Constants
from ..parameter import EquationParameter
from ..utils import slice_column, jacobian

# time invariant constraint {{{
class TimeInvariantConstraintParameter(EquationParameter, Constants):
    """ parameters for the time-invariant constraint
    """
    _EQUATION_TYPE = 'Time_Invariant' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y', 't']
        self.output = ['H', 's', 'C']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-6, 1.0e-6, 1.0e-8]
        self.residuals = ["db/dt", "dC/dt"]
        self.pde_weights = [1.0e6, 1.0e6]
        # scalar variables: name:value
        self.scalar_variables = {}

class TimeInvariantConstraint(EquationBase): #{{{
    """ A temporary solution to add time invariant constraint db/dt=0, dC/dt=0 to the PINN
        TODO: 
            1. define these for every time independent equation, similar as _pde_jax
            2. use tf.cond, and similar autograph for pytorch and jax (when they have these implemented), put everything all in _pde
    """
    _EQUATION_TYPE = 'Time_Invariant' 
    def __init__(self, parameters=TimeInvariantConstraintParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ time invariant constraint

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # get the ids
        tid = self.local_input_var["t"]

        sid = self.local_output_var["s"]
        Hid = self.local_output_var["H"]
        Cid = self.local_output_var["C"]

        # unpacking normalized output
        s = slice_column(nn_output_var, sid)
        H = slice_column(nn_output_var, Hid)
        C = slice_column(nn_output_var, Cid)

        # time derivative
        H_t = jacobian(nn_output_var, nn_input_var, i=Hid, j=tid)
        s_t = jacobian(nn_output_var, nn_input_var, i=sid, j=tid)
        C_t = jacobian(nn_output_var, nn_input_var, i=Cid, j=tid)

        # residual
        fdbdt = s_t - H_t
        fdCdt = C_t

        return [fdbdt, fdCdt] #}}}
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ time invariant constraint, jax version

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}
