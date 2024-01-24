from abc import ABC, abstractmethod

class EquationBase(ABC):
    """ base class of all the equations
    """
    def __init__(self):
        # Physical constants in [SI]
        self.rhoi   = 917.0             # ice density (kg/m^3)
        self.rhow   = 1023.0            # sea water density (kg/m^3)
        self.g      = 9.81              # gravitational force (m/s^2)
        self.yts    = 3600.0*24*365     # year to second (s)

        # Dict of dependent and independent variables of the model, the values are
        # the global component id in the Physics, these two dicts are maps from local 
        # to global
        self.local_input_var = {}       # x, y, z, t, etc.
        self.local_output_var = {}      # u, v, s, H, etc.

        # default lower and upper bounds of the output in [SI] unit
        self.output_lb = {}
        self.output_ub = {}

        # default weights to scale the data misfit
        self.data_weights = {}

        # residual name list
        self.residuals = []

        # default pde weights
        self.pde_weights = []

    def get_input_list(self):
        """ get the List of names of input variables
        """
        return list(self.local_input_var.keys())

    def get_output_list(self):
        """ get the List of names of output variables
        """
        return list(self.local_output_var.keys())

    def update_id(self, global_input_var=None, global_output_var=None):
        """ update component id, always remeber to call this in compiling the model

        Args:
            global_input_var: List of input_variables to nn, these variables are 
                shared across all the physics
            global_output_var: List of output_variables from nn, these variables 
                are shared across all the physics
        """
        if global_input_var is not None:
            self.local_input_var = {o:global_input_var.index(o) for o in self.local_input_var}
        if global_output_var is not None:
            self.local_output_var = {o:global_output_var.index(o) for o in self.local_output_var}
        
    @abstractmethod
    def pde(self, nn_input_var, nn_output_var):
        """ pde function used in deepxde
        """
        return
