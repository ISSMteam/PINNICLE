from abc import ABC, abstractmethod

class PhysicsBase(ABC):
    """ base class of all the physics
    """
    def __init__(self):
        # Physical constants in [SI]
        self.rhoi   = 917.0             # ice density (kg/m^3)
        self.rhow   = 1023.0            # sea water density (kg/m^3)
        self.g      = 9.81              # gravitational force (m/s^2)
        self.yts    = 3600.0*24*365     # year to second (s)

        # Dict of dependent and independent variables of the model, the values are
        # the global component id in the NN
        self.input_var = {}       # x, y, z, t, etc.
        self.output_var = {}      # u, v, s, H, etc.

        # residual list
        self.residuals = []

    def get_input_list(self):
        """ get the List of names of input variables
        """
        return list(self.input_var.keys())

    def get_output_list(self):
        """ get the List of names of output variables
        """
        return list(self.output_var.keys())

    def update_id(self, global_input_var=None, global_output_var=None):
        """ update component id, always remeber to call this in compiling the model

        Args:
            global_input_var: List of input_variables to nn, these variables are 
                shared across all the physics
            global_output_var: List of output_variables from nn, these variables 
                are shared across all the physics
        """
        if global_input_var is not None:
            self.input_var = {o:global_input_var.index(o) for o in self.input_var}
        if global_output_var is not None:
            self.output_var = {o:global_output_var.index(o) for o in self.output_var}
        
    @abstractmethod
    def pde(self, nn_input_var, nn_output_var):
        """ pde function used in deepxde
        """
        return
