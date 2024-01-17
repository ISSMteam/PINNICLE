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

        # list of dependent and independent variables of the model
        self.local_input_var = []       # x, y, z, t, etc.
        self.local_output_var = []      # u, v, s, H, etc.

        # list of global component id of the local input and output
        self.input_id = []
        self.output_id = []

    def update_id(self, global_input_var=None, global_output_var=None):
        """ update component id, always remeber to call this in compiling the model

        Args:
            global_output_variables: List of output_variables from nn, these variables 
                are shared across all the physics
        """
        if global_input_var is not None:
            self.input_id = [global_input_var.index(o) for o in self.local_input_var]
        if global_output_var is not None:
            self.output_id = [global_output_var.index(o) for o in self.local_output_var]
        
    @abstractmethod
    def pde(self, input_var, output_var):
        """ pde function used in deepxde
        """
        return
