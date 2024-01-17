from abc import ABC, abstractmethod

class PhysicsBase(ABC):
    """
    base class of all the physics
    """
    def __init__(self):
        # Physical constants in [SI]
        self.rhoi   = 917.0             # ice density (kg/m^3)
        self.rhow   = 1023.0            # sea water density (kg/m^3)
        self.g      = 9.81              # gravitational force (m/s^2)
        self.yts    = 3600.0*24*365     # year to second (s)

        # list of dependent and independent variables of the model
        self.input_variables = []       # x, y, z, t, etc.
        self.output_variables = []      # u, v, s, H, etc.

    @abstractmethod
    def pde(self, input_var, output_var):
        """
        pde function used in deepxde
        """
        return
