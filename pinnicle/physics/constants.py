class Constants():
    """ Base class of all physical constants, in [SI]
    """
    def __init__(self, **kwargs):
        # Physical constants in [SI]
        self.rhoi   = 917.0             # ice density (kg/m^3)
        self.rhow   = 1023.0            # sea water density (kg/m^3)
        self.g      = 9.81              # gravitational force (m/s^2)
        self.yts    = 3600.0*24*365     # year to second (s)
        self.eps    = 1.0e-15           # constant epsilon

        # Typical range of variables
        self.variable_lb = {'u': -1.0e4/self.yts, 'v':-1.0e4/self.yts, 's':-1.0e3, 'H':10.0,   'C':0.01,  'a': -5.0/self.yts, 'B': 7.0e7}
        self.variable_ub = {'u': 1.0e4/self.yts,  'v':1.0e4/self.yts,  's':3.6e3,  'H':3500.0, 'C':1.0e4, 'a':  5.0/self.yts, 'B': 7.0e8}
        self.variable_lb['taub'] = 0.0
        self.variable_ub['taub'] = 1.0e6
        self.variable_lb['B11'] = -1.0e10
        self.variable_ub['B11'] = 1.0e10
        self.variable_lb['B12'] = -1.0e10
        self.variable_ub['B12'] = 1.0e10
        self.variable_lb['B22'] = -1.0e10
        self.variable_ub['B22'] = 1.0e10
        # add more if needed
        self.variable_lb['u_base'] = -1.0e4/self.yts
        self.variable_ub['u_base'] =  1.0e4/self.yts
        self.variable_lb['v_base'] = -1.0e4/self.yts
        self.variable_ub['v_base'] =  1.0e4/self.yts
