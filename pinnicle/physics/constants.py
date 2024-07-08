class Constants():
    """ Base class of all physical constants, in [SI]
    """
    def __init__(self, **kwargs):
        # Physical constants in [SI]
        self.rhoi   = 917.0             # ice density (kg/m^3)
        self.rhow   = 1023.0            # sea water density (kg/m^3)
        self.g      = 9.81              # gravitational force (m/s^2)
        self.yts    = 3600.0*24*365     # year to second (s)

        # Typical range of variables
        self.variable_lb = {'u': -1.0e4/self.yts, 'v':-1.0e4/self.yts, 's':-1.0e3, 'H':10.0,   'C':0.01,  'a': -5.0/self.yts, 'B': 7.0e7}
        self.variable_ub = {'u': 1.0e4/self.yts,  'v':1.0e4/self.yts,  's':3.6e3,  'H':3500.0, 'C':1.0e4, 'a':  5.0/self.yts, 'B': 7.0e8}
        # add more if needed
        self.variable_lb['u_base'] = -1.0e4/self.yts
        self.variable_ub['u_base'] =  1.0e4/self.yts
        self.variable_lb['v_base'] = -1.0e4/self.yts
        self.variable_ub['v_base'] =  1.0e4/self.yts
