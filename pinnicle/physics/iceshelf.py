import deepxde as dde
from . import EquationBase, Constants
from ..parameter import EquationParameter

# ==========================================================
# ==========================================================
# UNDER DEVELOPMENT
# Boundary conditions still need implementation. 
# ==========================================================
# ==========================================================

# SiSA constant B {{{
class SSAShelfEquationParameter(EquationParameter, Constants):
    """ default parameters for SSA on ice shelves
    """
    _EQUATION_TYPE = 'SSA_SHELF' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u', 'v', 's', 'H']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-6, 1.0e-6]
        self.residuals = ["f"+self._EQUATION_TYPE+"1", "f"+self._EQUATION_TYPE+"2"]
        self.pde_weights = [1.0e-10, 1.0e-10]

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                'B':1.26802073401e+08   # -8 degree C, cuffey
                }
class SSAShelf(EquationBase): #{{{
    """ SSA ice shelf, on 2D problem with uniform B
    """
    _EQUATION_TYPE = 'SSA_SHELF' 
    def __init__(self, parameters=SSAShelfEquationParameter()):
        super().__init__(parameters)

    def pde(self, nn_input_var, nn_output_var):
        """ residual of ice shelf SSA 2D PDEs

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # get the ids
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]

        uid = self.local_output_var["u"]
        vid = self.local_output_var["v"]
        sid = self.local_output_var["s"]
        Hid = self.local_output_var["H"]

        # unpacking normalized output
        u, v, H = nn_output_var[:, uid:uid+1], nn_output_var[:, vid:vid+1], nn_output_var[:, Hid:Hid+1]
    
        # spatial derivatives
        u_x = dde.grad.jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        v_x = dde.grad.jacobian(nn_output_var, nn_input_var, i=vid, j=xid)
        s_x = dde.grad.jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        u_y = dde.grad.jacobian(nn_output_var, nn_input_var, i=uid, j=yid)
        v_y = dde.grad.jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        s_y = dde.grad.jacobian(nn_output_var, nn_input_var, i=sid, j=yid)
    
        eta = 0.5*self.B *(u_x**2.0 + v_y**2.0 + 0.25*(u_y+v_x)**2.0 + u_x*v_y+1.0e-15)**(0.5*(1.0-self.n)/self.n)
        # stress tensor
        etaH = eta * H
        B11 = etaH*(4*u_x + 2*v_y)
        B22 = etaH*(4*v_y + 2*u_x)
        B12 = etaH*(  u_y +   v_x)
    
        # Getting the other derivatives
        sigma11 = dde.grad.jacobian(B11, nn_input_var, i=0, j=xid)
        sigma12 = dde.grad.jacobian(B12, nn_input_var, i=0, j=yid)
    
        sigma21 = dde.grad.jacobian(B12, nn_input_var, i=0, j=xid)
        sigma22 = dde.grad.jacobian(B22, nn_input_var, i=0, j=yid)
    
        # compute the basal stress
        #u_norm = (u**2+v**2)**0.5
        #alpha = C**2 * (u_norm)**(1.0/self.n)
    
        f1 = sigma11 + sigma12 - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - self.rhoi*self.g*H*s_y
    
        return [f1, f2] #}}}
#}}}

# SSA variable B {{{
class SSAShelfVariableBEquationParameter(EquationParameter, Constants):
    """ default parameters for SSA, with spatially varying rheology B
    """
    _EQUATION_TYPE = 'SSA_SHELF_VB' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u', 'v', 's', 'H', 'B']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-6, 1.0e-6, 1e-16]
        self.residuals = ["f"+self._EQUATION_TYPE+"1", "f"+self._EQUATION_TYPE+"2"]
        self.pde_weights = [1.0e-10, 1.0e-10]

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                }
class SSAShelfVariableB(EquationBase): 
    """ SSA for ice shelves on 2D problem with spatially varying B
    """
    _EQUATION_TYPE = 'SSA_SHELF_VB' 
    def __init__(self, parameters=SSAShelfVariableBEquationParameter()):
        super().__init__(parameters)

    def pde(self, nn_input_var, nn_output_var):
        """ residual of SSA 2D PDEs

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # no cover: start
        # get the ids
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]

        uid = self.local_output_var["u"]
        vid = self.local_output_var["v"]
        sid = self.local_output_var["s"]
        Hid = self.local_output_var["H"]
        Bid = self.local_output_var["B"]

        # unpacking normalized output
        u, v, H, B = nn_output_var[:, uid:uid+1], nn_output_var[:, vid:vid+1], nn_output_var[:, Hid:Hid+1], nn_output_var[:, Bid:Bid+1]
    
        # spatial derivatives
        u_x = dde.grad.jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        v_x = dde.grad.jacobian(nn_output_var, nn_input_var, i=vid, j=xid)
        s_x = dde.grad.jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        u_y = dde.grad.jacobian(nn_output_var, nn_input_var, i=uid, j=yid)
        v_y = dde.grad.jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        s_y = dde.grad.jacobian(nn_output_var, nn_input_var, i=sid, j=yid)
    
        eta = 0.5*B *(u_x**2.0 + v_y**2.0 + 0.25*(u_y+v_x)**2.0 + u_x*v_y+1.0e-15)**(0.5*(1.0-self.n)/self.n)
        # stress tensor
        etaH = eta * H
        B11 = etaH*(4*u_x + 2*v_y)
        B22 = etaH*(4*v_y + 2*u_x)
        B12 = etaH*(  u_y +   v_x)
    
        # Getting the other derivatives
        sigma11 = dde.grad.jacobian(B11, nn_input_var, i=0, j=xid)
        sigma12 = dde.grad.jacobian(B12, nn_input_var, i=0, j=yid)
    
        sigma21 = dde.grad.jacobian(B12, nn_input_var, i=0, j=xid)
        sigma22 = dde.grad.jacobian(B22, nn_input_var, i=0, j=yid)
    
        # compute the basal stress
        #u_norm = (u**2+v**2)**0.5
        #alpha = C**2 * (u_norm)**(1.0/self.n)
    
        f1 = sigma11 + sigma12 - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - self.rhoi*self.g*H*s_y
    
        return [f1, f2] # }}} 
        # no cover: stop
#}}}
