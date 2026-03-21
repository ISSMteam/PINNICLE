import deepxde as dde
from deepxde.backend import jax
from . import EquationBase, Constants
from ..parameter import EquationParameter
from ..utils import slice_column, jacobian, slice_function_jax

class CalvingFrontBCEquationParameter(EquationParameter, Constants):
    """ default parameters for Boundary Conditions
    """
    _EQUATION_TYPE = 'CalvingFront' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u', 'v', 's', 'H', 'B', 'nx', 'ny']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-6, 1.0e-6, 1e-16, 1e1, 1e1]
        self.residuals = []
        self.pde_weights = []
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                }

class CalvingFrontBC(EquationBase): #{{{
    """ 
    """
    _EQUATION_TYPE = 'CalvingFront' 
    def __init__(self, parameters=CalvingFrontBCEquationParameter()):
        super().__init__(parameters)

    def _bc(self, nn_input_var, nn_output_var): #{{{
        """ compute the residual of calving front boundary condition of SSA 2D

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
            nx, ny: out pointing normal vector
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
        nxid = self.local_output_var["nx"]
        nyid = self.local_output_var["ny"]

        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        H = slice_column(nn_output_var, Hid)
        B = slice_column(nn_output_var, Bid)
        s = slice_column(nn_output_var, sid)
        nx = slice_column(nn_output_var, nxid)
        ny = slice_column(nn_output_var, nyid)
        base = s - H

        # spatial derivatives
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)

        eta = 0.5*B *(u_x**2.0 + v_y**2.0 + 0.25*(u_y+v_x)**2.0 + u_x*v_y+1.0e-15)**(0.5*(1.0-self.n)/self.n)
        # stress tensor
        etaH = eta * H
        B11 = etaH*(4*u_x + 2*v_y)
        B22 = etaH*(4*v_y + 2*u_x)
        B12 = etaH*(  u_y +   v_x)

        # pde residual
        bc1 = B11*nx + B12*ny - 0.5*self.g*(self.rhoi*H*H - self.rhow*base*base)*nx
        bc2 = B12*nx + B22*ny - 0.5*self.g*(self.rhoi*H*H - self.rhow*base*base)*ny
        bc = (bc1**2.0+bc2**2.0)**0.5

        return bc # }}}
    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ Dummy PDE returns nothing, to train the NN with data only

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return []
    def _pde_jax(self, nn_input_var, nn_output_var):
        """ Dummy PDE returns nothing, to train the NN with data only

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return [] #}}}
