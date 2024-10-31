import deepxde as dde
from deepxde.backend import jax
from . import EquationBase, Constants
from ..parameter import EquationParameter
from ..utils import slice_column, jacobian, slice_function_jax

# SSA constant B {{{
class SSAEquationParameter(EquationParameter, Constants):
    """ default parameters for SSA
    """
    _EQUATION_TYPE = 'SSA' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u', 'v', 's', 'H', 'C']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-6, 1.0e-6, 1.0e-8]
        self.residuals = ["f"+self._EQUATION_TYPE+"1", "f"+self._EQUATION_TYPE+"2"]
        self.pde_weights = [1.0e-10, 1.0e-10]

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                'B':1.26802073401e+08   # -8 degree C, cuffey
                }
class SSA(EquationBase): #{{{
    """ SSA on 2D problem with uniform B
    """
    _EQUATION_TYPE = 'SSA' 
    def __init__(self, parameters=SSAEquationParameter()):
        super().__init__(parameters)
    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ residual of SSA 2D PDEs

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
        Cid = self.local_output_var["C"]

        # spatial derivatives
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        H = slice_column(nn_output_var, Hid)
        C = slice_column(nn_output_var, Cid)

        eta = 0.5*self.B *(u_x**2.0 + v_y**2.0 + 0.25*(u_y+v_x)**2.0 + u_x*v_y+1.0e-15)**(0.5*(1.0-self.n)/self.n)
        # stress tensor
        etaH = eta * H
        B11 = etaH*(4*u_x + 2*v_y)
        B22 = etaH*(4*v_y + 2*u_x)
        B12 = etaH*(  u_y +   v_x)

        # Getting the other derivatives
        sigma11 = jacobian(B11, nn_input_var, i=0, j=xid)
        sigma12 = jacobian(B12, nn_input_var, i=0, j=yid)

        sigma21 = jacobian(B12, nn_input_var, i=0, j=xid)
        sigma22 = jacobian(B22, nn_input_var, i=0, j=yid)


        # compute the basal stress
        u_norm = (u**2+v**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)

        f1 = sigma11 + sigma12 - alpha*u/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - alpha*v/(u_norm+1e-30) - self.rhoi*self.g*H*s_y

        return [f1, f2] #}}}
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of SSA 2D PDEs

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
        Cid = self.local_output_var["C"]

        # get the spatial derivatives functions
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid, val=1)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid, val=1)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid, val=1)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid, val=1)

        # get variable function
        H_func = lambda x: slice_function_jax(nn_output_var, x, Hid)
        # stress tensor
        etaH = lambda x: 0.5*H_func(x)*self.B *(u_x(x)**2.0 + v_y(x)**2.0 + 0.25*(u_y(x)+v_x(x))**2.0 + u_x(x)*v_y(x)+1.0e-15)**(0.5*(1.0-self.n)/self.n)

        B11 = lambda x: etaH(x)*(4*u_x(x) + 2*v_y(x))
        B22 = lambda x: etaH(x)*(4*v_y(x) + 2*u_x(x))
        B12 = lambda x: etaH(x)*(  u_y(x) +   v_x(x))

        # Getting the other derivatives
        sigma11 = jacobian((jax.vmap(B11)(nn_input_var), B11), nn_input_var, i=0, j=xid)
        sigma12 = jacobian((jax.vmap(B12)(nn_input_var), B12), nn_input_var, i=0, j=yid)

        sigma21 = jacobian((jax.vmap(B12)(nn_input_var), B12), nn_input_var, i=0, j=xid)
        sigma22 = jacobian((jax.vmap(B22)(nn_input_var), B22), nn_input_var, i=0, j=yid)

        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        H = slice_column(nn_output_var, Hid)
        C = slice_column(nn_output_var, Cid)
        # compute the basal stress
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        u_norm = (u**2+v**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)

        f1 = sigma11 + sigma12 - alpha*u/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - alpha*v/(u_norm+1e-30) - self.rhoi*self.g*H*s_y

        return [f1, f2] #}}}
    #}}}
#}}}
# SSA variable B {{{
class SSAVariableBEquationParameter(EquationParameter, Constants):
    """ default parameters for SSA, with spatially varying rheology B
    """
    _EQUATION_TYPE = 'SSA_VB' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u', 'v', 's', 'H', 'C', 'B']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-6, 1.0e-6, 1.0e-8, 1e-16]
        self.residuals = ["f"+self._EQUATION_TYPE+"1", "f"+self._EQUATION_TYPE+"2"]
        self.pde_weights = [1.0e-10, 1.0e-10]

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                }
class SSAVariableB(EquationBase): # {{{
    """ SSA on 2D problem with spatially varying B
    """
    _EQUATION_TYPE = 'SSA_VB' 
    def __init__(self, parameters=SSAVariableBEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ residual of SSA 2D PDEs

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
        Cid = self.local_output_var["C"]
        Bid = self.local_output_var["B"]

        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        H = slice_column(nn_output_var, Hid)
        C = slice_column(nn_output_var, Cid)
        B = slice_column(nn_output_var, Bid)

        # spatial derivatives
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid)
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        eta = 0.5*B *(u_x**2.0 + v_y**2.0 + 0.25*(u_y+v_x)**2.0 + u_x*v_y+1.0e-15)**(0.5*(1.0-self.n)/self.n)
        # stress tensor
        etaH = eta * H
        B11 = etaH*(4*u_x + 2*v_y)
        B22 = etaH*(4*v_y + 2*u_x)
        B12 = etaH*(  u_y +   v_x)

        # Getting the other derivatives
        sigma11 = jacobian(B11, nn_input_var, i=0, j=xid)
        sigma12 = jacobian(B12, nn_input_var, i=0, j=yid)

        sigma21 = jacobian(B12, nn_input_var, i=0, j=xid)
        sigma22 = jacobian(B22, nn_input_var, i=0, j=yid)

        # compute the basal stress
        u_norm = (u**2+v**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)

        f1 = sigma11 + sigma12 - alpha*u/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - alpha*v/(u_norm+1e-30) - self.rhoi*self.g*H*s_y

        return [f1, f2] # }}} 
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of SSA 2D PDEs

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
        Cid = self.local_output_var["C"]
        Bid = self.local_output_var["B"]

        # get the spatial derivatives functions
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid, val=1)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid, val=1)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid, val=1)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid, val=1)

        # get variable function
        H_func = lambda x: slice_function_jax(nn_output_var, x, Hid)
        B_func = lambda x: slice_function_jax(nn_output_var, x, Bid)
        # stress tensor
        etaH = lambda x: 0.5*H_func(x)*B_func(x)*(u_x(x)**2.0 + v_y(x)**2.0 + 0.25*(u_y(x)+v_x(x))**2.0 + u_x(x)*v_y(x)+1.0e-15)**(0.5*(1.0-self.n)/self.n)

        B11 = lambda x: etaH(x)*(4*u_x(x) + 2*v_y(x))
        B22 = lambda x: etaH(x)*(4*v_y(x) + 2*u_x(x))
        B12 = lambda x: etaH(x)*(  u_y(x) +   v_x(x))

        # Getting the other derivatives
        sigma11 = jacobian((jax.vmap(B11)(nn_input_var), B11), nn_input_var, i=0, j=xid)
        sigma12 = jacobian((jax.vmap(B12)(nn_input_var), B12), nn_input_var, i=0, j=yid)

        sigma21 = jacobian((jax.vmap(B12)(nn_input_var), B12), nn_input_var, i=0, j=xid)
        sigma22 = jacobian((jax.vmap(B22)(nn_input_var), B22), nn_input_var, i=0, j=yid)

        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        H = slice_column(nn_output_var, Hid)
        C = slice_column(nn_output_var, Cid)
        # compute the basal stress
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        u_norm = (u**2+v**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)

        f1 = sigma11 + sigma12 - alpha*u/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - alpha*v/(u_norm+1e-30) - self.rhoi*self.g*H*s_y

        return [f1, f2] #}}}
#}}}
#}}}
# MOLHO constant B{{{
class MOLHOEquationParameter(EquationParameter, Constants):
    """ default parameters for MOLHO
    """
    _EQUATION_TYPE = 'MOLHO' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u', 'v', 'u_base', 'v_base', 's', 'H', 'C']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-6, 1.0e-6, 1.0e-8]
        self.residuals = ["f"+self._EQUATION_TYPE+" 1", "f"+self._EQUATION_TYPE+" 2", "f"+self._EQUATION_TYPE+" base 1", "f"+self._EQUATION_TYPE+" base 2"]
        self.pde_weights = [1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10]

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                'B':1.26802073401e+08   # -8 degree C, cuffey
                }

class MOLHO(EquationBase): #{{{
    """ MOLHO on 2D problem with uniform B
    """
    _EQUATION_TYPE = 'MOLHO' 
    def __init__(self, parameters=EquationParameter()):
        super().__init__(parameters)

        # gauss points for integration
        self.constants = {"gauss_x":[0.5, 0.23076534494715845, 0.7692346550528415, 0.04691007703066802, 0.9530899229693319],
                "gauss_weights":[0.5688888888888889,0.4786286704993665,0.4786286704993665,0.2369268850561891,0.2369268850561891]}

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ residual of MOLHO 2D PDEs

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # get the ids
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]

        #
        uid = self.local_output_var["u"]
        vid = self.local_output_var["v"]
        ubid = self.local_output_var["u_base"]
        vbid = self.local_output_var["v_base"]
        sid = self.local_output_var["s"]
        Hid = self.local_output_var["H"]
        Cid = self.local_output_var["C"]

        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        ub = slice_column(nn_output_var, ubid)
        vb = slice_column(nn_output_var, vbid)
        H = slice_column(nn_output_var, Hid)
        C = slice_column(nn_output_var, Cid)

        ushear = u - ub
        vshear = v - vb

        # spatial derivatives
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid)
        ub_x = jacobian(nn_output_var, nn_input_var, i=ubid, j=xid)
        vb_x = jacobian(nn_output_var, nn_input_var, i=vbid, j=xid)
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)

        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        ub_y = jacobian(nn_output_var, nn_input_var, i=ubid, j=yid)
        vb_y = jacobian(nn_output_var, nn_input_var, i=vbid, j=yid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        # compute mus
        mu1 = 0.0
        mu2 = 0.0
        mu3 = 0.0
        mu4 = 0.0

        for i,zeta in enumerate(self.constants["gauss_x"]):
            shear_comp = 1.0 - zeta**(self.n+1.0)
            epsilon_eff2 = (ub_x + (u_x-ub_x)*shear_comp)**2.0 + (vb_y + (v_y-vb_y)*shear_comp)**2.0 + (0.5*(ub_y+vb_x+(u_y-ub_y+v_x-vb_x)*shear_comp))**2.0 \
                    + (0.5*(self.n+1)/H*(ushear)*(1-shear_comp))**2.0 + (0.5*(self.n+1)/H*(vshear)*(1-shear_comp))**2.0 + (ub_x + (u_x-ub_x)*shear_comp)*(vb_y + (v_y-vb_y)*shear_comp)

            mu = 0.5*self.B*(epsilon_eff2 + 1.0e-15)**(0.5*(1.0-self.n)/self.n)
            mu1 += 0.5*H*mu*self.constants["gauss_weights"][i]
            mu2 += 0.5*H*mu*self.constants["gauss_weights"][i]*(shear_comp)
            mu3 += 0.5*H*mu*self.constants["gauss_weights"][i]*(shear_comp**2.0)
            mu4 += 0.5*H*mu*self.constants["gauss_weights"][i]*(((self.n+1.0)/H*zeta**self.n)**2.0)

        # stress tensor
        B11 = mu1*(4.0*ub_x+2.0*vb_y) + mu2*(4.0*(u_x-ub_x)+2.0*(v_y-vb_y))
        B12 = mu1*(ub_y+vb_x) + mu2*(u_y-ub_y+v_x-vb_x)
        # B21 = B12
        B22 = mu1*(2.0*ub_x+4.0*vb_y) + mu2*(2.0*(u_x-ub_x)+4.0*(v_y-vb_y))
        B31 = mu2*(4.0*ub_x+2.0*vb_y) + mu3*(4.0*(u_x-ub_x)+2.0*(v_y-vb_y))
        B32 = mu2*(ub_y+vb_x) + mu3*(u_y-ub_y+v_x-vb_x)
        #B41 = B32
        B42 = mu2*(2.0*ub_x+4.0*vb_y) + mu3*(2.0*(u_x-ub_x)+4.0*(v_y-vb_y))

        # Getting the other derivatives
        sigma11 = jacobian(B11, nn_input_var, i=0, j=xid)
        sigma12 = jacobian(B12, nn_input_var, i=0, j=yid)

        sigma21 = jacobian(B12, nn_input_var, i=0, j=xid)
        sigma22 = jacobian(B22, nn_input_var, i=0, j=yid)

        sigma31 = jacobian(B31, nn_input_var, i=0, j=xid)
        sigma32 = jacobian(B32, nn_input_var, i=0, j=yid)

        sigma41 = jacobian(B32, nn_input_var, i=0, j=xid)
        sigma42 = jacobian(B42, nn_input_var, i=0, j=yid)

        # compute the basal stress
        u_norm = (ub**2+vb**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)

        f1 = sigma11 + sigma12 - alpha*ub/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - alpha*vb/(u_norm+1e-30) - self.rhoi*self.g*H*s_y
        f3 = sigma31 + sigma32 + mu4*ushear - self.rhoi*self.g*H*s_x*(self.n+1.0)/(self.n+2.0)
        f4 = sigma41 + sigma42 + mu4*vshear - self.rhoi*self.g*H*s_y*(self.n+1.0)/(self.n+2.0)

        return [f1, f2, f3, f4] #}}}
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of MOLHO 2D PDEs

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        pass
    #}}}
#}}}
#}}}
