import deepxde as dde
from deepxde.backend import jax, abs
from . import EquationBase, Constants
from ..parameter import EquationParameter
from ..utils import slice_column, jacobian, slice_function_jax

# mass-conserving SSA with taub
class SSAMCTauEquationParameter(EquationParameter, Constants):
    """ default parameters for SSA Taub
    """
    _EQUATION_TYPE = 'SSA_MC Taub' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['D', 'R', 's', 'H', 'beta']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0, 1.0, 1.0e-6, 1.0e-6, 1.0e-10]
        self.residuals = ["f"+self._EQUATION_TYPE+"1", "f"+self._EQUATION_TYPE+"2"]
        self.pde_weights = [1.0e-10, 1.0e-10]

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                'B':1.26802073401e+08   # -8 degree C, cuffey
                }
class SSA_MC_Taub(EquationBase): #{{{
    """ SSA on 2D problem with uniform B, no friction law, but use taub=-beta*u
    """
    _EQUATION_TYPE = 'SSA_MC Taub' 
    def __init__(self, parameters=SSAMCTauEquationParameter()):
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

        Did = self.local_output_var["D"]
        Rid = self.local_output_var["R"]
        sid = self.local_output_var["s"]
        Hid = self.local_output_var["H"]
        betaid = self.local_output_var["beta"]

        # unpacking normalized output
        H = slice_column(nn_output_var, Hid)
        beta = slice_column(nn_output_var, betaid)
        
        # recovering u,v,a
        D_x = jacobian(nn_output_var, nn_input_var, i=Did, j=xid)
        D_y = jacobian(nn_output_var, nn_input_var, i=Did, j=yid)
        R_x = jacobian(nn_output_var, nn_input_var, i=Rid, j=xid)
        R_y = jacobian(nn_output_var, nn_input_var, i=Rid, j=yid)

        # a = D_x + D_y ## == div(Hv)
        u = (D_x - R_y) / H
        v = (D_y + R_x) / H

        # spatial derivatives
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        eta = 0.5*self.B *(u_x**2.0 + v_y**2.0 + 0.25*(u_y+v_x)**2.0 + u_x*v_y+self.eps)**(0.5*(1.0-self.n)/self.n)
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
        # u_norm = (u**2+v**2+self.eps**2)**0.5

        f1 = sigma11 + sigma12 - abs(beta)*u - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - abs(beta)*v - self.rhoi*self.g*H*s_y

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

        Did = self.local_output_var["D"]
        Rid = self.local_output_var["R"]
        sid = self.local_output_var["s"]
        Hid = self.local_output_var["H"]
        betaid = self.local_output_var["beta"]

        # unpacking normalized output
        H = slice_column(nn_output_var, Hid)
        beta = slice_column(nn_output_var, betaid)
        
        # recovering u,v,a
        D_x = jacobian(nn_output_var, nn_input_var, i=Did, j=xid)
        D_y = jacobian(nn_output_var, nn_input_var, i=Did, j=yid)
        R_x = jacobian(nn_output_var, nn_input_var, i=Rid, j=xid)
        R_y = jacobian(nn_output_var, nn_input_var, i=Rid, j=yid)

        a = D_x + D_y ## == div(Hv)
        u = (D_x - R_y) / H
        v = (D_y + R_x) / H

        # get the spatial derivatives functions
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid, val=1)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid, val=1)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid, val=1)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid, val=1)

        # get variable function
        H_func = lambda x: slice_function_jax(nn_output_var, x, Hid)
        # stress tensor
        etaH = lambda x: 0.5*H_func(x)*self.B *(u_x(x)**2.0 + v_y(x)**2.0 + 0.25*(u_y(x)+v_x(x))**2.0 + u_x(x)*v_y(x)+self.eps)**(0.5*(1.0-self.n)/self.n)

        B11 = lambda x: etaH(x)*(4*u_x(x) + 2*v_y(x))
        B22 = lambda x: etaH(x)*(4*v_y(x) + 2*u_x(x))
        B12 = lambda x: etaH(x)*(  u_y(x) +   v_x(x))

        # Getting the other derivatives
        sigma11 = jacobian((jax.vmap(B11)(nn_input_var), B11), nn_input_var, i=0, j=xid)
        sigma12 = jacobian((jax.vmap(B12)(nn_input_var), B12), nn_input_var, i=0, j=yid)

        sigma21 = jacobian((jax.vmap(B12)(nn_input_var), B12), nn_input_var, i=0, j=xid)
        sigma22 = jacobian((jax.vmap(B22)(nn_input_var), B22), nn_input_var, i=0, j=yid)

        # compute the basal stress
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        # u_norm = (u**2+v**2+self.eps**2)**0.5

        f1 = sigma11 + sigma12 - abs(beta)*u - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - abs(beta)*v - self.rhoi*self.g*H*s_y

        return [f1, f2] #}}}
    #}}}
#}}}


# mass-conserving weak-form SSA with taub
"""
UNDER CONSTRUCTION
DO NOT USE
"""
class SSAweakMCTauEquationParameter(EquationParameter, Constants):
    """ default parameters for SSA Taub
    """
    _EQUATION_TYPE = 'SSAweak_MC Taub' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['D', 'R', 's', 'H', 'beta']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-6, 1.0e-6, 1.0e-10]
        self.residuals = ["f"+self._EQUATION_TYPE+"1", "f"+self._EQUATION_TYPE+"2"]
        self.pde_weights = [1.0e-10, 1.0e-10]

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                'B':1.26802073401e+08   # -8 degree C, cuffey
                }
class SSAweak_MC_Taub(EquationBase): #{{{
    """ SSA on 2D problem with uniform B, no friction law, but use taub=-beta*u
    """
    _EQUATION_TYPE = 'SSAweak_MC Taub' 
    def __init__(self, parameters=SSAweakMCTauEquationParameter()):
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

        Did = self.local_output_var["D"]
        Rid = self.local_output_var["R"]
        sid = self.local_output_var["s"]
        Hid = self.local_output_var["H"]
        betaid = self.local_output_var["beta"]

        # unpacking normalized output
        H = slice_column(nn_output_var, Hid)
        beta = slice_column(nn_output_var, betaid)
        
        # recovering u,v,a
        D_x = jacobian(nn_output_var, nn_input_var, i=Did, j=xid)
        D_y = jacobian(nn_output_var, nn_input_var, i=Did, j=yid)
        R_x = jacobian(nn_output_var, nn_input_var, i=Rid, j=xid)
        R_y = jacobian(nn_output_var, nn_input_var, i=Rid, j=yid)

        a = D_x + D_y ## == div(Hv)
        u = (D_x - R_y) / H
        v = (D_y + R_x) / H

        # spatial derivatives
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        eta = 0.5*self.B *(u_x**2.0 + v_y**2.0 + 0.25*(u_y+v_x)**2.0 + u_x*v_y+self.eps)**(0.5*(1.0-self.n)/self.n)
        # stress tensor
        etaH = eta * H
        B11 = etaH*(4*u_x + 2*v_y)
        B22 = etaH*(4*v_y + 2*u_x)
        B12 = etaH*(  u_y +   v_x)

        # trace of strain rate tensor
        tr_sr = B11 + B22
        # Frobenius inner product of strain rate tensor
        Frob_sr = B11**2. + 2.*B12**2. + B22**2.
        # Frob_sr = torch.square(B11) + torch.square(B12) + torch.square(B21) + torch.square(B22)
        # effective strain rate
        # eff_sr = torch.sqrt( 0.5 * (Frob_sr + tr_sr) )
        eff_sr = ( 0.5 * (Frob_sr + tr_sr) ) ** 2.

        # _ = torch.mean(Frob_sr) # If this line is removed, Frob_sr evaluates to NaN. No clue why...

        # dot product of surface gradient and velocity
        ds_dot_u = (s_x*u) + (s_y*v)

        grav = self.rhoi*self.g*H
        n1 = 1/(self.n + 1)
        n2 = (2 * self.n)/(self.n + 1)

        # compute the integral components
        E_visc = n2 * H * self.B * eff_sr**n1
        E_fric = 0.5 * beta * (u**2 + v**2)
        E_grav = grav * ds_dot_u

        # total energy 
        E = E_visc + E_fric + E_grav

        return E #}}}
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of SSA 2D PDEs

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # get the ids
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]

        Did = self.local_output_var["D"]
        Rid = self.local_output_var["R"]
        sid = self.local_output_var["s"]
        Hid = self.local_output_var["H"]
        betaid = self.local_output_var["beta"]

        # unpacking normalized output
        H = slice_column(nn_output_var, Hid)
        beta = slice_column(nn_output_var, betaid)
        
        # recovering u,v,a
        D_x = jacobian(nn_output_var, nn_input_var, i=Did, j=xid)
        D_y = jacobian(nn_output_var, nn_input_var, i=Did, j=yid)
        R_x = jacobian(nn_output_var, nn_input_var, i=Rid, j=xid)
        R_y = jacobian(nn_output_var, nn_input_var, i=Rid, j=yid)

        a = D_x + D_y ## == div(Hv)
        u = (D_x - R_y) / H
        v = (D_y + R_x) / H

        # get the spatial derivatives functions
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid, val=1)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid, val=1)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid, val=1)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid, val=1)

        # get variable function
        H_func = lambda x: slice_function_jax(nn_output_var, x, Hid)
        # stress tensor
        etaH = lambda x: 0.5*H_func(x)*self.B *(u_x(x)**2.0 + v_y(x)**2.0 + 0.25*(u_y(x)+v_x(x))**2.0 + u_x(x)*v_y(x)+self.eps)**(0.5*(1.0-self.n)/self.n)

        B11 = lambda x: etaH(x)*(4*u_x(x) + 2*v_y(x))
        B22 = lambda x: etaH(x)*(4*v_y(x) + 2*u_x(x))
        B12 = lambda x: etaH(x)*(  u_y(x) +   v_x(x))

        # Getting the other derivatives
        sigma11 = jacobian((jax.vmap(B11)(nn_input_var), B11), nn_input_var, i=0, j=xid)
        sigma12 = jacobian((jax.vmap(B12)(nn_input_var), B12), nn_input_var, i=0, j=yid)

        sigma21 = jacobian((jax.vmap(B12)(nn_input_var), B12), nn_input_var, i=0, j=xid)
        sigma22 = jacobian((jax.vmap(B22)(nn_input_var), B22), nn_input_var, i=0, j=yid)

        # compute the basal stress
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        u_norm = (u**2+v**2+self.eps**2)**0.5

        f1 = sigma11 + sigma12 - abs(beta)*u - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - abs(beta)*v - self.rhoi*self.g*H*s_y

        return [f1, f2] #}}}
    #}}}
#}}}