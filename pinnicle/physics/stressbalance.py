import deepxde as dde
from . import EquationBase, Constants
from ..parameter import EquationParameter

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
        self.output_lb = [-1.0e4/self.yts, -1.0e4/self.yts, -1.0e3, 10.0, 0.01]
        self.output_ub = [ 1.0e4/self.yts,  1.0e4/self.yts,  2.5e3, 2000.0, 1.0e4]
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

    def pde(self, nn_input_var, nn_output_var):
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

        # unpacking normalized output
        u, v, H, C = nn_output_var[:, uid:uid+1], nn_output_var[:, vid:vid+1], nn_output_var[:, Hid:Hid+1], nn_output_var[:, Cid:Cid+1]
    
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
        u_norm = (u**2+v**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)
    
        f1 = sigma11 + sigma12 - alpha*u/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - alpha*v/(u_norm+1e-30) - self.rhoi*self.g*H*s_y
    
        return [f1, f2] #}}}

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
        self.output_lb = [-1.0e4/self.yts, -1.0e4/self.yts, -1.0e4/self.yts, -1.0e4/self.yts, -1.0e3, 10.0, 0.01]
        self.output_ub = [ 1.0e4/self.yts,  1.0e4/self.yts,  1.0e4/self.yts,  1.0e4/self.yts,  2.5e3, 2000.0, 1.0e4]
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

    def pde(self, nn_input_var, nn_output_var):
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
        u, v, ub, vb = nn_output_var[:, uid:uid+1], nn_output_var[:, vid:vid+1], nn_output_var[:, ubid:ubid+1], nn_output_var[:, vbid:vbid+1]
        H, C = nn_output_var[:, Hid:Hid+1], nn_output_var[:, Cid:Cid+1]
        ushear = u - ub
        vshear = v - vb
    
        # spatial derivatives
        u_x = dde.grad.jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        v_x = dde.grad.jacobian(nn_output_var, nn_input_var, i=vid, j=xid)
        ub_x = dde.grad.jacobian(nn_output_var, nn_input_var, i=ubid, j=xid)
        vb_x = dde.grad.jacobian(nn_output_var, nn_input_var, i=vbid, j=xid)
        s_x = dde.grad.jacobian(nn_output_var, nn_input_var, i=sid, j=xid)

        u_y = dde.grad.jacobian(nn_output_var, nn_input_var, i=uid, j=yid)
        v_y = dde.grad.jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        ub_y = dde.grad.jacobian(nn_output_var, nn_input_var, i=ubid, j=yid)
        vb_y = dde.grad.jacobian(nn_output_var, nn_input_var, i=vbid, j=yid)
        s_y = dde.grad.jacobian(nn_output_var, nn_input_var, i=sid, j=yid)
    
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
        sigma11 = dde.grad.jacobian(B11, nn_input_var, i=0, j=xid)
        sigma12 = dde.grad.jacobian(B12, nn_input_var, i=0, j=yid)
    
        sigma21 = dde.grad.jacobian(B12, nn_input_var, i=0, j=xid)
        sigma22 = dde.grad.jacobian(B22, nn_input_var, i=0, j=yid)

        sigma31 = dde.grad.jacobian(B31, nn_input_var, i=0, j=xid)
        sigma32 = dde.grad.jacobian(B32, nn_input_var, i=0, j=yid)

        sigma41 = dde.grad.jacobian(B32, nn_input_var, i=0, j=xid)
        sigma42 = dde.grad.jacobian(B42, nn_input_var, i=0, j=yid)
    
        # compute the basal stress
        u_norm = (ub**2+vb**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)
    
        f1 = sigma11 + sigma12 - alpha*ub/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - alpha*vb/(u_norm+1e-30) - self.rhoi*self.g*H*s_y
        f3 = sigma31 + sigma32 + mu4*ushear - self.rhoi*self.g*H*s_x*(self.n+1.0)/(self.n+2.0)
        f4 = sigma41 + sigma42 + mu4*vshear - self.rhoi*self.g*H*s_y*(self.n+1.0)/(self.n+2.0)
    
        return [f1, f2, f3, f4] #}}}

