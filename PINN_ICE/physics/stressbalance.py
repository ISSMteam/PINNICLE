import deepxde as dde
from .constants import PhysicsBase

class SSA2DUniformB(PhysicsBase):
    """ SSA on 2D problem with uniform B
    """
    def __init__(self, B, n=3.0):
        super().__init__()
        # viscosity 
        self.B = B
        self.n = n

        # Dict of input and output used in this model, and their component id
        self.input_var = {"x":0, "y":1}        
        self.output_var = {"u":0, "v":1, "s":2, "H":3, "C":4}

    def pde(self, nn_input_var, nn_output_var):
        """ residual of SSA 2D PDEs

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # get the ids
        xid = self.input_var["x"]
        yid = self.input_var["y"]

        uid = self.output_var["u"]
        vid = self.output_var["v"]
        sid = self.output_var["s"]
        Hid = self.output_var["H"]
        Cid = self.output_var["C"]

        # unpacking normalized output
        u, v, s, H, C = nn_output_var[:, uid:uid+1], nn_output_var[:, vid:vid+1], nn_output_var[:, sid:sid+1], nn_output_var[:, Hid:Hid+1], nn_output_var[:, Cid:Cid+1]
    
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
        sigma12 = dde.grad.jacobian(B12, nn_input_var, i=0, j=xid)
    
        sigma21 = dde.grad.jacobian(B12, nn_input_var, i=0, j=xid)
        sigma22 = dde.grad.jacobian(B22, nn_input_var, i=0, j=xid)
    
        # compute the basal stress
        u_norm = (u**2+v**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)
    
        f1 = sigma11 + sigma12 - alpha*u/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - alpha*v/(u_norm+1e-30) - self.rhoi*self.g*H*s_y
    
        return [f1, f2]
