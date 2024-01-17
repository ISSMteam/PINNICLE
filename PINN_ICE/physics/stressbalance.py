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

    def pde(self, input_var, output_var):
        """ residual of SSA 2D PDEs

        Args:
            input_var: global input to the nn
            output_var: global output from the nn
        """
        # unpacking normalized output
        u, v, s, H, C = sol[:, 0:1], sol[:, 1:2], sol[:, 2:3], sol[:, 3:4], sol[:, 4:5]
    
        # spatial derivatives
        u_x = dde.grad.jacobian(sol, x, i=0, j=0)
        v_x = dde.grad.jacobian(sol, x, i=1, j=0)
        s_x = dde.grad.jacobian(sol, x, i=2, j=0)
        u_y = dde.grad.jacobian(sol, x, i=0, j=1)
        v_y = dde.grad.jacobian(sol, x, i=1, j=1)
        s_y = dde.grad.jacobian(sol, x, i=2, j=1)
    
        eta = 0.5*self.B *(u_x**2.0 + v_y**2.0 + 0.25*(u_y+v_x)**2.0 + u_x*v_y+1.0e-15)**(0.5*(1.0-self.n)/self.n)
        # stress tensor
        etaH = eta * H
        B11 = etaH*(4*u_x + 2*v_y)
        B22 = etaH*(4*v_y + 2*u_x)
        B12 = etaH*(  u_y +   v_x)
    
        # Getting the other derivatives
        sigma11 = dde.grad.jacobian(B11, x, i=0, j=0)
        sigma12 = dde.grad.jacobian(B12, x, i=0, j=1)
    
        sigma21 = dde.grad.jacobian(B12, x, i=0, j=0)
        sigma22 = dde.grad.jacobian(B22, x, i=0, j=1)
    
        # compute the basal stress
        u_norm = (u**2+v**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)
    
        f1 = sigma11 + sigma12 - alpha*u/(u_norm+1e-30) - self.constants.rhoi*self.constants.g*H*s_x
        f2 = sigma21 + sigma22 - alpha*v/(u_norm+1e-30) - self.constants.rhoi*self.constants.g*H*s_y
    
        return [f1, f2]
