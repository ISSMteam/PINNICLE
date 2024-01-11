import deepxde as dde
from . import physics

class SSA2D(physics):
    """
    SSA on 2D problem
    """
    def __init__(self, mu, n=3.0):
        super().__init__()
        # viscosity 
        self.mu = mu
        self.n = n

    def pde(self, x, sol):
        # unpacking normalized values
        u, v, s, H, C = sol[:, 0:1], sol[:, 1:2], sol[:, 2:3], sol[:, 3:4], sol[:, 4:5]
    
        # spatial derivatives
        u_x = dde.grad.jacobian(sol, x, i=0, j=0)
        v_x = dde.grad.jacobian(sol, x, i=1, j=0)
        s_x = dde.grad.jacobian(sol, x, i=2, j=0)
        u_y = dde.grad.jacobian(sol, x, i=0, j=1)
        v_y = dde.grad.jacobian(sol, x, i=1, j=1)
        s_y = dde.grad.jacobian(sol, x, i=2, j=1)
    
        eta = 0.5*self.mu *(u_x**2.0 + v_y**2.0 + 0.25*(u_y+v_x)**2.0 + u_x*v_y+1.0e-15)**(0.5*(1.0-self.n)/self.n)
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
    
        f1 = sigma11 + sigma12 - alpha*u/(u_norm+1e-30) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - alpha*v/(u_norm+1e-30) - self.rhoi*self.g*H*s_y
    
        return [f1, f2]
