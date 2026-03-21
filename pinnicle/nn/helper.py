import deepxde as dde
import deepxde.backend as bkd
from ..utils import matmul

def minmax_scale(x, lb, ub, scale=2.0, offset=1.0):
    """
    min-max scale
    """
    return 1.0/(ub - lb)*scale*(x -lb) - offset

def up_scale(x, lb, ub, scale=0.5, offset=1.0):
    """
    reverse min-max scale
    """
    return lb + scale*(x + offset)*(ub - lb)


def fourier_feature(x, B):
    """
    Apply Fourier Feature Transform
    """
    return bkd.concat([
                      bkd.cos(matmul(x, B)),
                      bkd.sin(matmul(x, B))
                      ], 
                      1)
