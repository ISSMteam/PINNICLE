import deepxde as dde
import deepxde.backend as bkd

def minmax_scale(x, lb, ub, scale=2.0, offset=1.0):
    """
    min-max scale
    """
    return scale*(x - lb)/(ub - lb) - offset

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
                      bkd.cos(bkd.matmul(x, B)),
                      bkd.sin(bkd.matmul(x, B))
                      ], 
                      1)

def default_float_type():
    """ 
    Return the default float type according to the backend used
    """
    return bkd.data_type_dict[dde.config.default_float()]
