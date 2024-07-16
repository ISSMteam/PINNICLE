import deepxde.backend as bkd

def minmax_scale(x, lb, ub, scale=bkd.as_tensor(2.0), offset=bkd.as_tensor(1.0)):
    """
    min-max scale
    """
    return scale*(x - lb)/(ub - lb) - offset

def up_scale(x, lb, ub, scale=bkd.as_tensor(0.5), offset=bkd.as_tensor(1.0)):
    """
    reverse min-max scale
    """
    return lb + scale*(x + offset)*(ub - lb)
