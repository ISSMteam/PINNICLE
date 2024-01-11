def minmax_scale(x, lb, ub, scale=2.0, offset=1.0):
    return scale*(x - lb)/(ub - lb) - offset

def up_scale(x, lb, ub, scale=2.0, offset=1.0):
    return scale*(x + offset)*(ub - lb)


def input_transform(lb, ub):
    def _wrapper(x):
        return minmax_scale(x, lb, ub)
    return _wrapper

def output_transform(lb, ub):
    def _wrapper(dummy, x):
        return up_scale(x, lb, ub)
    return _wrapper
