import deepxde as dde
import deepxde.backend as bkd

def default_float_type():
    """
    Return the default float type according to the backend used
    """
    return bkd.data_type_dict[dde.config.default_float()]
