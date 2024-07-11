import deepxde as dde
import deepxde.backend as bkd
from deepxde.backend import backend_name, tf, jax

def slice_column_tf(variable, sid, eid):
    """ slice the column sid:eid of variable, tensorflow version
    """
    return variable[:, sid:eid]

def slice_column_jax(variable, sid, eid):
    """ slice the column sid:eid of variable, jax version
        currently jax output returns tuple (variable, function)
    """
    temp = variable[0]
    return temp[:, sid:eid]

def jacobian_jax(output_var, input_var, i, j):
    """ Compute jacobian using deepxde
        This is a hack for now to take the first entry from the tuple returned by jax
    """
    J = dde.grad.jacobian(output_var, input_var, i, j)
    return J[0]

if backend_name == "tensorflow":
    slice_column = slice_column_tf
    jacobian = dde.grad.jacobian
elif backend_name == "jax":
    slice_column = slice_column_jax
    jacobian = jacobian_jax
