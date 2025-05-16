import deepxde as dde
import deepxde.backend as bkd
from deepxde.backend import backend_name, jax

def slice_column_tf(variable, i):
    """ slice the column i:i+1 of variable, tensorflow version
    """
    return variable[:, i:i+1]

def slice_column_jax(variable, i):
    """ slice the column i:i+1 of variable, jax version
        currently jax output returns tuple (variable, function)
    """
    temp = variable[0]
    return temp[..., i:i+1]

def slice_function_jax(variable, x, i):
    """ slice the column i:i+1 of function, jax version
        currently jax output returns tuple (variable, function)
    """
    temp = variable[1]
    y = temp(x)
    return y[..., i:i+1]

def jacobian_tf(output_var, input_var, i, j, val=0):
    """ Compute jacobian using deepxde
    """
    J = dde.grad.jacobian(output_var, input_var, i, j)
    return J

def jacobian_jax(output_var, input_var, i, j, val=0):
    """ Compute jacobian using deepxde
        This is a hack for now to take the first entry from the tuple returned by jax
    """
    J = dde.grad.jacobian(output_var, input_var, i, j)
    return J[val]

def matmul_tf(A, B):
    return bkd.matmul(A,B)

def matmul_jax(A, B):
    return jax.numpy.matmul(A,B)

if backend_name in ["tensorflow", "pytorch"]:
    slice_column = slice_column_tf
    jacobian = jacobian_tf
    matmul = matmul_tf
elif backend_name == "jax":
    slice_column = slice_column_jax
    jacobian = jacobian_jax
    matmul = matmul_jax
else:
    raise ValueError(f"{backend_name} is not supported by PINNICLE")
