import deepxde as dde
import deepxde.backend as bkd
from deepxde.backend import backend_name, tf, jax, torch, paddle

# ---- tensorflow {{{
def surface_log_vel_misfit_tf(v_true, v_pred):
    """Compute SurfaceLogVelMisfit: This function is for tensorflow backend
    """
    epsvel=2.220446049250313e-16
    return bkd.reduce_mean(bkd.square((tf.math.log((tf.abs(v_pred)+epsvel)/(tf.abs(v_true)+epsvel)))))

def mean_squared_log_error_tf(y_true, y_pred):
    """ use tensorflow function to compute mean squared log error
    """
    return tf.keras.losses.MeanSquaredLogarithmicError()(y_true, y_pred)
#}}}
# ---- jax {{{
def surface_log_vel_misfit_jax(v_true, v_pred):
    """Compute SurfaceLogVelMisfit: This function is for jax
    """
    epsvel=2.220446049250313e-16
    return bkd.reduce_mean(bkd.square((jax.numpy.log((jax.numpy.abs(v_pred)+epsvel)/(jax.numpy.abs(v_true)+epsvel)))))

def mean_squared_log_error_jax(y_true, y_pred):
    """ use jax/numpy function to compute mean squared log error
    """
    return bkd.reduce_mean(bkd.square(jax.numpy.log(y_true+1.0) - jax.numpy.log(y_pred+1.0)))
#}}}
# ---- pytorch {{{
def surface_log_vel_misfit_pytorch(v_true, v_pred):
    """Compute SurfaceLogVelMisfit: This function is for pytorch backend
    """
    epsvel=2.220446049250313e-16
    return bkd.reduce_mean(bkd.square((torch.log((bkd.abs(v_pred)+epsvel)/(bkd.abs(v_true)+epsvel)))))

def mean_squared_log_error_pytorch(y_true, y_pred):
    """ use log function to compute mean squared log error
    """
    return bkd.reduce_mean(bkd.square(torch.log(y_true+1.0) - torch.log(y_pred+1.0)))
#}}}
# ---- paddle {{{
def surface_log_vel_misfit_paddle(v_true, v_pred):
    """Compute SurfaceLogVelMisfit: This function is for paddle backend
    """
    epsvel=2.220446049250313e-16
    return bkd.reduce_mean(bkd.square((paddle.log((bkd.abs(v_pred)+epsvel)/(bkd.abs(v_true)+epsvel)))))

def mean_squared_log_error_paddle(y_true, y_pred):
    """ use log function to compute mean squared log error
    """
    return bkd.reduce_mean(bkd.square(paddle.log(y_true+1.0) - paddle.log(y_pred+1.0)))
#}}}
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates the Mean Absolute Percentage Error (MAPE).
    """
    # Ensure y_true is not zero to avoid division by zero
    # Add a small epsilon to the denominator to prevent NaN values
    epsvel=2.220446049250313e-16
    return bkd.reduce_mean(bkd.abs((y_true - y_pred) / (y_true + epsvel))) * 100
# ---------------
def loss_dict_tf():
    return {
            "VEL_LOG": surface_log_vel_misfit_tf,
            "MEAN_SQUARE_LOG": mean_squared_log_error_tf,
            "MAPE":mean_absolute_percentage_error,
            }

def loss_dict_jax():
    return {
            "VEL_LOG": surface_log_vel_misfit_jax,
            "MEAN_SQUARE_LOG": mean_squared_log_error_jax,
            "MAPE":mean_absolute_percentage_error,
            }

def loss_dict_pytorch():
    return {
            "VEL_LOG": surface_log_vel_misfit_pytorch,
            "MEAN_SQUARE_LOG": mean_squared_log_error_pytorch,
            "MAPE":mean_absolute_percentage_error,
            }

def loss_dict_paddle():
    return {
            "VEL_LOG": surface_log_vel_misfit_paddle,
            "MEAN_SQUARE_LOG": mean_squared_log_error_paddle,
            "MAPE":mean_absolute_percentage_error,
            }

if backend_name ==  "tensorflow":
    LOSS_DICT = loss_dict_tf()
elif backend_name == "jax":
    LOSS_DICT = loss_dict_jax()
elif backend_name == "pytorch":
    LOSS_DICT = loss_dict_pytorch()
elif backend_name == "paddle":
    LOSS_DICT = loss_dict_paddle()

def get(identifier):
    """Retrieves a loss function.

    Args:
        identifier: A loss identifier. String name of a loss function, or a loss function.

    Returns:
        A loss function.
    """
    if isinstance(identifier, (list, tuple)):
        return list(map(get, identifier))
    if isinstance(identifier, str):
        if identifier in LOSS_DICT:
            return LOSS_DICT[identifier]
        elif identifier in dde.losses.LOSS_DICT:
            return identifier
    if callable(identifier):
        return identifier

    raise ValueError("Could not interpret loss function identifier:", identifier)
