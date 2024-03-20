import deepxde as dde
import deepxde.backend as bkd
from deepxde.backend import tf


def surface_log_vel_misfit(v_true, v_pred):
    """Compute SurfaceLogVelMisfit:
    This function can only work with tensorflow backend for now, since we use tf.math.log()
            [        vel + eps     ] 2
       J =  | log ( -----------  ) |
            [       vel   + eps    ]
                       obs
    """
    epsvel=2.220446049250313e-16
    return bkd.reduce_mean(bkd.square((tf.math.log((tf.abs(v_pred)+epsvel)/(tf.abs(v_true)+epsvel)))))

def mean_squared_log_error(y_true, y_pred):
    """ use tensorflow function to compute mean squared log error
    """
    return tf.keras.losses.MeanSquaredLogarithmicError()(y_true, y_pred)

LOSS_DICT = {
    "VEL_LOG": surface_log_vel_misfit,
    "MEAN_SQUARE_LOG": mean_squared_log_error
    }

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
        else:
            return dde.losses.get(identifier)
    if callable(identifier):
        return identifier

    raise ValueError("Could not interpret loss function identifier:", identifier)
