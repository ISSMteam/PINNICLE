import deepxde.backend as bkd
import tensorflow as tf

def surface_log_vel_misfit(v_true, v_pred):
    """Compute SurfaceLogVelMisfit:
    This function can only work with tensorflow backend for now, since we use tf.math.log()
                       [        vel + eps     ] 2
       J = 4 \bar{v}^2 | log ( -----------  ) |
                       [       vel   + eps    ]
                                  obs
    """
    epsvel=2.220446049250313e-16
    meanvel=3.170979198376458e-05 # /*1000 m/yr*/
    return bkd.reduce_mean(bkd.square(2.0*meanvel*(tf.math.log((bkd.abs(v_pred)+epsvel)/(bkd.abs(v_true)+epsvel)))))

