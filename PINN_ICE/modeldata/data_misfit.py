import deepxde.backend as bkd
import tensorflow as tf

def surface_log_vel_misfit(v_true, v_pred):
    """Compute SurfaceLogVelMisfit:
                       [        vel + eps     ] 2
       J = 4 \bar{v}^2 | log ( -----------  ) |
                       [       vel   + eps    ]
                                  obs
    """
    epsvel=2.220446049250313e-16
    meanvel=3.170979198376458e-05 # /*1000 m/yr*/
    return bkd.reduce_mean(bkd.square(2.0*meanvel*(tf.math.log((bkd.abs(y_pred)+epsvel)/(bkd.abs(y_true)+epsvel)))))

