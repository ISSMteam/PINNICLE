from .parameter import *
from . import utils
from . import physics
from . import modeldata
from . import nn
from . import domain

from .pinn import PINN

import deepxde as dde
dde.config.set_default_float('float64')
