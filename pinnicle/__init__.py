from .parameter import *
from . import utils
from . import physics
from . import modeldata
from . import nn
from . import domain

from .pinn import PINN

import deepxde as dde
dde.config.set_default_float('float64')

if dde.backend.backend_name == "pytorch":
    if dde.backend.torch.mps.is_available():
        dde.backend.torch.set_default_device("cpu")
