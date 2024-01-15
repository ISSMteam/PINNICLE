import deepxde as dde

from . import physics
from . import domain
from .parameters import Parameters
from .modeldata import Data
from .nn import NN


class PINN:
    def __init__(self, params={}, data=Data()):
        # initialize data
        self.param = Parameters(params)

        # main components
        self.domain = domain.Domain(self.param.domain.shapefile)
        self.param.nn.set_parameters({"input_lb": self.domain.geometry.bbox[0,:], "input_ub": self.domain.geometry.bbox[1,:]})

        # training data
        self.training_data = [dde.icbc.PointSetBC(data.X[d], data.sol[d], component=i) for i,d in enumerate(self.param.physics.variables)]

        # set pde
        self.pde = physics.SSA2DUniformMu(params["mu"])

        #  deepxde data object
        data = dde.data.PDE(
                self.domain.geometry,
                self.pde,
                self.training_data,  # all the data loss will be evaluated
                num_domain=self.param.domain.num_collocation_points, # collocation points
                num_boundary=0,  # no need to set for data misfit, unless add calving front boundary, etc.
                num_test=None)

        # define the neural network in use
        self.nn = NN(self.param.nn)

        # weights

        # setup the deepxde model


#    def train(self):

