import deepxde as dde
import numpy as np
import json
import os

from . import physics
from . import domain
from .parameters import Parameters
from .modeldata import Data
from .nn import FNN
from .utils import save_dict_to_json, load_dict_from_json, History


class PINN:
    """
    class of a PINN model
    """
    def __init__(self, params={}, training_data=Data()):
        # load setup parameters
        self.param = Parameters(params)

        # main components
        self.domain = domain.Domain(self.param.domain.shapefile)
        self.param.nn.set_parameters({"input_lb": self.domain.geometry.bbox[0,:], "input_ub": self.domain.geometry.bbox[1,:]})

        # update training data
        self.training_data = self.update_training_data(training_data)

        # check if training data exceed the scaling range, also wrap output_lb and output_ub with np.array
        for i,d in enumerate(self.param.nn.output_variables):
            if d in training_data.sol:
                if np.max(training_data.sol[d]) > self.param.nn.output_ub[i]:
                    self.param.nn.output_ub[i] = np.max(training_data.sol[d])
                if np.min(training_data.sol[d]) < self.param.nn.output_lb[i]:
                    self.param.nn.output_lb[i] = np.min(training_data.sol[d])
        self.param.nn.output_lb = np.array(self.param.nn.output_lb)
        self.param.nn.output_ub = np.array(self.param.nn.output_ub)

        # set physics
        # TODO: change to add physics
        if "SSA" in self.param.physics.equations:
            self.physics = physics.SSA2DUniformB(params["B"])

        #  deepxde data object
        self.data = dde.data.PDE(
                self.domain.geometry,
                self.physics.pde,
                self.training_data,  # all the data loss will be evaluated
                num_domain=self.param.domain.num_collocation_points, # collocation points
                num_boundary=0,  # no need to set for data misfit, unless add calving front boundary, etc.
                num_test=None)

        # TODO: add more physics
        self.loss_names = self.physics.residuals + list(training_data.sol.keys())

        # define the neural network in use
        self.nn = FNN(self.param.nn)

        # setup the deepxde model: data+pde
        self.model = dde.Model(self.data, self.nn.net)


    def update_training_data(self, training_data):
        """
        update data set used for the training
        """
        return [dde.icbc.PointSetBC(training_data.X[d], training_data.sol[d], component=i) 
                for i,d in enumerate(self.param.nn.output_variables) if d in training_data.sol]

    # TODO: add update data, update net, ...

    def compile(self, opt=None, loss=None, lr=None, loss_weights=None):
        """
        compile the model  
        """
        # load from param
        if opt is None:
            opt = self.param.training.optimizer

        if loss is None:
            loss = self.param.training.loss_function

        if lr is None:
            lr = self.param.training.learning_rate

        if loss_weights is None:
            loss_weights = self.param.training.loss_weights

        # compile the model
        self.model.compile(opt, loss=loss, lr=0.001, loss_weights=loss_weights)

    def train(self, iterations=0):
        """
        train the model
        """
        if iterations == 0:
            iterations = self.param.training.epochs
        # start training
        self._loss_history, self._train_state = self.model.train(iterations=iterations,
                display_every=10000, disregard_previous_best=True)

        # save history and model variables
        if self.param.training.is_save: 
            self.save_history()
            self.save_setting()
            self.save_model()

    def save_setting(self, path=""):
        """ save settings from self.param.param_dict
        """
        path = self.check_path(path)
        save_dict_to_json(self.param.param_dict, path, "params.json")
    
    def load_setting(self, path="", filename="params.json"):
        """ load the settings from file
        """
        path = self.check_path(path, loadOnly=True)
        return load_dict_from_json(path, filename)

    def save_history(self, path=""):
        """ save training history
        """
        path = self.check_path(path)
        self.history = History(self._loss_history, self.loss_names)
        self.history.save(path)
    
    def save_model(self, path="", subfolder="pinn", name="model"):
        """
        save the neural network to the hard disk

        """
        path = self.check_path(path)
        self.model.save(f"{path}/{subfolder}/{name}")

    def load_model(self, path="", epochs=-1, subfolder="pinn", name="model"):
        """
        laod the neural network from saved model
        """
        if epochs == -1:
            epochs = self.param.training.epochs

        path = self.check_path(path, loadOnly=True)
        self.model.restore(f"{path}/{subfolder}/{name}-{epochs}.ckpt")

    def check_path(self, path, loadOnly=False):
        """
        check the path, set to default, and create folder if needed
        """
        if path == "":
            path = self.param.training.save_path
        # recursively create paths
        if not loadOnly:
            os.makedirs(path, exist_ok=True)
        return path
