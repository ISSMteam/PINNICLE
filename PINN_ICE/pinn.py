import deepxde as dde
import numpy as np
import json
import os

from .utils import save_dict_to_json, load_dict_from_json, History, plot_solutions
from .nn import FNN
from .physics import Physics
from .domain import Domain
from .parameter import Parameters
from .modeldata import Data


class PINN:
    """ a basic PINN model
    """
    def __init__(self, params={}, training_data=Data()):
        # Step 1: load setup parameters
        self.param = Parameters(params)

        # Step 2: set physics, all the rest steps depend on what pdes are included in the model
        self.physics = Physics(self.param.physics)
        # assign default physic.input_var, output_var, outout_lb, and output_ub to nn
        self._update_nn_parameters()

        # Step 3: set up deepxde training data object: pde+data
        # domain of the model
        self.domain = Domain(self.param.domain)
        # update training data
        self.training_data = self.update_training_data(training_data)
        #  deepxde data object
        self.data = dde.data.PDE(
                self.domain.geometry,
                self.physics.pdes,
                self.training_data,  # all the data loss will be evaluated
                num_domain=self.param.domain.num_collocation_points, # collocation points
                num_boundary=0,  # no need to set for data misfit, unless add calving front boundary, etc.
                num_test=None)

        # the names of the loss: the order of data follows 'output_variables'
        self.loss_names = self.physics.residuals + [d for d in self.physics.output_var if d in training_data.sol]
        # update the weights for training in the same order
        self.param.training.loss_weights = self.physics.pde_weights + [self.physics.data_weights[i] for i,d in enumerate(self.physics.output_var) if d in training_data.sol]

        # Step 4: set up neural networks
        # automate the input scaling according to the domain, this step need to be done before setting up NN
        self._update_ub_lb_in_nn(training_data)
        # define the neural network in use
        self.nn = FNN(self.param.nn)

        # Step 5: setup the deepxde PINN model
        self.model = dde.Model(self.data, self.nn.net)

    def check_path(self, path, loadOnly=False):
        """check the path, set to default, and create folder if needed
        """
        if path == "":
            path = self.param.training.save_path
        # recursively create paths
        if not loadOnly:
            os.makedirs(path, exist_ok=True)
        return path

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

    def load_model(self, path="", epochs=-1, subfolder="pinn", name="model"):
        """laod the neural network from saved model
        """
        if epochs == -1:
            epochs = self.param.training.epochs

        path = self.check_path(path, loadOnly=True)
        self.model.restore(f"{path}/{subfolder}/{name}-{epochs}.ckpt")

    def load_setting(self, path="", filename="params.json"):
        """ load the settings from file
        """
        path = self.check_path(path, loadOnly=True)
        return load_dict_from_json(path, filename)

    def plot_history(self, path=""):
        """ plot training history
        """
        path = self.check_path(path)
        self.history.plot(path)

    def plot_predictions(self, path="", **kwargs):
        """ plot model predictions
        Args:
            path (Path, str): Path to save the figures
            X_ref (dict): Coordinates of the reference solutions, if None, then just plot the predicted solutions
            u_ref (dict): Reference solutions, if None, then just plot the predicted solutions
            cols (int): Number of columns of subplot
        """
        path = self.check_path(path)
        plot_solutions(self, path=path, **kwargs)

    def save_history(self, path=""):
        """ save training history
        """
        path = self.check_path(path)
        self.history.save(path)

    def save_model(self, path="", subfolder="pinn", name="model"):
        """save the neural network to the hard disk
        """
        path = self.check_path(path)
        self.model.save(f"{path}/{subfolder}/{name}")

    def save_setting(self, path=""):
        """ save settings from self.param.param_dict
        """
        path = self.check_path(path)
        save_dict_to_json(self.param.param_dict, path, "params.json")
    
    def train(self, iterations=0):
        """ train the model
        """
        if iterations == 0:
            iterations = self.param.training.epochs
        # save settings before training
        if self.param.training.is_save:
            self.save_setting()

        # start training
        self._loss_history, self._train_state = self.model.train(iterations=iterations,
                display_every=10000, disregard_previous_best=True)
        
        # prepare history
        self.history = History(self._loss_history, self.loss_names)

        # save history and model variables after training
        if self.param.training.is_save: 
            self.save_history()
            self.save_model()

        # plot history and best results
        if self.param.training.is_plot: 
            self.plot_history()

    def update_training_data(self, training_data):
        """ update data set used for the training, the order follows 'output_variables'
        """
        return [dde.icbc.PointSetBC(training_data.X[d], training_data.sol[d], component=i) 
                for i,d in enumerate(self.param.nn.output_variables) if d in training_data.sol]

    def _update_nn_parameters(self):
        """ assign physic.input_var, output_var, output_lb, and output_ub to nn
        """
        self.param.nn.set_parameters({"input_variables": self.physics.input_var, 
            "output_variables": self.physics.output_var, 
            "output_lb": self.physics.output_lb,
            "output_ub": self.physics.output_ub})

    def _update_ub_lb_in_nn(self, training_data):
        """ update input/output scalings parameters for nn
        """
        # automate the input scaling according to the domain
        self.param.nn.set_parameters({"input_lb": self.domain.geometry.bbox[0,:], "input_ub": self.domain.geometry.bbox[1,:]})

        # check if training data exceed the scaling range
        for i,d in enumerate(self.param.nn.output_variables):
            if d in training_data.sol:
                if np.max(training_data.sol[d]) > self.param.nn.output_ub[i]:
                    self.param.nn.output_ub[i] = np.max(training_data.sol[d])
                if np.min(training_data.sol[d]) < self.param.nn.output_lb[i]:
                    self.param.nn.output_lb[i] = np.min(training_data.sol[d])

        # wrap output_lb and output_ub with np.array
        self.param.nn.output_lb = np.array(self.param.nn.output_lb)
        self.param.nn.output_ub = np.array(self.param.nn.output_ub)
