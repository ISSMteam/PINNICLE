import os
import glob
import deepxde as dde
import numpy as np

from .utils import save_dict_to_json, load_dict_from_json, History, plot_solutions, data_misfit
from .nn import FNN
from .physics import Physics
from .domain import Domain
from .parameter import Parameters
from .modeldata import Data


class PINN:
    """ a basic PINN model
    """
    def __init__(self, params={}, loadFrom=""):
        # load setup parameters
        if os.path.exists(loadFrom):
            # overwrite params with saved params.json file
            params = self.load_setting(path=loadFrom)
        self.params = Parameters(params)

        # set up the model according to self.params
        self.setup()

    def check_path(self, path, loadOnly=False):
        """check the path, set to default, and create folder if needed
        """
        if path == "":
            path = self.params.training.save_path
        # recursively create paths
        if not loadOnly:
            os.makedirs(path, exist_ok=True)
        return path

    def compile(self, opt=None, loss=None, lr=None, loss_weights=None, decay=None):
        """ compile the model  
        """
        # load from params
        if opt is None:
            opt = self.params.training.optimizer

        if loss is None:
            loss = self.params.training.loss_functions

        if lr is None:
            lr = self.params.training.learning_rate

        if (decay is None) and (self.params.training.decay_steps > 0) and (self.params.training.decay_rate>0.0):
            decay = ("inverse time", self.params.training.decay_steps, self.params.training.decay_rate)

        if loss_weights is None:
            loss_weights = self.params.training.loss_weights

        # compile the model
        self.model.compile(opt, loss=loss, lr=lr, decay=decay, loss_weights=loss_weights)

    def load_model(self, path="", epochs=-1, subfolder="pinn", name="model", fileformat=""):
        """laod the neural network from saved model
        """
        if epochs == -1:
            epochs = self.params.training.epochs

        # get the path
        path = self.check_path(path, loadOnly=True)

        # find the model file 
        if fileformat == "":
            filename = glob.glob(f"{path}/{subfolder}/{name}-{epochs}.*")[0]
        else:
            filename = f"{path}/{subfolder}/{name}-{epochs}.{fileformat}"

        # TODO: remove this step
        # need to predict once, otherwise the weights can not be restored to the nn
        self.compile()
        self.model.predict(np.zeros([1, self.params.nn.input_size]))

        # now the weights can be loaded
        self.model.restore(filename)
        self.compile()

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

    def plot_predictions(self, path="", filename="2Dsolution.png", **kwargs):
        """ plot model predictions

        Args:
            path (Path, str): Path to save the figures
            filename (str): name to save the figures, if set to None, then the figure will not be saved
            X_ref (dict): Coordinates of the reference solutions, if None, then just plot the predicted solutions
            u_ref (dict): Reference solutions, if None, then just plot the predicted solutions
            cols (int): Number of columns of subplot
        """
        path = self.check_path(path)
        plot_solutions(self, path=path, filename=filename, **kwargs)

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

    def save_setting(self, path="", subfolder="pinn"):
        """ save settings from self.params.param_dict
        """
        path = self.check_path(path)
        save_dict_to_json(self.params.param_dict, path, "params.json")
        # create path/pinn/ to save the model weights
        self.check_path(f"{path}/{subfolder}/")
    
    def setup(self):
        """ setup the model according to `self.params` from the constructor
        """
        # Step 2: set physics, all the rest steps depend on what pdes are included in the model
        self.physics = Physics(self.params.physics)
        # assign default physic.input_var, output_var, outout_lb, and output_ub to nn
        self._update_nn_parameters()

        # Step 3: load all avaliable data on the given domain and set up for training data
        # domain of the model
        self.domain = Domain(self.params.domain)
        # create an instance of Data
        self.model_data = Data(self.params.data)
        # load from data file
        self.model_data.load_data(self.domain, self.physics)
        # update according to the setup: input_variables defined by the physics
        self.model_data.prepare_training_data(transient=self.params.domain.time_dependent, default_time=self.params.domain.start_time)

        # Step 4: update training data
        self.training_data, self.loss_names, self.params.training.loss_weights, self.params.training.loss_functions = self.update_training_data(self.model_data)

        # Step 5: set up deepxde training data object using PDE + data
        #  deepxde data object
        self.dde_data = dde.data.PDE(
                self.domain.geometry,
                self.physics.pdes,
                self.training_data,  # all the data loss will be evaluated
                num_domain=self.params.domain.num_collocation_points, # collocation points
                num_boundary=0,  # no need to set for data misfit, unless add calving front boundary, etc.
                num_test=None)

        # Step 6: set up neural networks
        # automate the input scaling according to the domain, this step need to be done before setting up NN
        self._update_ub_lb_in_nn(self.model_data)
        # define the neural network in use
        self.nn = FNN(self.params.nn)
        # save B if it is not defined by the user and generated by FFT
        if (self.params.nn.B is None) and (self.params.nn.fft):
            self.params.param_dict.update({"B": dde.backend.to_numpy(self.nn.B).tolist()})

        # Step 7: setup the deepxde PINN model
        self.model = dde.Model(self.dde_data, self.nn.net)

    def train(self, iterations=0):
        """ train the model
        """
        if iterations == 0:
            iterations = self.params.training.epochs
        # save settings before training
        if self.params.training.is_save:
            self.save_setting()

        # get callback function list
        callbacks = self.update_callbacks()

        # start training
        self._loss_history, self._train_state = self.model.train(iterations=iterations,
                display_every=10000, disregard_previous_best=True, callbacks=callbacks)
        
        # prepare history
        self.history = History(self._loss_history, self.loss_names)

        # save history and model variables after training
        if self.params.training.is_save: 
            self.save_history()
            self.save_model()

        # plot history and best results
        if self.params.training.is_plot: 
            self.plot_history()

    def update_callbacks(self, params=None):
        """ update callback functions for the training according to the settings in params
        """
        if params is None:
            params = self.params.training

        # add callbacks
        if params.has_callbacks:
            callbacks = []
            # early stop
            if params.has_EarlyStopping():
                callbacks.append(dde.callbacks.EarlyStopping(min_delta=params.min_delta, patience=params.patience))
            # check points will be saved to
            if params.has_ModelCheckpoint():
                path = self.check_path("")
                subfolder = "pinn"
                name = "model"
                cpPath = f"{path}/{subfolder}/{name}"
                callbacks.append(dde.callbacks.ModelCheckpoint(cpPath, save_better_only=True))
            # resampler of the collocation points
            if params.has_PDEPointResampler():
                callbacks.append(dde.callbacks.PDEPointResampler(period=params.period))
            return callbacks
        else:
            return None

    def update_training_data(self, training_data):
        """ update data set used for the training, the order follows 'output_variables'
        """
        # loop through all the PDEs, find those avaliable in the training data, add to the PointSetBC
        training_temp = [dde.icbc.PointSetBC(training_data.X[d], training_data.sol[d], component=i) 
                  for i,d in enumerate(self.params.nn.output_variables) if d in training_data.sol]

        # the names of the loss: the order of data follows 'output_variables'
        loss_names = self.physics.residuals + [d for d in self.physics.output_var if d in self.model_data.sol]
        # update the weights for training in the same order
        loss_weights = self.physics.pde_weights + [self.physics.data_weights[i] for i,d in enumerate(self.physics.output_var) if d in self.model_data.sol]
    
        # update loss functions to a list, if not
        if not isinstance(self.params.training.loss_functions, list):
            loss_functions = [self.params.training.loss_functions]*len(loss_weights)
        else:
            loss_functions = self.params.training.loss_functions


        # if additional_loss is not empty
        if self.params.training.additional_loss:
            for d in self.params.training.additional_loss:
                if (d in training_data.X):
                    # append to training_temp for those in the physics
                    if d in self.params.nn.output_variables:
                        # get the index in output of nn
                        i = self.params.nn.output_variables.index(d)
                        # training data
                        training_temp.append(dde.icbc.PointSetBC(training_data.X[d], training_data.sol[d], component=i))
                    # if the variable is not part of the output from nn
                    # currently, only implement 'vel'
                    elif d == "vel":
                        training_temp.append(dde.icbc.PointSetOperatorBC(training_data.X[d], training_data.sol[d], self.physics.vel_mag))
                    elif d == "sx":
                        training_temp.append(dde.icbc.PointSetOperatorBC(training_data.X[d], training_data.sol[d], self.physics.user_defined_gradient('s','x')))
                    elif d == "sy":
                        training_temp.append(dde.icbc.PointSetOperatorBC(training_data.X[d], training_data.sol[d], self.physics.user_defined_gradient('s','y')))
                    else:
                        raise ValueError(f"{d} is not found in the output_variable of the nn, and not defined")

                    # loss name
                    loss_names.append(self.params.training.additional_loss[d].name)
                    # weights
                    loss_weights.append(self.params.training.additional_loss[d].weight)
                    # append loss functions
                    loss_functions.append(self.params.training.additional_loss[d].function)

        # load the callable loss functions
        loss_functions = [data_misfit.get(l) for l in loss_functions]

        return training_temp, loss_names, loss_weights, loss_functions

    def update_parameters(self, params):
        """ update self.params according to the input params. If key already exists, update the value; if not, add the pair
        """
        # update self.params.param_dict from params
        self.params.param_dict.update(params)

        # call the constructor
        self.params = Parameters(self.params.param_dict)

        # setup the model
        self.setup()
        
    def _update_nn_parameters(self):
        """ assign physic.input_var, output_var, output_lb, and output_ub to nn
        """
        self.params.nn.set_parameters({"input_variables": self.physics.input_var, 
            "output_variables": self.physics.output_var, 
            "output_lb": self.physics.output_lb,
            "output_ub": self.physics.output_ub})

    def _update_ub_lb_in_nn(self, training_data):
        """ update input/output scalings parameters for nn
        """
        # automate the input scaling according to the domain
        if self.params.domain.time_dependent:
            self.params.nn.set_parameters({"input_lb": np.hstack((self.domain.geometry.geometry.bbox[0,:], self.domain.geometry.timedomain.bbox[0])),
                "input_ub": np.hstack((self.domain.geometry.geometry.bbox[1,:], self.domain.geometry.timedomain.bbox[1]))})
        else:
            self.params.nn.set_parameters({"input_lb": self.domain.geometry.bbox[0,:], "input_ub": self.domain.geometry.bbox[1,:]})

        # check if training data exceed the scaling range
        for i,d in enumerate(self.params.nn.output_variables):
            if d in training_data.sol:
                if np.max(training_data.sol[d]) > self.params.nn.output_ub[i]:
                    self.params.nn.output_ub[i] = np.max(training_data.sol[d])
                if np.min(training_data.sol[d]) < self.params.nn.output_lb[i]:
                    self.params.nn.output_lb[i] = np.min(training_data.sol[d])

        # wrap output_lb and output_ub with np.array
        self.params.nn.output_lb = np.array(self.params.nn.output_lb)
        self.params.nn.output_ub = np.array(self.params.nn.output_ub)
