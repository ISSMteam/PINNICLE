from abc import ABC, abstractmethod
from ..parameter import DataParameter, SingleDataParameter
from ..physics import Constants
import numpy as np
import deepxde as dde


class DataBase(ABC):
    """ Base class of data
    """
    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls._DATA_TYPE] = cls

    @classmethod
    def create(cls, data_type,  **kwargs):
        if data_type not in cls.subclasses:
            raise ValueError(f"Data type {format(data_type)} is not defined")
        return cls.subclasses[data_type](**kwargs)

    def __init__(self, parameters=SingleDataParameter()):
        # parameters
        self.parameters = parameters
        # load data to dict
        self.X_dict = {}
        self.data_dict = {}
        self.mask_dict = {}
        self.mesh_dict = {}

        # input to PINN
        self.X = None
        # reference solution of the output of PINN
        self.sol = None

    @abstractmethod
    def get_ice_coordinates(self, mask_name=""):
        """ get ice masks if available from the data
        """
        pass

    @abstractmethod
    def load_data(self, domain, physics):
        """ load data from `self.parameters.data_path`, within the given `domain` and `physics`
        """
        pass

    @abstractmethod
    def prepare_training_data(self):
        """ prepare training data according to the `data_size`
        """
        pass


class Data(Constants):
    """ class of data with all data used
    """
    def __init__(self, parameters=DataParameter()):
        super().__init__()
        self.parameters = parameters
        # create all instances of Data based on its source, we can have multiple data from the same source
        self.data = {k:DataBase.create(parameters.data[k].source, parameters=parameters.data[k]) for k in parameters.data}

        # input to PINN
        self.X = {}
        # reference solution of the output of PINN
        self.sol = {}

    def get_ice_coordinates(self, mask_name=""):
        """ get the coordinates of ice covered region from all the data, put them in one array
        """
        return np.vstack([self.data[k].get_ice_coordinates(mask_name=mask_name) for k in self.data])

    def load_data(self, domain=None, physics=None):
        """ laod all the data in `self.data`
        """
        for k in self.data:
            self.data[k].load_data(domain, physics)

    def prepare_training_data(self, transient=False, default_time=0.0):
        """ merge all `X` and `sol` in `self.data` to `self.X` and `self.sol` with the keys
        Args:
            transient: if the problem is a time dependent simulation
            default_time: default value of the third column (time) in `X`, if not provided by the data
        """
        # prepare the training data according to data_size
        for key in self.data:
            self.data[key].prepare_training_data()
            # merge all X and sol
            for xkey, xval in self.data[key].X.items():
                # check if the data has time dimension, if not, append one column with default_time to the end
                if transient:
                    if xval.shape[1] < 3:
                        # check default_time setting in the data
                        if self.data[key].parameters.default_time is not None:
                            default_time = self.data[key].parameters.default_time
                        xval = np.hstack((xval, np.ones([xval.shape[0],1])*default_time))
                if xkey not in self.X:
                    self.X[xkey] = xval.astype(dde.config.default_float())
                else:
                    self.X[xkey] = np.vstack((self.X[xkey], xval.astype(dde.config.default_float())))

            for xkey in self.data[key].sol:
                if xkey not in self.sol:
                    self.sol[xkey] = self.data[key].sol[xkey].astype(dde.config.default_float())
                else:
                    self.sol[xkey] = np.vstack((self.sol[xkey], self.data[key].sol[xkey].astype(dde.config.default_float())))
