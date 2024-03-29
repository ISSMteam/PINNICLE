from abc import ABC, abstractmethod
from ..parameter import DataParameter, SingleDataParameter
from ..physics import Constants
import numpy as np


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
    def load_data(self):
        """ load data from self.path
        """
        pass

    @abstractmethod
    def prepare_training_data(self):
        """ prepare training data according to the data_size
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

    def load_data(self):
        """ laod all the data in self.data
        """
        for k in self.data:
            self.data[k].load_data()

    def prepare_training_data(self):
        """ merge all X and sol in self.data to self.X and self.sol with the keys 
        """
        # prepare the training data according to data_size
        for key in self.data:
            self.data[key].prepare_training_data()
            # merge all X and sol
            for xkey in self.data[key].X:
                if xkey not in self.X:
                    self.X[xkey] = self.data[key].X[xkey]
                else:
                    self.X[xkey] = np.vstack((self.X[xkey], self.data[key].X[xkey]))

            for xkey in self.data[key].sol:
                if xkey not in self.sol:
                    self.sol[xkey] = self.data[key].sol[xkey]
                else:
                    self.sol[xkey] = np.vstack((self.sol[xkey], self.data[key].sol[xkey]))
