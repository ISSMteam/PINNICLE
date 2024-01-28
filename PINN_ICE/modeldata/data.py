from abc import ABC, abstractmethod
from ..parameter import DataParameter
import os
import numpy as np
import mat73

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

    def __init__(self, parameters=DataParameter()):
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
    def load_data(self):
        """ load data from self.path
        """
        pass


class ISSMmdData(DataBase):
    """ data loaded from model in ISSM
    """
    _DATA_TYPE = "ISSM"
    def __init__(self, parameters=DataParameter()):
        super().__init__(parameters)

    def load_data(self):
        """ load ISSM model from a .mat file, return a dict with the required data
        """
        # Reading matlab data
        data = mat73.loadmat(self.parameters.data_path)
        # get the model
        md = data['md']
        # create the output dict
        # x,y coordinates
        self.X_dict['x'] = md['mesh']['x']
        self.X_dict['y'] = md['mesh']['y']
        # data
        self.data_dict['u'] = md['inversion']['vx_obs']
        self.data_dict['v'] = md['inversion']['vy_obs']
        self.data_dict['s'] = md['geometry']['surface']
        self.data_dict['H'] = md['geometry']['thickness']
        self.data_dict['C'] = md['friction']['C']
        self.data_dict['B'] = md['materials']['rheology_B']
        # ice mask
        self.mask_dict['icemask'] = md['mask']['ice_levelset']
        # B.C.
        self.mask_dict['DBC_mask'] = md['mesh']['vertexonboundary']

    def prepare_training_data(self, data_size=None):
        """ prepare data for PINNs according to the settings in datasize
        """
        if data_size is None:
            data_size = self.parameters.data_size

        # initialize
        self.X = {}
        self.sol = {}

        # prepare x,y coordinates
        icemask = self.mask_dict['icemask']
        iice = np.asarray(icemask<0).nonzero()
        X_temp = np.hstack((self.X_dict['x'][iice].flatten()[:,None], self.X_dict['y'][iice].flatten()[:,None]))
        max_data_size = X_temp.shape[0]

        # prepare boundary coordinates
        DBC = self.mask_dict['DBC_mask']
        idbc = np.asarray(DBC>0).nonzero()
        X_bc = np.hstack((self.X_dict['x'][idbc].flatten()[:,None], self.X_dict['y'][idbc].flatten()[:,None]))

        # go through all keys in data_dict
        for k in self.data_dict:
            # if datasize has the key, then add to X and sol
            if k in data_size:
                if data_size[k] is not None:
                    # apply ice mask
                    sol_temp = self.data_dict[k][iice].flatten()[:,None]
                    # randomly choose, replace=False for no repeat data
                    idx = np.random.choice(max_data_size, min(data_size[k],max_data_size), replace=False)
                    self.X[k] = X_temp[idx, :]
                    self.sol[k] = sol_temp[idx, :]
                else:
                    # if the size is None, then only use boundary conditions
                    self.X[k] = X_bc
                    self.sol[k] = self.data_dict[k][idbc].flatten()[:,None]

# TODO: add plot
