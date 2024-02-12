from abc import ABC, abstractmethod
from ..parameter import DataParameter
from ..physics import Constants
from ..utils import plot_data
import mat73
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


class ISSMmdData(DataBase, Constants):
    """ data loaded from model in ISSM
    """
    _DATA_TYPE = "ISSM"
    def __init__(self, parameters=DataParameter()):
        Constants.__init__(self)
        super().__init__(parameters)

    def get_ice_coordinates(self, mask_name=""):
        """ get the coordinates and index of ice covered region for X_dict and data_dict
        """
        if (not mask_name) or (mask_name not in self.mask_dict):
            mask_name = "icemask"

        # get ice mask
        icemask = self.mask_dict[mask_name]
        iice = np.asarray(icemask<0).nonzero()
        return iice

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
        self.data_dict['u'] = md['inversion']['vx_obs']/self.yts
        self.data_dict['v'] = md['inversion']['vy_obs']/self.yts
        self.data_dict['s'] = md['geometry']['surface']
        self.data_dict['H'] = md['geometry']['thickness']
        self.data_dict['C'] = md['friction']['C']
        self.data_dict['B'] = md['materials']['rheology_B']
        self.data_dict['vel'] = np.sqrt(self.data_dict['u']**2.0+self.data_dict['v']**2.0)
        # ice mask
        self.mask_dict['icemask'] = md['mask']['ice_levelset']
        # B.C.
        self.mask_dict['DBC_mask'] = md['mesh']['vertexonboundary']

    def plot(self, data_names=[], vranges={}, axs=None, resolution=200, **kwargs):
        """ use utils.plot_data to plot the ISSM data 
        Args:
            data_names (list): Names of the variables. if not specified, plot all variables in data_dict
            vranges (dict): range of the data
            axs (array of AxesSubplot): axes to plot each data, if not given, then generate a subplot according to the size of data_names
            resolution (int): number of pixels in horizontal and vertical direction
        return:
            axs (array of AxesSubplot): axes of the subplots
        """
        if not data_names:
            # default value of data_names
            data_names = list(self.data_dict.keys())
        else:
            # compare with data_dict, find all avaliable
            data_names = [k for k in data_names if k in self.data_dict]

        # get the subdict of the data to plot
        data_dict = {k:self.data_dict[k] for k in data_names}

        # call the function in utils
        axs = plot_data(self.X_dict, data_dict, vranges=vranges, axs=axs, resolution=resolution, **kwargs)

        return axs

    def prepare_training_data(self, data_size=None):
        """ prepare data for PINNs according to the settings in datasize
        """
        if data_size is None:
            data_size = self.parameters.data_size

        # initialize
        self.X = {}
        self.sol = {}

        # prepare x,y coordinates
        iice = self.get_ice_coordinates()
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

