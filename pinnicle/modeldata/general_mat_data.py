from . import DataBase
from ..parameter import SingleDataParameter
from ..physics import Constants
from ..utils import plot_dict_data, load_mat, down_sample
import numpy as np


class MatData(DataBase, Constants):
    """ data loaded from a `.mat` file
    """
    _DATA_TYPE = "mat"
    def __init__(self, parameters=SingleDataParameter()):
        Constants.__init__(self)
        super().__init__(parameters)

    def get_ice_coordinates(self, mask_name=""):
        """ stack the coordinates `x` and `y`, assuming all the data in .mat 
            are in the ice covered region. This function is currently only 
            called by plotting to generate ice covered region.
        """
        # get the coordinates
        X_mask = np.hstack([self.X_dict[k].flatten()[:,None] for k in self.parameters.X_map if k in self.X_dict])

        return X_mask

    def load_data(self):
        """ load scatter data from a `.mat` file, return a dict with the required data
        """
        # Reading matlab data
        data = load_mat(self.parameters.data_path)

        # load the coordinates
        for k, v in self.parameters.X_map.items():
            if v in data:
                self.X_dict[k] = data[v]
            else:
                print(f"{v} is not found in the data from {self.parameters.data_path}, please specify the mapping in 'X_map'")

        # load all variables from parameters.name_map
        for k in self.parameters.name_map:
            self.data_dict[k] = data[self.parameters.name_map[k]]

    def plot(self, data_names=[], vranges={}, axs=None, **kwargs):
        """ TODO: scatter plot of the selected data from data_names
        """
        pass

    def prepare_training_data(self, data_size=None):
        """ prepare data for PINNs according to the settings in `data_size`
        """
        if data_size is None:
            data_size = self.parameters.data_size

        # initialize
        self.X = {}
        self.sol = {}

        # prepare x,y coordinates
        X_temp = self.get_ice_coordinates()
        max_data_size = X_temp.shape[0]

        # go through all keys in data_dict
        for k in self.data_dict:
            # if datasize has the key, then add to X and sol
            if k in data_size:
                if data_size[k] is not None:
                    # apply ice mask
                    sol_temp = self.data_dict[k].flatten()[:,None]
                    # random choose to a downscale sampling of the scatter data
                    idx = down_sample(X_temp, data_size[k])
                    self.X[k] = X_temp[idx, :]
                    self.sol[k] = sol_temp[idx, :]
                else:
                    # if the size is None, then only use boundary conditions
                    raise ValueError(f"{k} can not be set to None in .mat data. \
                                     If {k} is not needed in training, please remove it from `data_size`")

