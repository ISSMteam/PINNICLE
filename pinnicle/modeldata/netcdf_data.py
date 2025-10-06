from . import DataBase
from ..parameter import SingleDataParameter
from ..physics import Constants
from ..utils import down_sample
from netCDF4 import Dataset
import numpy as np


class NetCDFData(DataBase, Constants):
    """ data loaded from a `.nc` file
    """
    _DATA_TYPE = "nc"
    def __init__(self, parameters=SingleDataParameter()):
        Constants.__init__(self)
        super().__init__(parameters)

    def get_ice_coordinates(self, mask=None):
        """ stack the coordinates `x` and `y`, assuming all the data in .mat 
            are in the ice covered region. This function is currently only 
            called by plotting to generate ice covered region.
        """
        # get the coordinates
        X_mask = np.hstack([self.X_dict[k][mask].flatten()[:,None] for k in self.parameters.X_map if k in self.X_dict])

        return X_mask

    def load_data(self, domain=None, physics=None):
        """ load grid data from a `.nc` file, based on the domain, return a dict with the required data
        """
        # Reading .nc data handler
        data = Dataset(self.parameters.data_path, "r")

        # pre load x, y, the spatial coordinates, from now on, use X and its keys only, X_map translate the data already
        X = {}
        for k, v in self.parameters.X_map.items():
            if v in data.dimensions:
                X[k] = data[v]
            else:
                print(f"{v} is not found in the data from {self.parameters.data_path}, please specify the mapping in 'X_map'")

        # use the order in physics.input_var to determine x and y names
        if physics:
            xkeys = physics.input_var[0:2]
        else:
            xkeys = list(X.keys()) 

        # get the bbox from domain, set the rectangle, works for both static and time dependent domain
        if domain:
            bbox = domain.bbox()
            # set the flag based on the bbox region
            xmin = [bbox[0][0], bbox[0][1]]
            xmax = [bbox[1][0], bbox[1][1]]
        else:
            # otherwise use the whole data
            xmin = [data.variables[k][:].min() for k in xkeys]
            xmax = [data.variables[k][:].max() for k in xkeys]

        # Load coordinate arrays
        x_coord = [data[k][:] for k in xkeys]

        # Find indices in Xs
        x_start = []
        x_end = []
        for i, k in enumerate(xkeys):
            x_inds = np.where((x_coord[i] >= xmin[i]) & (x_coord[i] <= xmax[i]))[0]
            if len(x_inds) > 0:
                x_start.append(x_inds[0])
                x_end.append(x_inds[-1] + 1)
            else:
                raise ValueError("No x indices found in range.")

        # load all variables from parameters.name_map
        for k,v in self.parameters.name_map.items():
            self.data_dict[k] = (data.variables[v][x_start[1]:x_end[1], x_start[0]:x_end[0]].flatten()[:,None])*self.parameters.scaling[k]

        # load and generate the coordinates
        x_slice = X[xkeys[0]][x_start[0]: x_end[0]]
        y_slice = X[xkeys[1]][x_start[1]: x_end[1]]

        # Create meshgrid
        X_mesh, Y_mesh = np.meshgrid(x_slice, y_slice)

        self.X_dict[xkeys[0]] = X_mesh.flatten()[:,None]
        self.X_dict[xkeys[1]] = Y_mesh.flatten()[:,None]

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

        # go through all keys in data_dict
        for k in self.data_dict:
            # if datasize has the key, then add to X and sol
            if k in data_size:
                if data_size[k] is not None:
                    # apply nan mask
                    _temp = self.data_dict[k].flatten()[:,None]
                    mask = ~np.isnan(_temp)
                    sol_temp = _temp[mask].flatten()[:,None]

                    # prepare x,y coordinates
                    X_temp = self.get_ice_coordinates(mask=mask)

                    # random choose to a downscale sampling of the scatter data
                    idx = down_sample(X_temp, data_size[k])
                    self.X[k] = X_temp[idx, :]
                    self.sol[k] = sol_temp[idx, :]
                else:
                    # if the size is None, then only use boundary conditions
                    raise ValueError(f"{k} can not be set to None in .mat data. \
                                     If {k} is not needed in training, please remove it from `data_size`")

