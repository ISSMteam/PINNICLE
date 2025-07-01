from . import DataBase
from ..parameter import SingleDataParameter
from ..physics import Constants
from ..utils import down_sample
import netCDF4  # this need to be imported to avoid xarray conflict with h5py
import xarray as xr
import h5py
import numpy as np


class NetCDFData(DataBase, Constants):
    """ data loaded from a `.nc` file
    """
    _DATA_TYPE = "nc"
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

    def load_data(self, domain=None, physics=None):
        """ load grid data from a `.nc` file, based on the domain, return a dict with the required data
        """
        # Reading .nc data handler
        data = xr.load_dataset(self.parameters.data_path, engine='netcdf4')

        # pre load x, y, the spatial coordinates, from now on, use X and its keys only, X_map translate the data already
        X = {}
        for k, v in self.parameters.X_map.items():
            if v in data.coords:
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
            xmin = bbox[0][0]
            xmax = bbox[1][0]
            ymin = bbox[0][1]
            ymax = bbox[1][1]
        else:
            # otherwise use the whole data
            xmin = (X[xkeys[0]].min()).values
            xmax = (X[xkeys[0]].max()).values
            ymin = (X[xkeys[1]].min()).values
            ymax = (X[xkeys[1]].max()).values

        # Load coordinate arrays
        x_coord = X[xkeys[0]].values
        y_coord = X[xkeys[1]].values

        # Find indices in x
        x_inds = np.where((x_coord >= xmin) & (x_coord <= xmax))[0]
        if len(x_inds) > 0:
            x_start = x_inds[0]
            x_end   = x_inds[-1] + 1
        else:
            raise ValueError("No x indices found in range.")
        
        # Find indices in y
        y_inds = np.where((y_coord >= ymin) & (y_coord <= ymax))[0]
        if len(y_inds) > 0:
            y_start = y_inds[0]
            y_end   = y_inds[-1] + 1
        else:
            raise ValueError("No y indices found in range.")

        # load all variables from parameters.name_map
        sub_data = data.isel(
                **{xkeys[1]: slice(y_start, y_end), 
                    xkeys[0]: slice(x_start, x_end)})

        for k in self.parameters.name_map:
            self.data_dict[k] = sub_data[self.parameters.name_map[k]].values.flatten()[:,None]

        # load and generate the coordinates
        x_slice = X[xkeys[0]].isel({xkeys[0]: slice(x_start, x_end)}).values
        y_slice = X[xkeys[1]].isel({xkeys[1]: slice(y_start, y_end)}).values

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

