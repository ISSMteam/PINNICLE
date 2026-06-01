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
        if mask is None:
            X_mask = np.hstack([self.X_dict[k].flatten()[:,None] for k in self.parameters.X_map if k in self.X_dict])
        else:
            X_mask = np.hstack([self.X_dict[k][mask].flatten()[:,None] for k in self.parameters.X_map if k in self.X_dict])

        return X_mask

    def _coordinate_slice(self, coord, lower, upper, name):
        """Return a contiguous slice for a coordinate vector and requested bounds."""
        coord = np.asarray(coord)
        if coord.ndim != 1:
            inds = np.where((coord >= lower) & (coord <= upper))[0]
            if len(inds) == 0:
                raise ValueError(f"No {name} indices found in range.")
            return slice(inds[0], inds[-1] + 1)

        if coord.size == 0:
            raise ValueError(f"No {name} coordinates found.")

        is_increasing = coord[0] <= coord[-1]
        if is_increasing:
            start = np.searchsorted(coord, lower, side="left")
            end = np.searchsorted(coord, upper, side="right")
        else:
            coord_rev = coord[::-1]
            rev_start = np.searchsorted(coord_rev, lower, side="left")
            rev_end = np.searchsorted(coord_rev, upper, side="right")
            start = coord.size - rev_end
            end = coord.size - rev_start

        if start >= end:
            raise ValueError(f"No {name} indices found in range.")
        return slice(start, end)

    def load_data(self, domain=None, physics=None):
        """ load grid data from a `.nc` file, based on the domain, return a dict with the required data
        """
        with Dataset(self.parameters.data_path, "r") as data:
            # pre load x, y, the spatial coordinates, from now on, use X and its keys only, X_map translate the data already
            X = {}
            for k, v in self.parameters.X_map.items():
                if v in data.variables:
                    X[k] = data.variables[v]
                else:
                    raise KeyError(
                        f"{v} is not found in the data from {self.parameters.data_path}, "
                        "please specify the mapping in 'X_map'"
                    )

            # use the order in physics.input_var to determine x and y names
            if physics:
                xkeys = physics.input_var[0:2]
            else:
                xkeys = list(X.keys())

            # Load coordinate arrays once. They are tiny compared with gridded fields
            # and are reused for slicing, mesh generation, and optional polygon masks.
            x_coord = {k: np.asarray(X[k][:]) for k in xkeys}

            # get the bbox from domain, set the rectangle, works for both static and time dependent domain
            if domain:
                bbox = domain.bbox()
                xmin = [bbox[0][0], bbox[0][1]]
                xmax = [bbox[1][0], bbox[1][1]]
            else:
                xmin = [np.nanmin(x_coord[k]) for k in xkeys]
                xmax = [np.nanmax(x_coord[k]) for k in xkeys]

            coord_slices = {
                xkeys[i]: self._coordinate_slice(x_coord[xkeys[i]], xmin[i], xmax[i], xkeys[i])
                for i in range(2)
            }

            # load and generate the coordinates
            x_slice = x_coord[xkeys[0]][coord_slices[xkeys[0]]]
            y_slice = x_coord[xkeys[1]][coord_slices[xkeys[1]]]
            self.X_dict[xkeys[0]] = np.tile(x_slice, y_slice.size).reshape(-1, 1)
            self.X_dict[xkeys[1]] = np.repeat(y_slice, x_slice.size).reshape(-1, 1)

            # load all variables from parameters.name_map
            data_slice = (coord_slices[xkeys[1]], coord_slices[xkeys[0]])
            for k, v in self.parameters.name_map.items():
                if v not in data.variables:
                    raise KeyError(
                        f"{v} is not found in the data from {self.parameters.data_path}, "
                        "please specify the mapping in 'name_map'"
                    )
                scaling = self.parameters.scaling.get(k, 1.0)
                self.data_dict[k] = (np.ma.asarray(data.variables[v][data_slice]).reshape(-1, 1))*scaling

            if self.parameters.sample_only_inside:
                P = np.hstack((self.X_dict[xkeys[0]], self.X_dict[xkeys[1]]))
                mask = np.asarray(domain.inside(P)).astype(bool).reshape(-1)
                self.X_dict[xkeys[0]] = self.X_dict[xkeys[0]][mask]
                self.X_dict[xkeys[1]] = self.X_dict[xkeys[1]][mask]
                for k, v in self.parameters.name_map.items():
                    self.data_dict[k] = self.data_dict[k][mask]


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
                    _temp = np.ma.masked_invalid(np.ma.asarray(self.data_dict[k]).reshape(-1, 1))
                    mask = ~np.ma.getmaskarray(_temp).reshape(-1)
                    sol_temp = np.asarray(_temp.compressed()).reshape(-1, 1)

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
