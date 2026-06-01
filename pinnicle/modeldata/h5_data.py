from . import DataBase
from ..parameter import SingleDataParameter
from ..physics import Constants
from ..utils import down_sample
import numpy as np
import h5py


class H5Data(DataBase, Constants):
    """ data loaded from a `.h5` file
    """
    _DATA_TYPE = "h5"
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
        """ load grid data from a `.h5` file, based on the domain, return a dict with the required data
        """
        with h5py.File(self.parameters.data_path, 'r') as data:
            # pre load x, y, then use inside() to further get the inflag
            X = {}
            for k, v in self.parameters.X_map.items():
                if v in data.keys():
                    X[k] = data[v]
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

            X_arrays = {k: np.asarray(X[k]) for k in X.keys()}

            # get the bbox from domain, set the rectangle, works for both static and time dependent domain
            if domain:
                bbox = domain.bbox()
                boxflag = (
                    (X_arrays[xkeys[0]] >= bbox[0][0])
                    & (X_arrays[xkeys[0]] <= bbox[1][0])
                    & (X_arrays[xkeys[1]] >= bbox[0][1])
                    & (X_arrays[xkeys[1]] <= bbox[1][1])
                )
            else:
                boxflag = np.ones_like(X_arrays[xkeys[0]], dtype=bool)

            if not np.any(boxflag):
                raise ValueError("No HDF5 coordinates found in domain range.")

            data_selection = tuple(slice(None) for _ in boxflag.shape)
            selection_mask = None
            if domain and boxflag.ndim == 2:
                rows, cols = np.where(boxflag)
                data_selection = (slice(rows[0], rows[-1] + 1), slice(cols.min(), cols.max() + 1))
                selection_mask = boxflag[data_selection]
            elif domain:
                selection_mask = boxflag

            def select_dataset(dataset):
                arr = np.asarray(dataset[data_selection])
                if selection_mask is not None:
                    arr = arr[selection_mask]
                return arr.reshape(-1, 1)

            def select_array(arr):
                arr = np.asarray(arr[data_selection])
                if selection_mask is not None:
                    arr = arr[selection_mask]
                return arr.reshape(-1, 1)

            # load the coordinates
            for k in X.keys():
                self.X_dict[k] = select_array(X_arrays[k])

            inside_mask = None
            if self.parameters.sample_only_inside:
                P = np.hstack((self.X_dict[xkeys[0]], self.X_dict[xkeys[1]]))
                inside_mask = np.asarray(domain.inside(P)).astype(bool).reshape(-1)
                for k in X.keys():
                    self.X_dict[k] = self.X_dict[k][inside_mask]

            # load all variables from parameters.name_map
            for k, v in self.parameters.name_map.items():
                if v not in data.keys():
                    raise KeyError(
                        f"{v} is not found in the data from {self.parameters.data_path}, "
                        "please specify the mapping in 'name_map'"
                    )
                data_values = select_dataset(data[v])*self.parameters.scaling.get(k, 1.0)
                if inside_mask is not None:
                    data_values = data_values[inside_mask]
                self.data_dict[k] = data_values

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
        sample_cache = {}

        # go through all keys in data_dict
        for k in self.data_dict:
            # if datasize has the key, then add to X and sol
            if k in data_size:
                if data_size[k] is not None:
                    # apply ice mask
                    sol_temp = self.data_dict[k].flatten()[:,None]
                    # random choose to a downscale sampling of the scatter data
                    cache_key = str(data_size[k]).lower()
                    if cache_key not in sample_cache:
                        sample_cache[cache_key] = down_sample(X_temp, data_size[k])
                    idx = sample_cache[cache_key]
                    self.X[k] = X_temp[idx, :]
                    self.sol[k] = sol_temp[idx, :]
                else:
                    # if the size is None, then only use boundary conditions
                    raise ValueError(f"{k} can not be set to None in .mat data. \
                                     If {k} is not needed in training, please remove it from `data_size`")
