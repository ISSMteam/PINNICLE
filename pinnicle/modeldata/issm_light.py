from . import DataBase
from ..parameter import SingleDataParameter
from ..physics import Constants
from ..utils import plot_dict_data, load_mat
import numpy as np


class ISSMLightData(DataBase, Constants):
    """ 
    Data loader for the model from ISSM
    A light version, does not contain mesh and boundary info, so that one can use the bbox of a domain to select the data only inside
    """
    _DATA_TYPE = "ISSM Light"
    def __init__(self, parameters=SingleDataParameter()):
        Constants.__init__(self)
        super().__init__(parameters)

    def get_ice_coordinates(self, mask_name=""):
        """ Use `get_ice_indices` defined by each individual class,
            get the coordinates `(x,y)` of ice covered region from `X_dict`.
            This function is currently only called by plotting to generate
            ice covered region.
        """
        iice = self.get_ice_indices(mask_name=mask_name)
        X_mask = np.hstack([self.X_dict[k][iice].flatten()[:,None] for k in self.parameters.X_map if k in self.X_dict])
        return X_mask

    def get_ice_indices(self, mask_name=""):
        """ get the indices of ice covered region for `X_dict` and `data_dict`
        """
        if (not mask_name) or (mask_name not in self.mask_dict):
            mask_name = "icemask"

        # get ice mask
        icemask = self.mask_dict[mask_name]
        iice = np.asarray(icemask<0).nonzero()
        return iice

    def load_data(self, domain=None, physics=None):
        """ load ISSM model from a `.mat` file
        """
        # Reading matlab data
        mddata = load_mat(self.parameters.data_path)
        # get the model
        md = mddata['md']
        # initialize temp dict
        X = {}
        data = {}
        mask = {}
        # create the output dict
        # x,y coordinates
        for k, v in self.parameters.X_map.items():
            if v in md['mesh']:
                X[k] = md['mesh'][v]
        # data
        data['u'] = md['inversion']['vx_obs']/self.yts
        data['v'] = md['inversion']['vy_obs']/self.yts
        data['s'] = md['geometry']['surface']
        data['a'] = (md['smb']['mass_balance'] - md['balancethickness']['thickening_rate'])/self.yts
        data['H'] = md['geometry']['thickness']
        data['B'] = md['materials']['rheology_B']
        data['vel'] = np.sqrt(data['u']**2.0+data['v']**2.0)
        # check the friction law
        if 'C' in md['friction']:
            data['C'] = md['friction']['C'] # Weertman
        else:
            # convert Budd to Weertman type friction coefficient (m=1/3 by default)
            C_b = md['friction']['coefficient'] # Budd
            rho_ice = md['materials']['rho_ice']
            rho_w = md['materials']['rho_water']
            g = md['constants']['g']
            base = md['geometry']['base']
            N = rho_ice*g*self.data_dict['H'] + rho_w*g*base
            N[np.where(N <= 1.0, True, False)] = 1.0
            data['C'] = C_b*np.sqrt(N)*(data['vel']**(1.0/3.0))

        # clean up is any of the keys are empty
        data = {k:data[k] for k in data if data[k].shape != ()}
        # ice mask
        mask['icemask'] = md['mask']['ice_levelset']

        # use the order in physics.input_var to determine x and y names
        if physics:
            xkeys = physics.input_var[0:2]
        else:
            xkeys = list(X.keys())

        # get the bbox from domain, set the rectangle, works for both static and time dependent domain
        if domain:
            bbox = domain.bbox()
            # set the flag based on the bbox region
            boxflag = (X[xkeys[0]]>=bbox[0][0]) & (X[xkeys[0]]<=bbox[1][0]) & (X[xkeys[1]]>=bbox[0][1]) & (X[xkeys[1]]<=bbox[1][1])
        else:
            boxflag = np.ones_like(X[xkeys[0]], dtype=bool)

        # load the coordinates
        for k in X.keys():
            self.X_dict[k] = X[k][boxflag].flatten()[:,None]

        # load all variables from parameters.name_map
        for k in self.parameters.name_map:
            self.data_dict[k] = data[self.parameters.name_map[k]][boxflag].flatten()[:,None]

        for k in mask.keys():
            self.mask_dict[k] =  mask[k][boxflag].flatten()[:,None]

    def plot(self, data_names=[], vranges={}, axs=None, resolution=200, **kwargs):
        """ use `utils.plot_dict_data` to plot the ISSM data

        Args:
            data_names (list): Names of the variables. if not specified, plot all variables in data_dict
            vranges (dict): range of the data
            axs (array of AxesSubplot): axes to plot each data, if not given, then generate a subplot according to the size of data_names
            resolution (int): number of pixels in horizontal and vertical direction
        return:
            X (np.array): x-coordinates of the 2D plot
            Y (np.array): y-coordinates of the 2D plot
            im_data (dict): Dict of data for the 2D plot, each element has the same size as X and Y
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
        X, Y, im_data, axs = plot_dict_data(self.X_dict, data_dict, vranges=vranges, axs=axs, resolution=resolution, **kwargs)

        return X, Y, im_data, axs

    def prepare_training_data(self, data_size=None):
        """ prepare data for PINNs according to the settings in `data_size`
        """
        if data_size is None:
            data_size = self.parameters.data_size

        # initialize
        self.X = {}
        self.sol = {}

        # prepare x,y coordinates
        iice = self.get_ice_indices()
        X_temp = np.hstack([self.X_dict[k][iice].flatten()[:,None] for k in self.parameters.X_map if k in self.X_dict])
        max_data_size = X_temp.shape[0]

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
                    raise ValueError(f"{k} can not be set to None in light version ISSM data. \
                                     If {k} is not needed in training, please remove it from `data_size`")
