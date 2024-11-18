from . import DataBase
from ..parameter import SingleDataParameter
from ..physics import Constants
from ..utils import plot_dict_data, load_mat
import numpy as np


class ISSMmdData(DataBase, Constants):
    """ data loaded from model in ISSM
    """
    _DATA_TYPE = "ISSM"
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
        X_mask = np.hstack((self.X_dict['x'][iice].flatten()[:,None], self.X_dict['y'][iice].flatten()[:,None]))
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
        data = load_mat(self.parameters.data_path)
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
        self.data_dict['a'] = (md['smb']['mass_balance'] - md['balancethickness']['thickening_rate'])/self.yts
        self.data_dict['H'] = md['geometry']['thickness']
        self.data_dict['B'] = md['materials']['rheology_B']
        self.data_dict['vel'] = np.sqrt(self.data_dict['u']**2.0+self.data_dict['v']**2.0)
        # check the friction law
        if 'C' in md['friction']:
            self.data_dict['C'] = md['friction']['C'] # Weertman
        else:
            # convert Budd to Weertman type friction coefficient (m=1/3 by default)
            C_b = md['friction']['coefficient'] # Budd
            rho_ice = md['materials']['rho_ice']
            rho_w = md['materials']['rho_water']
            g = md['constants']['g']
            base = md['geometry']['base']
            N = rho_ice*g*self.data_dict['H'] + rho_w*g*base
            N[np.where(N <= 1.0, True, False)] = 1.0
            self.data_dict['C'] = C_b*np.sqrt(N)*(self.data_dict['vel']**(1.0/3.0))

        # clean up is any of the keys are empty
        self.data_dict = {k:self.data_dict[k] for k in self.data_dict if self.data_dict[k].shape != ()}
        # ice mask
        self.mask_dict['icemask'] = md['mask']['ice_levelset']
        # B.C.
        self.mask_dict['DBC_mask'] = md['mesh']['vertexonboundary']
        # mesh information
        self.mesh_dict['edges'] = md['mesh']['edges']
        self.mesh_dict['elements'] = md['mesh']['elements']
        self.mesh_dict['lat'] = md['mesh']['lat']
        self.mesh_dict['long'] = md['mesh']['long']

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

        # prepare boundary coordinates
        DBC = self.mask_dict['DBC_mask']
        idbc = np.asarray(DBC>0).nonzero()
        X_bc = np.hstack([self.X_dict[k][idbc].flatten()[:,None] for k in self.parameters.X_map if k in self.X_dict])

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

