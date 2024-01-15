import scipy.io
import numpy as np

def load_matlab_data(path): #{{{
    """
    load all data from a .mat file to a dict, remove matlab headers, reshape all the data to column vectors
    """
    # Reading matlab data
    data = scipy.io.loadmat(path,  mat_dtype=True)
    # remove headers of .mat, make all data as vector
    outdict = {k:data[k].flatten()[:,None] for k in data if not k.startswith('_')}
    return outdict #}}}
def get_lower_bound(data): #{{{
    """
    get the lower bound of the each vector in data
    """
    lb = {k:data[k].min() for k in data}
    return lb #}}}
def get_upper_bound(data): #{{{
    """
    get the upper bound of the each vector in data
    """
    ub = {k:data[k].max() for k in data}
    return ub #}}}

#def prepare_training_data(path, dim=2, numofdata={}, dataformat="mat"):
#    if isinstance(path, str):
#        # for .mat data file
#        if path.endswith(".mat") or dataformat=="mat":
#            data = load_matlab_data(path)
#            # go through all the variables in the dict
#            for k in numofdata:
#                # 
#                if numofdata[k]:
#                    idx = np.random.choice(X.shape[0], numofdata[k], replace=False)
#
#
#        else:
#            raise TypeError(dataformat+" is not implemented!")
#    else:
#        raise ValueError("path is not a string")



def prep_2D_data_all(path, N_f=None, N_u=None, N_s=None, N_H=None, N_C=None, FlightTrack=False): #{{{
    # Reading SSA ref solutions: x, y-coordinates, provide ALL the variables in u_train
    data = load_matlab_data(path)

    # viscosity
    mu = data['mu']

    x = data['x']
    y = data['y']

    # real() is to make it float by default, in case of zeroes
    Exact_vx = np.real(data['vx'].flatten()[:,None])
    Exact_vy = np.real(data['vy'].flatten()[:,None])
    Exact_h = np.real(data['h'].flatten()[:,None])
    Exact_H = np.real(data['H'].flatten()[:,None])
    Exact_C = np.real(data['C'].flatten()[:,None])

    # boundary nodes
    DBC = data['DBC'].flatten()[:,None]

    # Preparing the inputs x and y for predictions in one single array, as X_star
    X_star = np.hstack((x.flatten()[:,None], y.flatten()[:,None]))

    # Preparing the testing u_star and vy_star
    u_star = np.hstack((Exact_vx.flatten()[:,None], Exact_vy.flatten()[:,None], Exact_h.flatten()[:,None], Exact_H.flatten()[:,None], Exact_C.flatten()[:,None] ))

    # Domain bounds: for regularization and generate training set
    xlb = X_star.min(axis=0)
    xub = X_star.max(axis=0)
    umin = u_star.min(axis=0)
    umax = u_star.max(axis=0)
    names = ['u', 'v', 's', 'H', 'C']
    ulb = {k:umin[i] for i, k in enumerate(names)}
    uub = {k:umax[i] for i, k in enumerate(names)}

    # set Dirichlet boundary conditions
    idbc = np.transpose(np.asarray(DBC>0).nonzero())
    X_bc = X_star[idbc[:,0],:]
    u_bc = u_star[idbc[:,0],:]

    # Stacking them in multidimensional tensors for training, only use ice covered area
    icemask = data['icemask'].flatten()[:,None]
    iice = np.transpose(np.asarray(icemask>0).nonzero())
    X_ = np.vstack([X_star[iice[:,0],:]])
    u_ = np.vstack([u_star[iice[:,0],:]])

    # calving front info
    cx = data['cx'].flatten()[:,None]
    cy = data['cy'].flatten()[:,None]
    nx = data['smoothnx'].flatten()[:,None]
    ny = data['smoothny'].flatten()[:,None]

    X_cf = np.hstack((cx.flatten()[:,None], cy.flatten()[:,None]))
    n_cf = np.hstack((nx.flatten()[:,None], ny.flatten()[:,None]))

    # Getting the corresponding X_train and u_train(which is now scarce boundary/initial coordinates)
    X_train = {}
    u_train = {}

    # Generating a uniform random sample from ints between 0, and the size of x_u_train, of size N_u (initial data size) and without replacement (unique)
    # velocity data
    if N_u:
        idx = np.random.choice(X_.shape[0], N_u, replace=False)
        X_train["u"] = X_[idx,:]
        u_train["u"] = u_[idx, 0:1]
        X_train["v"] = X_[idx,:]
        u_train["v"] = u_[idx, 1:2]
    else:
        X_train["u"] = X_bc
        u_train["u"] = u_bc[:, 0:1]
        X_train["v"] = X_[idx,:]
        u_train["v"] = u_[idx, 1:2]

    # surface elevation, always available, use the maximum points among all the other data set
    if N_s is None:
        Nlist = [N_u, N_H, N_C]
        N_s = max([i for i in Nlist if i is not None])

    idx = np.random.choice(X_.shape[0], N_s, replace=False)
    X_train["s"] = X_[idx,:]
    u_train["s"] = u_[idx, 2:3]

    # ice thickness, or bed elevation
    if N_H:
        if FlightTrack:
            H_ft = np.real(data['H_ft'].flatten()[:,None])
            x_ft = np.real(data['x_ft'].flatten()[:,None])
            y_ft = np.real(data['y_ft'].flatten()[:,None])
            X_ft = np.hstack((x_ft.flatten()[:,None], y_ft.flatten()[:,None]))

            N_H = min(X_ft.shape[0], N_H)
            print(f"Use {N_H} flight track data for the ice thickness training data")
            idx = np.random.choice(X_ft.shape[0], N_H, replace=False)
            X_train["H"] = np.vstack([X_bc, X_ft[idx, :]])
            u_train["H"] = np.vstack([u_bc[:, 3:4], H_ft[idx,:]])
        else:
            idx = np.random.choice(X_.shape[0], N_H, replace=False)
            X_train["H"] = X_[idx,:]
            u_train["H"] = u_[idx, 3:4]
    else:
        if 'x_fl' in data.keys():
            print('Warning, using flowlines, this should only be used for proof of concept')
            # load thickness along flowlines
            H_fl = np.real(data['H_fl'].flatten()[:,None])
            x_fl = np.real(data['x_fl'].flatten()[:,None])
            y_fl = np.real(data['y_fl'].flatten()[:,None])
            X_fl = np.hstack((x_fl.flatten()[:,None], y_fl.flatten()[:,None]))

            X_train["H"] = np.vstack([X_bc, X_fl])
            u_train["H"] = np.vstack([u_bc[:, 3:4], H_fl])
        else:
            X_train["H"] = X_bc
            u_train["H"] = u_bc[:, 3:4]

    # friction coefficients
    if N_C:
        idx = np.random.choice(X_.shape[0], N_C, replace=False)
        X_train["C"] = X_[idx,:]
        u_train["C"] = u_[idx, 4:5]
    else:
        X_train["C"] = X_bc
        u_train["C"] = u_bc[:, 4:5]

    return X_star, u_star, X_train, u_train, X_bc, u_bc, X_cf, n_cf, xub, xlb, uub, ulb, mu  #}}}
def prep_2D_data(path, datasize={}): #{{{
    # Reading SSA ref solutions: x, y-coordinates, provide ALL the variables in u_train
    data = load_matlab_data(path)
    # names of the data needed
    names = ["u", "v", "s", "H", "C"]
    # viscosity
    mu = data['mu']

    x = data['x']
    y = data['y']

    # real() is to make it float by default, in case of zeroes
    Exact_vx = np.real(data['vx'].flatten()[:,None])
    Exact_vy = np.real(data['vy'].flatten()[:,None])
    Exact_h = np.real(data['h'].flatten()[:,None])
    Exact_H = np.real(data['H'].flatten()[:,None])
    Exact_C = np.real(data['C'].flatten()[:,None])

    # boundary nodes
    DBC = data['DBC'].flatten()[:,None]

    # Preparing the inputs x and y for predictions in one single array, as X_star
    X_star = np.hstack((x.flatten()[:,None], y.flatten()[:,None]))

    # Preparing the testing u_star and vy_star
    u_star = np.hstack((Exact_vx.flatten()[:,None], Exact_vy.flatten()[:,None], Exact_h.flatten()[:,None], Exact_H.flatten()[:,None], Exact_C.flatten()[:,None] ))

    # Domain bounds: for regularization and generate training set
    umin = u_star.min(axis=0)
    umax = u_star.max(axis=0)
    ulb = {k:umin[i] for i, k in enumerate(names)}
    uub = {k:umax[i] for i, k in enumerate(names)}

    # set Dirichlet boundary conditions
    idbc = np.transpose(np.asarray(DBC>0).nonzero())
    X_bc = X_star[idbc[:,0],:]
    u_bc = u_star[idbc[:,0],:]

    # Stacking them in multidimensional tensors for training, only use ice covered area
    icemask = data['icemask'].flatten()[:,None]
    iice = np.transpose(np.asarray(icemask>0).nonzero())
    X_ = np.vstack([X_star[iice[:,0],:]])
    u_ = np.vstack([u_star[iice[:,0],:]])

    # calving front info
    cx = data['cx'].flatten()[:,None]
    cy = data['cy'].flatten()[:,None]
    nx = data['smoothnx'].flatten()[:,None]
    ny = data['smoothny'].flatten()[:,None]

    X_cf = np.hstack((cx.flatten()[:,None], cy.flatten()[:,None]))
    n_cf = np.hstack((nx.flatten()[:,None], ny.flatten()[:,None]))

    # Getting the corresponding X_train and u_train(which is now scarce boundary/initial coordinates)
    X_train = {}
    u_train = {}

    # Generating a uniform random sample from ints between 0, and the size of x_u_train, of size N_u (initial data size) and without replacement (unique)
    for i, key in enumerate(names):
        trainFlag = True
        if key not in datasize:
            trainFlag = False
        else:
            if datasize[key]:
                trainFlag = True
            else:
                trainFlag = False

        if trainFlag:
            idx = np.random.choice(X_.shape[0], datasize[key], replace=False)
            X_train[key] = X_[idx,:]
            u_train[key] = u_[idx, i:i+1]
        else:
            X_train[key] = X_bc
            u_train[key] = u_bc[:, i:i+1]

    return X_star, u_star, X_train, u_train, X_bc, u_bc, X_cf, n_cf, uub, ulb, mu  #}}}
