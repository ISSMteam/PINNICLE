import json
import os
import mat73
import scipy.io
from sklearn.neighbors import KDTree
import numpy as np


def is_file_ext(path, ext):
    """ check if a given path is ended by ext
    """
    if isinstance(path, str):
        if path.endswith(ext):
            return True
    return False

def save_dict_to_json(data, path, filename):
    """ Save the dict to the path as a .json file

    Args:
        data (dict): Dictionary data to save
        path (Path, str): Path to save
        filename (str): name to save, attach '.json' if needed
    """
    if not filename.endswith('.json'):
        filename += '.json'
    with open(os.path.join(path,filename), "w") as fp:
        json.dump(data, fp)

def load_dict_from_json(path, filename):
    """ Load a dict data from a .json file 

    Args:
        path (Path, str): Path to load
        filename (str): name to load, attach '.json' if needed
    Returns:
        data (dict): Dictionary data to load
    """
    if not filename.endswith('.json'):
        filename += '.json'
    f = os.path.join(path, filename)
    if os.path.isfile(f):
        with open(f, 'r') as fp:
            data = json.load(fp)
    else:
        data = {}
    return data

def load_mat(file):
    """ load .mat file, if the file is in MATLAB 7.3 format use mat73.loadmat, otherwise use scipy.io.loadmat()
    """
    try:
        data = mat73.loadmat(file)
    except TypeError:
        data = scipy.io.loadmat(file)
    return data

def down_sample_core(points, resolution=100):
    """ downsample the given scatter points using `KDtree` with the nearest neighbors on a Cartisian grid

    Args:
        points (np array), 1,2 or 3 dimensional coordinates, from get_ice_coordinates
        resolution (Integer): resolution of the downsample grid
    Returns:
        ind: indices of the downsample
    """
    Xmin = points.min(axis=0)
    Xmax = points.max(axis=0)

    kdt = KDTree(points, metric='euclidean')

    # create Cartisian grid
    if Xmin.shape[0] == 1:
        X, = np.meshgrid(np.linspace(Xmin,Xmax,resolution))
        dist, ink = kdt.query(np.c_[X.ravel()], k=1)
    elif Xmin.shape[0] == 2:
        X, Y = np.meshgrid(np.linspace(Xmin[0],Xmax[0],resolution), np.linspace(Xmin[1],Xmax[1],resolution))
        dist, ink = kdt.query(np.c_[X.ravel(), Y.ravel()], k=1)
    elif Xmin.shape[0] == 3:
        X, Y, Z = np.meshgrid(np.linspace(Xmin[0],Xmax[0],resolution), np.linspace(Xmin[1],Xmax[1],resolution), np.linspace(Xmin[2],Xmax[2],resolution))
        dist, ink = kdt.query(np.c_[X.ravel(), Y.ravel(), Z.ravel()], k=1)
    else:
        raise ValueError(f"data points in {Xmin.shape[0]} dimensional is not supported")

    ind = np.unique(ink)
    return ind

def down_sample(points, data_size):
    """ downsample points to be a size of `data_size`, the strategy is to call `down_sample_core` with at least double resolution required, then randomly choose

    Args:
        points (np array), 2d coordinates, from get_ice_coordinates
        data_size (Integer): number of data points needed
    Returns:
        ind: indices of the downsample
    """
    # if data_size is larger than the number of points, use all points
    if isinstance(data_size, str):
        if data_size.lower() == 'max':
            # use all the points
            data_size = points.shape[0]

    data_size = min(points.shape[0], data_size)

    # start with double resolution
    resolution = 2*int(np.ceil(data_size**(1.0/points.shape[1])))
    ind = down_sample_core(points, resolution=resolution)

    while (resolution**points.shape[1] < points.shape[0]) and (ind.shape[0]< data_size):
        resolution *= 2
        ind = down_sample_core(points, resolution=resolution)

    if ind.shape[0] < data_size:
        # not enough data, then just return all available data
        return ind
    else:
        # randomly choose 
        idx = np.random.choice(ind, data_size, replace=False)
        return idx

def createsubdomain(xmin, ymin, xid, yid, dx=50000, dy=50000, margin=0):
    """ Create a rectangle subdomain within the large domain defined by the bottom-left corner and x, y indices

    Args:
        xmin (float): x coordinate of bottom-left corner of the large domain
        ymin (float): y coordinate of bottom-left corner of the large domain
        xid (int): x index of the subdomain (starting from 0)
        yid (int): y index of the subdomain (starting from 0)
        dx (float, optional): width of the subdomain. Defaults to 50000.
        dy (float, optional): height of the subdomain. Defaults to 50000.
        margin (float, optional): between 0 and 1, percentage of the margin extended beyond the given subdomain size

    Returns:
        tuple: (x0, x1, y0, y1) coordinates of the subdomain, this need to be consistent with domain.shapebox
    """
    if (margin < 0) or (margin > 1):
        raise ValueError(f"margin={margin} need to be a percentage between 0 and 1")
    
    x0 = xmin + (xid-margin) * dx
    y0 = ymin + (yid-margin) * dy
    x1 = xmin + (xid + 1 + margin) * dx
    y1 = ymin + (yid + 1 + margin) * dy

    return x0, x1, y0, y1    

def feathered_rect_weights(x, y, rect, width):
    """ Compute 2D weights for points (x, y) relative to an axis-aligned rectangle.
    Inside the rectangle:    weight = 1
    Outside, within `width`: weight = 1 - d / width   (linear falloff)
    Outside, beyond `width`: weight = 0
    
    Args:
        x (float or array): x-coordinates
        y (float or array): y-coordinates  
        rect (tuple):Rectangle bounds (xmin, xmax, ymin, ymax).
        width (float): Non-negative feather distance outside the rectangle over which the weight 
        falls linearly from 1 to 0. If width <= 0, this reduces to a hard mask.
    
    Returns:
        (ndarray): Weights in [0, 1] with shape broadcast from x and y.
    """
    xmin, xmax, ymin, ymax = rect
    x = np.asarray(x)
    y = np.asarray(y)

    # Distance to rectangle (signed): negative inside, positive outside.
    # Outside distance (Euclidean) to the rectangle:
    dx_out = np.maximum(np.maximum(xmin - x, 0), x - xmax)
    dy_out = np.maximum(np.maximum(ymin - y, 0), y - ymax)
    dist_outside = np.hypot(dx_out, dy_out)

    # Inside distance to the nearest edge (>= 0)
    inside = (dx_out == 0) & (dy_out == 0)
    dist_to_edge_inside = np.minimum.reduce([x - xmin, xmax - x, y - ymin, ymax - y])

    # Signed distance: negative inside, positive outside
    signed_d = np.where(inside, -dist_to_edge_inside, dist_outside)

    # Linear falloff outside over `width`
    w = 1.0 - np.clip(signed_d, 0, None) / width
    w = np.clip(w, 0.0, 1.0)
    return w
