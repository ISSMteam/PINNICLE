import numpy as np
from netCDF4 import Dataset


def interpfrombedmachine(x, y, name, path, method="linear"):
    """ interpolate data from bedmachine

    Args:
        x (np.ndarray): x coordinates
        y (np.ndarray): y coordinates
        name (str): variable name in the bedmachine data
        path (str): path to the bedmachine .nc file
        method (str, optional): interpolation method. Defaults to "linear".

    Returns:
        np.ndarray: interpolated data
    """
    from scipy.interpolate import griddata

    # load bedmachine data
    data = Dataset(path, "r")

    # get the minimum bounding rectangle of the input coordinates
    xmin = [np.min(x), np.min(y)]
    xmax = [np.max(x), np.max(y)]

    # Load coordinate arrays
    xkeys = ["x", "y"]
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
            
    x_bm = data["x"][x_start[0]:x_end[0]]
    y_bm = data["y"][x_start[1]:x_end[1]]
    X_bm, Y_bm = np.meshgrid(x_bm, y_bm)
    var_bm = data[name][x_start[1]:x_end[1], x_start[0]:x_end[0]]

    # flatten the arrays
    points = np.vstack((X_bm.flatten(), Y_bm.flatten())).T
    values = var_bm.flatten()
    xi = np.vstack((x.flatten(), y.flatten())).T

    # interpolate
    var_interp = griddata(points, values, xi, method=method)

    return var_interp.reshape(x.shape)

def subdomainmask(subdomain, input_file, name='mask', resolution=200):
    """ Check if there is ice in the subdomain by interpolating the BedMachine data

    Args:
        subdomain (tuple): (xmin, xmax, ymin, ymax) of the subdomain
        input_file (str): path to the input BedMachine .nc file
        name (str, optional): variable name to interpolate. Defaults to 'mask'.
        resolution (int, optional): resolution of the test points within the subdomain. Defaults to 200.

    Returns:
        bool: true if there is any mask>0 in the subdomain, false otherwise
    """
    x0, x1, y0, y1 = subdomain
    x = np.linspace(x0, x1, resolution)
    y = np.linspace(y0, y1, resolution)
    X, Y = np.meshgrid(x, y)

    mask = interpfrombedmachine(X, Y, 'mask', input_file,  method="linear")
    return np.any(mask > 0)