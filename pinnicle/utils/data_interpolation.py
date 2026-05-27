import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from .helper import *


def interpfrombedmachine(x, y, name, path, method="linear"):
    """
    Faster interpolation from BedMachine using RegularGridInterpolator.

    This is intended to replace interpfrombedmachine().
    It is much faster than scipy.interpolate.griddata for regular gridded data.

    Args:
        x (np.ndarray): x coordinates
        y (np.ndarray): y coordinates
        name (str): variable name in the bedmachine data
        path (str): path to the bedmachine .nc file
        method (str, optional): interpolation method. Defaults to "linear".

    Returns:
        np.ndarray: interpolated data
    """

    x_query = np.asarray(x)
    y_query = np.asarray(y)

    with Dataset(path, "r") as data:
        x_coord = np.asarray(data["x"][:])
        y_coord = np.asarray(data["y"][:])

        ix0, ix1 = coord_window_indices(
            x_coord,
            np.nanmin(x_query),
            np.nanmax(x_query),
            pad=1
        )

        iy0, iy1 = coord_window_indices(
            y_coord,
            np.nanmin(y_query),
            np.nanmax(y_query),
            pad=1
        )

        x_bm = np.asarray(x_coord[ix0:ix1])
        y_bm = np.asarray(y_coord[iy0:iy1])
        values = np.asarray(data[name][iy0:iy1, ix0:ix1])

    # RegularGridInterpolator requires ascending coordinates
    if x_bm[0] > x_bm[-1]:
        x_bm = x_bm[::-1]
        values = values[:, ::-1]

    if y_bm[0] > y_bm[-1]:
        y_bm = y_bm[::-1]
        values = values[::-1, :]

    interpolator = RegularGridInterpolator(
        (y_bm, x_bm),
        values,
        method=method,
        bounds_error=False,
        fill_value=np.nan
    )

    points = np.column_stack(
        [y_query.ravel(), x_query.ravel()]
    )

    out = interpolator(points)

    return out.reshape(x_query.shape)

def subdomainmask(subdomain, input_file, name='mask', threshold=0):
    """ Check if there is ice in the subdomain by interpolating the BedMachine data

    Args:
        subdomain (tuple): (xmin, xmax, ymin, ymax) of the subdomain
        input_file (str): path to the input BedMachine .nc file
        name (str, optional): variable name to interpolate. Defaults to 'mask'.
        threshold (float, optional): threshold value of the mask

        For many subdomains, use find_subdomains_with_mask() instead.
    Returns:
        bool: true if there is any mask>0 in the subdomain, false otherwise
    """

    x0, x1, y0, y1 = subdomain

    with Dataset(input_file, "r") as data:
        x = np.asarray(data["x"][:])
        y = np.asarray(data["y"][:])

        ix0, ix1 = coord_window_indices(x, x0, x1)
        iy0, iy1 = coord_window_indices(y, y0, y1)

        if ix0 >= ix1 or iy0 >= iy1:
            return False

        mask = np.ma.asarray(data[name][iy0:iy1, ix0:ix1])
        return bool(np.ma.filled(mask > threshold, False).any())

def find_subdomains_with_mask(
    input_file,
    xmin,
    ymin,
    Lx,
    Ly,
    dx,
    dy,
    name="mask",
    threshold=0
):
    """
    Find all dx-by-dy subdomains that contain any BedMachine mask value > threshold.

    This is much faster than calling subdomainmask() for every subdomain because:
    1. The NetCDF file is opened only once.
    2. No scipy.griddata interpolation is used.
    3. The mask field is scanned in chunks.

    Returns:
        indlist (list[tuple[int, int]]): List of (xid, yid) subdomain indices where mask > threshold.
        Nx (int): Number of subdomains in x.
        Ny (int): Number of subdomains in y.
    """
    Nx = int(Lx // dx)
    Ny = int(Ly // dy)

    xmax = xmin + Nx * dx
    ymax = ymin + Ny * dy

    found = set()

    with Dataset(input_file, "r") as data:
        x = np.asarray(data["x"][:])
        y = np.asarray(data["y"][:])

        ix0, ix1 = coord_window_indices(x, xmin, xmax)
        iy0, iy1 = coord_window_indices(y, ymin, ymax)

        x_win = x[ix0:ix1]
        y_win = y[iy0:iy1]

        mask = np.ma.asarray(data[name][iy0:iy1, ix0:ix1])
        ice_bool = np.ma.filled(mask > threshold, False)

        rows, cols = np.where(ice_bool)

        xs = x_win[cols]
        ys = y_win[rows]

        xid = np.floor((xs - xmin) / dx).astype(int)
        yid = np.floor((ys - ymin) / dy).astype(int)

        valid = (
            (xid >= 0)
            & (xid < Nx)
            & (yid >= 0)
            & (yid < Ny)
        )
        print(np.shape(valid))

        found.update(zip(xid[valid].tolist(), yid[valid].tolist()))

    indlist = sorted(found)

    return indlist, Nx, Ny
