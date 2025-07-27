import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree
import deepxde.backend as bkd
import numpy as np

def plotmodel(pinn, path="", filename="", **kwargs):
    """ plotmodel
    """
    pass


def plot2d(axs, X, Y, data, mask=None, resolution=200, **kwargs):
    """ plot 2d scattered data, make a triangular mesh and plot the data on the mesh

    Args:
        axs (AxesSubplot): handler for plotting
        X (np.array): x-coordinates of the 2D plot
        Y (np.array): y-coordinates of the 2D plot
        data (np.array): data for the 2D plot, it has the same size as X and Y
    return:
        axs (AxesSubplot): axes of the subplots
    """
    # generate triagular mesh, the plot all the 1d-array data, no matter if it has a grid or not

    triangles = mpl.tri.Triangulation(X, Y)

    # TODO: add masks to enable concave shape of the domain
    if mask is not None:
        grid_size = 2.0*((max(X)-min(X))**2.0+(max(Y)-min(Y))**2.0)**0.5/resolution
        mask_coord = np.c_[X[mask].ravel(), Y[mask].ravel()]
        tree = KDTree(mask_coord)
        dist, _ = tree.query(np.c_[X.ravel(), Y.ravel()], k=1)
        dist = dist.reshape(X.shape)
        data[dist > grid_size] = np.nan

    axs = plottriangle(axs, triangles, data, **kwargs)

    return axs

def plottriangle(axs, triangles,  data, **kwargs):
    """ plot a triagular mesh

    Args:
        axs (AxesSubplot): handler for plotting
        triangles (ntri, 3): a triangluar mesh generated using mpl.tri.Triangulation
        data (np.array): data for the 2D plot, it has the same size as X and Y
    return:
        axs (AxesSubplot): axes of the subplots
    """
    axs = axs.tripcolor(triangles, data, **kwargs)

    return axs

def plotscatter(axs, X, Y, data, **kwargs):
    """ plot 2d data as scattered data

    Args:
        axs (AxesSubplot): handler for plotting
        X (np.array): x-coordinates of the 2D plot
        Y (np.array): y-coordinates of the 2D plot
        data (np.array): data for the 2D plot, it has the same size as X and Y
    return:
        axs (AxesSubplot): axes of the subplots
    """
    axs = axs.scatter(X, Y, s=1, c=data, **kwargs)

    return axs

# plot_prediction
# with mask 



