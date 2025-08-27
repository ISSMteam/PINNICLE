import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree
import deepxde.backend as bkd
import numpy as np

def plotmodel(pinn, path="", filename="", **kwargs):
    """ plotmodel
    """
    pass

def plotprediction(axs, model, X, Y, key, scaling=1, **kwargs):
    """ plot predictions of the keys from the pinn model

    Args:
        axs (AxesSubplot): handler for plotting
        model (pinnicle.pinn): PINNICLE model
        X (np.array): x-coordinates of the 2D plot
        Y (np.array): y-coordinates of the 2D plot
        key (str): key of the output variable
        scaling (float, optional): scaling factor for the data. Defaults to 1
    return:
        axs (AxesSubplot): axes of the subplots
    """
    # compute the prediction
    X_nn = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
    sol_pred = model.model.predict(X_nn)

    # get the index of the key
    keylist = model.params.nn.output_variables
    ind = keylist.index(key)

    # plot
    axs = plot2d(axs, X, Y, scaling*sol_pred[:,ind:ind+1].flatten(), **kwargs)
    return axs

def plotdiff(axs, model, X, Y, data, key, scaling=1, **kwargs):
    """ plot the difference between the prediction of the keys from the pinn model and the data

    Args:
        axs (AxesSubplot): handler for plotting
        model (pinnicle.pinn): PINNICLE model
        X (np.array): x-coordinates of the 2D plot
        Y (np.array): y-coordinates of the 2D plot
        data (np.array): data for the 2D plot, it has the same size as X and Y
        key (str): key of the output variable
        scaling (float, optional): scaling factor for the data. Defaults to 1
    return:
        axs (AxesSubplot): axes of the subplots
    """
    # compute the prediction
    X_nn = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
    sol_pred = model.model.predict(X_nn)

    # get the index of the key
    keylist = model.params.nn.output_variables
    ind = keylist.index(key)

    # plot
    axs = plot2d(axs, X, Y, scaling*(sol_pred[:,ind:ind+1].flatten()-data), **kwargs)
    return axs

def plot2d(axs, X, Y, data, mask=None, scaling=1, **kwargs):
    """ plot 2d scattered data, make a triangular mesh and plot the data on the mesh

    Args:
        axs (AxesSubplot): handler for plotting
        X (np.array): x-coordinates of the 2D plot
        Y (np.array): y-coordinates of the 2D plot
        data (np.array): data for the 2D plot, it has the same size as X and Y
        mask (np.array, optional): mask for the data, True for invalid data. Defaults to None
        scaling (float, optional): scaling factor for the data. Defaults to 1
    return:
        axs (AxesSubplot): axes of the subplots
    """
    # generate triagular mesh, the plot all the 1d-array data, no matter if it has a grid or not
    triangles = mpl.tri.Triangulation(X, Y)

    # apply the mask    
    if mask is None:
        if np.ma.is_masked(data):
            mask = np.isnan(data).filled(True)
        else:
            mask = np.isnan(data)
        
    data[mask] = np.nan
    
    # plot
    axs = plottriangle(axs, triangles, scaling*data, **kwargs)

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