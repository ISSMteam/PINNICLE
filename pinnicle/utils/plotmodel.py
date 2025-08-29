import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree
import deepxde.backend as bkd
import numpy as np

def plotmodel(pinn, path="", filename="", **kwargs):
    """ plotmodel
    """
    pass

def plotmodelcompare(model, dataname, output, scaling=1, diffrange=None, iscatter=False, **kwargs):
    """ plot the comparison between the prediction of the keys from the pinn model and the data

    Args:
        model (pinnicle.pinn): PINNICLE model
        dataname (str): name of the data in model.model_data.data
        output (str): key of the output variable
        scaling (float, optional): scaling factor for the data. Defaults to 1
        diffrange (float, optional): range of the difference plot. If not provided, it will be set to the max value of the data. Defaults to None
        iscatter (bool, optional): if True, use scatter plot for the data plot. Defaults to False
    return:
        axs (AxesSubplot): axes of the subplots
    """
    figsize = kwargs.pop('figsize', (12,5))
    fig, axs = plt.subplots(1, 3 , figsize=figsize)

    X = model.model_data.data[dataname].X_dict['x'].flatten()
    Y = model.model_data.data[dataname].X_dict['y'].flatten()
    data = model.model_data.data[dataname].data_dict[output].flatten()

    # get the mask
    if np.ma.is_masked(data):
        mask = np.isnan(data).filled(True)
    else:
        mask = np.isnan(data)

    # plot ref data
    if iscatter:
        im = plotscatter(axs[0], X, Y, data*scaling, **kwargs)
    else:
        im = plot2d(axs[0], X, Y, data, scaling=scaling, **kwargs)

    axs[0].set_title(f'Data: {output}')
    fig.colorbar(im, ax=axs[0], shrink=0.8, location='top')

    # plot prediction
    if iscatter:
        im = plotprediction(axs[1], model, output, scaling=scaling, **kwargs)
    else:
        im = plotprediction(axs[1], model, output, X=X, Y=Y, mask=mask, scaling=scaling, **kwargs)
    axs[1].set_title('Prediction')
    fig.colorbar(im, ax=axs[1], shrink=0.8, location='top')

    # plot difference
    if diffrange is None:
        diffrange = np.nanmax(np.abs(data))*scaling

    im = plotdiff(axs[2], model, X, Y, data, output, scaling=scaling, cmap='bwr', vmin=-0.1*diffrange, vmax=0.1*diffrange, iscatter=iscatter) 
    axs[2].set_title('Difference')
    fig.colorbar(im, ax=axs[2], shrink=0.8, location='top')

    return axs

def plotprediction(axs, model, key, X=None, Y=None, scaling=1, resolution=200, operator=None, **kwargs):
    """ plot predictions of the keys from the pinn model

    Args:
        axs (AxesSubplot): handler for plotting
        model (pinnicle.pinn): PINNICLE model
        key (str): key of the output variable
        X (np.array, optional): x-coordinates of the 2D plot. If not provided, a grid will be generated based on the domain bbox. Defaults to None
        Y (np.array, optional): y-coordinates of the 2D plot. If not provided, a grid will be generated based on the domain bbox. Defaults to None
        scaling (float, optional): scaling factor for the data. Defaults to 1
        resolution (int, optional): resolution of the generated grid if X and Y are not provided. Defaults to 200
    return:
        axs (AxesSubplot): axes of the subplots
    """
    # prepare the grid, if not provided, generate a grid based on the domain bbox
    if X is None or Y is None:
        bbox = model.domain.bbox()
        x = np.linspace(bbox[0,0], bbox[1,0], resolution)
        y = np.linspace(bbox[0,1], bbox[1,1], resolution)
        X, Y = np.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
    
    # compute the prediction
    X_nn = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
    sol_pred = model.model.predict(X_nn)

    # get the index of the key
    keylist = model.params.nn.output_variables
    if key in keylist:
        ind = keylist.index(key)
        data = scaling*sol_pred[:,ind:ind+1].flatten()
    elif key == 'bed':
        ind_s = keylist.index('s')
        ind_H = keylist.index('H')
        s = scaling*sol_pred[:,ind_s:ind_s+1].flatten()
        H = scaling*sol_pred[:,ind_H:ind_H+1].flatten()
        data = s - H
    else:
        raise ValueError(f"Key {key} not found in model output variables and is not 'bed'.")
    
    # apply operator if provided
    if operator is not None:
        data = operator(data)
    
    # plot
    axs = plot2d(axs, X, Y, data, **kwargs)
    return axs

def plotdiff(axs, model, X, Y, data, key, scaling=1, iscatter=False, **kwargs):
    """ plot the difference between the prediction of the keys from the pinn model and the data

    Args:
        axs (AxesSubplot): handler for plotting
        model (pinnicle.pinn): PINNICLE model
        X (np.array): x-coordinates of the 2D plot
        Y (np.array): y-coordinates of the 2D plot
        data (np.array): data for the 2D plot, it has the same size as X and Y
        key (str): key of the output variable
        scaling (float, optional): scaling factor for the data. Defaults to 1
        iscatter (bool, optional): if True, use scatter plot for the data plot. Defaults to False
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
    if iscatter:
        axs = plotscatter(axs, X, Y, scaling*(sol_pred[:,ind:ind+1].flatten()-data), **kwargs)
    else:
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