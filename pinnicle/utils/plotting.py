import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib as mpl
import deepxde.backend as bkd
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata
from scipy.spatial import cKDTree as KDTree
import scipy.io as sio
import pandas as pd

def cmap_Rignot():
    """ colormap from ISSM
    """
    alpha = 1
    cmap = np.array((np.linspace(0, 1, 128, False), np.ones(128, ), np.ones(128, ))).T
    cmap[:, 1] = np.maximum(np.minimum((0.1 + cmap[:, 0]**(1 / alpha)), 1), 0)
    cmap = mpl.colors.hsv_to_rgb(cmap)
    # construct a colormap object from an array of shape (n, 3 / 4)
    cmap = ListedColormap(cmap)
    return cmap


def plot_solutions(pinn, path="", filename="2Dsolution.png", X_ref=None, sol_ref=None, cols=None, resolution=200, absvariable=[], default_time=None, **kwargs):
    """ plot model predictions

    Args:
        path (Path, str): Path to save the figures
        filename (str): name to save the figures, if set to None, then the figure will not be saved
        X_ref (dict): Coordinates of the reference solutions, if None, then just plot the predicted solutions
        u_ref (dict): Reference solutions, if None, then just plot the predicted solutions
        cols (int): Number of columns of subplot
        resolution (int): Number of grid points per row/column for plotting
        absvariable (list): Names of variables in the predictions that will need to take abs() before comparison
    """
    # generate Cartisian grid of X, Y
    # currently only work on 2D static and time dependent slice
    # TODO: add 1D plot
    if pinn.domain.geometry.dim >= 2:
    # generate 200x200 mesh on the domain
        X, Y = np.meshgrid(np.linspace(pinn.params.nn.input_lb[0], pinn.params.nn.input_ub[0], resolution),
                np.linspace(pinn.params.nn.input_lb[1], pinn.params.nn.input_ub[1], resolution))
        X_nn = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))

        if pinn.nn.parameters.input_size == 3:
            if default_time is None:
                default_time = pinn.params.domain.start_time
            X_nn = np.hstack((X_nn, np.ones([X_nn.shape[0],1])*default_time))

        grid_size = 2.0*(((pinn.params.nn.input_ub[0] - pinn.params.nn.input_lb[0])/resolution)**2+
                         ((pinn.params.nn.input_ub[1] - pinn.params.nn.input_lb[1])/resolution)**2)**0.5
        if bkd.backend_name == "pytorch":
            grid_size = bkd.to_numpy(grid_size)

        # predicted solutions
        sol_pred = pinn.model.predict(X_nn)
        plot_data = {k+"_pred":np.reshape(sol_pred[:,i:i+1], X.shape) for i,k in enumerate(pinn.params.nn.output_variables)}
        vranges = {k+"_pred":[pinn.params.nn.output_lb[i], pinn.params.nn.output_ub[i]] for i,k in enumerate(pinn.params.nn.output_variables)}
        # take abs
        for k in absvariable:
            plot_data[k+"_pred"] = np.abs( plot_data[k+"_pred"])

        # if ref solution is provided
        if (sol_ref is not None) and (X_ref is not None):
            # convert X_ref to np.narray if it is a dict
            if isinstance(X_ref, dict):
                X_ref = np.hstack((X_ref['x'].flatten()[:,None],X_ref['y'].flatten()[:,None]))

            if isinstance(sol_ref, np.ndarray):
                plot_data.update({k+"_ref":griddata(X_ref, sol_ref[:,i].flatten(), (X, Y), method='cubic') for i,k in enumerate(pinn.params.nn.output_variables)})
            elif isinstance(sol_ref, dict):
                plot_data.update({k+"_ref":griddata(X_ref, sol_ref[k].flatten(), (X, Y), method='cubic') for k in pinn.params.nn.output_variables if k in sol_ref})
            else:
                raise TypeError(f"Type of sol_ref ({type(sol_ref)}) is not supported ")

            vranges.update({k+"_ref":vranges[k+"_pred"] for k in pinn.params.nn.output_variables})
            plot_data.update({k+"_diff":(plot_data[k+"_pred"] - plot_data[k+"_ref"]) for k in pinn.params.nn.output_variables if k in sol_ref})
            vranges.update({k+"_diff":[-0.1*max(np.abs(vranges[k+"_pred"])), 0.1*max(np.abs(vranges[k+"_pred"]))] for k in pinn.params.nn.output_variables if k in sol_ref})

        # set ice mask
        X_mask = pinn.model_data.get_ice_coordinates()
        tree = KDTree(X_mask)
        dist, _ = tree.query(np.c_[X.ravel(), Y.ravel()], k=1)
        dist = dist.reshape(X.shape)
        for k in plot_data:
            plot_data[k][dist > grid_size] = np.nan

        # plot
        n = len(plot_data)
        if cols is None:
            cols = len(pinn.params.nn.output_variables)

        fig, axs = plt.subplots(math.ceil(n/cols), cols, figsize=(16,12))
        for ax,name in zip(axs.ravel(), plot_data.keys()):
            vr = vranges.setdefault(name, [None, None])
            im = ax.imshow(plot_data[name], interpolation='nearest', cmap='rainbow',
                    extent=[X.min(), X.max(), Y.min(), Y.max()],
                    vmin=vr[0], vmax=vr[1],
                    origin='lower', aspect='auto', **kwargs)
            ax.set_title(name)

            fig.colorbar(im, ax=ax, shrink=0.8)

        if filename:
            plt.savefig(path+filename)

    else:
        raise ValueError("Plot is only implemented for 2D/3D problem")


def plot_nn(pinn, data_names=None, X_mask=None, axs=None, vranges={}, resolution=200, **kwargs):
    """ plot the prediction of the nerual network in pinn, according to the data_names

    Args:
        pinn (class PINN): The PINN model
        data_names (list): List of data names
        X_mask (np.array): xy-coordinates of the ice mask
        axs (array of AxesSubplot): axes to plot each data, if not given, then generate a subplot according to the size of data_names
        vranges (dict): range of the data
        resolution (int): number of pixels in horizontal and vertical direction
    return:
        X (np.array): x-coordinates of the 2D plot
        Y (np.array): y-coordinates of the 2D plot
        im_data (dict): Dict of data for the 2D plot, each element has the same size as X and Y
        axs (array of AxesSubplot): axes of the subplots
    """
    nn_params = pinn.params.nn
    # if not given, use the output list in pinn
    if not data_names:
        data_names = nn_params.output_variables

    ndata = len(data_names)
            
    #  generate 2d Cartisian grid
    X, Y = np.meshgrid(np.linspace(nn_params.input_lb[0], nn_params.input_ub[0], resolution), 
                       np.linspace(nn_params.input_lb[1], nn_params.input_ub[1], resolution))
    X_nn = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))

    grid_size = 2.0*(((nn_params.input_ub[0] - nn_params.input_lb[0])/resolution)**2+
                     ((nn_params.input_ub[1] - nn_params.input_lb[1])/resolution)**2)**0.5
    if bkd.backend_name == "pytorch":
        grid_size = bkd.to_numpy(grid_size)
    
    # ice mask coordinates
    if not X_mask:
        X_mask = pinn.model_data.get_ice_coordinates()
    tree = KDTree(X_mask)
    dist, _ = tree.query(np.c_[X.ravel(), Y.ravel()], k=1)
    dist = dist.reshape(X.shape)

    # get the predictions
    sol_pred = pinn.model.predict(X_nn)
    im_data = {k: np.reshape(sol_pred[:,i:i+1], X.shape) 
               for i,k in enumerate(nn_params.output_variables) if k in data_names}
    if not vranges:
        vranges = {k: [nn_params.output_lb[i], nn_params.output_ub[i]] 
                   for i,k in enumerate(nn_params.output_variables) if k in data_names}

    # set masked area to nan
    for k in im_data:
        im_data[k][dist > grid_size] = np.nan

    #plot
    axs = plot_data(X, Y, im_data=im_data, axs=axs, vranges=vranges, **kwargs)
    return X, Y, im_data, axs

def plot_dict_data(X_dict, data_dict, axs=None, vranges={}, resolution=200, **kwargs):
    """ plot the data in data_dict, with coordinates in X_dict

    Args:
        X_dict (dict): Dict of the coordinates, with keys 'x', 'y'
        data_dict (dict): Dict of data
        axs (array of AxesSubplot): axes to plot each data, if not given, then generate a subplot according to the size of data_names
        vranges (dict): range of the data
        resolution (int): number of pixels in horizontal and vertical direction
    return:
        X (np.array): x-coordinates of the 2D plot
        Y (np.array): y-coordinates of the 2D plot
        im_data (dict): Dict of data for the 2D plot, each element has the same size as X and Y
        axs (array of AxesSubplot): axes of the subplots
    """
    data_names = list(data_dict.keys())
    ndata = len(data_names)
    
            
    #  generate 2d Cartisian grid
    X, Y = np.meshgrid(np.linspace(min(X_dict['x']), max(X_dict['x']), resolution),
            np.linspace(min(X_dict['y']), max(X_dict['y']), resolution))
    grid_size = 2.0*(((max(X_dict['x']) - min(X_dict['x']))/resolution)**2+
                     ((max(X_dict['y']) - min(X_dict['y']))/resolution)**2)**0.5
    
    # combine x,y coordinates of the data
    X_ref = np.hstack((X_dict['x'].flatten()[:,None], X_dict['y'].flatten()[:,None]))
    
    tree = KDTree(X_ref)
    dist, _ = tree.query(np.c_[X.ravel(), Y.ravel()], k=1)
    dist = dist.reshape(X.shape)
    
    # project data_dict to the 2d grid
    im_data = {}
    for k in data_names:
        temp = griddata(X_ref, data_dict[k].flatten(), (X, Y), method='cubic')
        temp[dist > grid_size] = np.nan
        im_data[k] = temp

    #plot
    axs = plot_data(X, Y, im_data=im_data, axs=axs, vranges=vranges, **kwargs)
    return X, Y, im_data, axs
    
def plot_data(X, Y, im_data, axs=None, vranges={}, **kwargs):
    """ plot all the data in im_data

    Args:
        X (np.array): x-coordinates of the 2D plot
        Y (np.array): y-coordinates of the 2D plot
        im_data (dict): Dict of data for the 2D plot, each element has the same size as X and Y
        axs (array of AxesSubplot): axes to plot each data, if not given, then generate a subplot according to the size of data_names
        vranges (dict): range of the data
    return:
        axs (array of AxesSubplot): axes of the subplots
    """
    # number of data 
    ndata = len(im_data)
    # data names is the keys
    data_names = list(im_data.keys())
    # generate axes array, if not provided
    if axs is None:
        fig, axs = plt.subplots(1, ndata, figsize=(16,4))

    # plot
    for i in range(min(len(axs), ndata)):
        name = data_names[i]
        vr = vranges.setdefault(name, [None, None])
        im = axs[i].imshow(im_data[name], interpolation='nearest', cmap='rainbow',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            vmin=vr[0], vmax=vr[1],
            origin='lower', aspect='auto', **kwargs)
        axs[i].set_title(name)
        plt.colorbar(im, ax=axs[i], shrink=0.8)
    
    return axs
   
def diffplot(pinn, feature, feat_title=None, mdata='ISSM', sim='mae', figsize=(15, 4), cmap='jet', scale=1, clim=None, cbar_bins=10, elements=None):
    """
    """
    # init figure
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    if feat_title==None:
        if type(feature)==list:
            raise TypeError('feat_title must be provided')
        else:
            feat_title=feature

    # input and output names
    input_names = pinn.nn.parameters.input_variables
    output_names = pinn.nn.parameters.output_variables

    # inputs
    X_ref = pinn.model_data.data[mdata].X_dict
    xref = X_ref[input_names[0]].flatten()[:,None]
    for i in range(1, len(input_names)):
        xref = np.hstack((xref, X_ref[input_names[i]].flatten()[:,None]))
    meshx = np.squeeze(xref[:, 0])
    meshy = np.squeeze(xref[:, 1])

    # predictions
    pred = pinn.model.predict(xref)
    pred_sol = np.zeros_like(np.squeeze(pred[:, 0:1]))
    # reference solution
    if type(feature)==list:
        for feat in feature:
            if feat not in pinn.model_data.data[mdata].data_dict.keys():
                raise KeyError('feature not provided as input, reference solution cannot be plotted')
            else:
                continue
    else:
        if feature not in pinn.model_data.data[mdata].data_dict.keys():
            raise KeyError('feature not provided as input, reference solution cannot be plotted')
    X_sol = pinn.model_data.data[mdata].data_dict
    sol = X_sol[output_names[0]].flatten()[:,None] # initializing array
    for i in range(1, len(output_names)):
        sol = np.hstack((sol, X_sol[output_names[i]].flatten()[:,None]))
    ref_sol = np.zeros_like(np.squeeze(sol[:, 0:1]))
    
    # grab features
    if type(feature)==list:
        for feat in feature:
            fid = output_names.index(feat)
            pred_sol += (np.squeeze(pred[:, fid:fid+1]*scale))**2
            ref_sol += (np.squeeze(sol[:, fid:fid+1]*scale))**2
        pred_sol = np.sqrt(pred_sol)
        ref_sol = np.sqrt(ref_sol)
    else:
        fid = output_names.index(feature)
        ref_sol = np.squeeze(sol[:, fid:fid+1]*scale)
        pred_sol = np.squeeze(pred[:, fid:fid+1]*scale)

    # triangles / elements
    if elements==None:
        if pinn.model_data.data[mdata].mesh_dict == {}:
            triangles = mpl.tri.Triangulation(meshx, meshy)
        else:
            if pinn.params.param_dict['data'][mdata]['data_path'].endswith('mat'):
                elements = pinn.model_data.data[mdata].mesh_dict['elements']-1
            else:
                elements = pinn.model_data.data[mdata].mesh_dict['elements']
            triangles = mpl.tri.Triangulation(meshx, meshy, elements)
    else:
        triangles = elements
        if len(triangles)!=len(meshx):
            raise ValueError('number of elements must equal number of (x, y) inputs')

    if clim==None:
        [cmin, cmax] = [np.min(np.append(ref_sol, pred_sol)), np.max(np.append(ref_sol, pred_sol))]
    else:
        [cmin, cmax] = clim
    data_list = [ref_sol, pred_sol]
    title_list = [feat_title + r"$_{ref}$", feat_title + r"$_{pred}$"]

    # looping through the plot
    for c in range(3):
        if c==2:
            if sim.upper()=='MAE':
                diff = np.abs(ref_sol-pred_sol)
                diff_val = np.round(np.mean(diff), 2)
                title = r"|"+feat_title+r"$_{ref} - $"+feat_title+r"$_{pred}$|, MAE="+str(diff_val)
            elif sim.upper()=='MSE':
                diff = (ref_sol-pred_sol)**2
                diff_val = np.round(np.mean(diff), 2)
                title = r"$($"+feat_title+r"$_{ref} - $"+feat_title+r"$_{pred})^2$, MSE="+str(diff_val)
            elif sim.upper()=='RMSE':
                diff = (ref_sol-pred_sol)**2
                diff_val = np.round(np.sqrt(np.mean(diff)), 2)
                diff = np.sqrt(diff)
                title = r"$(($"+feat_title+r"$_{ref} - $"+feat_title+r"$_{pred})^2)^{1/2}$, RMSE="+str(diff_val)
            elif sim.upper()=='SIMPLE':
                diff = ref_sol-pred_sol
                diff_val = np.round(np.mean(diff), 2)
                title = feat_title+r"$_{ref} - $"+feat_title+r"$_{pred}$, AVG. DIFF="+str(diff_val)
            else:
                print('Default similarity MAE implemented.')
                diff = np.abs(ref_sol-pred_sol)
                diff_val = np.round(np.mean(diff), 2)
                title = r"|"+feat_title+r"$_{ref} - $"+feat_title+r"$_{pred}$|, MAE="+str(diff_val)
            diff_map = np.squeeze(diff)
            clim = np.max([np.abs(np.min(diff)*0.9), np.abs(np.max(diff)*1.1)])
            axes = axs[c].tripcolor(triangles, diff_map, cmap='RdBu', norm=colors.Normalize(vmin=-1*clim, vmax=clim))
        else:
            axes = axs[c].tripcolor(triangles, data_list[c], cmap=cmap, norm=colors.Normalize(vmin=cmin, vmax=cmax))
            title = title_list[c]

        # common settings
        axs[c].set_title(title, fontsize=14)
        axs[c].axis('off')
        cb = plt.colorbar(axes, ax=axs[c])
        cb.ax.tick_params(labelsize=12)
        cb.locator = ticker.MaxNLocator(nbins=cbar_bins)
        cb.update_ticks()
    
    return fig, axs

def resplot(pinn, mdata='ISSM', figsize=None, cmap='RdBu', cbar_bins=10, cbar_limits=[-5e3, 5e3], elements=None):
    """plotting the pde residuals
    """
    input_names = pinn.nn.parameters.input_variables
    output_names = pinn.nn.parameters.output_variables

    # inputs
    X_ref = pinn.model_data.data[mdata].X_dict
    xref = X_ref[input_names[0]].flatten()[:, None]
    for i in range(1, len(input_names)):
        xref = np.hstack((xref, X_ref[input_names[i]].flatten()[:,None]))
    meshx = np.squeeze(xref[:, 0])
    meshy = np.squeeze(xref[:, 1])

    # triangles / elements
    if elements==None:
        if pinn.model_data.data[mdata].mesh_dict == {}:
            triangles = mpl.tri.Triangulation(meshx, meshy)
        else:
            if pinn.params.param_dict['data'][mdata]['data_path'].endswith('mat'):
                elements = pinn.model_data.data[mdata].mesh_dict['elements']-1
            else:
                elements = pinn.model_data.data[mdata].mesh_dict['elements']
            triangles = mpl.tri.Triangulation(meshx, meshy, elements)
    else:
        triangles = elements
        if len(triangles) != len(meshx):
            raise ValueError('number of elements must equal number of (x, y) inputs')

    Nr = len(pinn.physics.residuals)
    fig, axs = plt.subplots(1, len(pinn.physics.residuals), figsize=(5*Nr, 4))

    pde_dict = {} # counting the number of residuals per pde
    for i in pinn.params.physics.equations.keys():
        pde_dict[i] = 0

    for r in range(Nr):
        # looping through the equation keys
        for p in pinn.params.physics.equations.keys():
            if p in pinn.physics.residuals[r]:
                pde_dict[p] += 1
                pde_pred = pinn.model.predict(xref, operator=pinn.physics.operator(p))

                op_pred = pde_pred[pde_dict[p]-1]
                if Nr <= 1:
                    axes = axs.tripcolor(triangles, np.squeeze(op_pred), cmap=cmap, norm=colors.CenteredNorm(clip=[cbar_limits[0], cbar_limits[-1]]))
                    cb = plt.colorbar(axes, ax=axs)
                    cb.ax.tick_params(labelsize=14)
                    # adjusting the number of ticks
                    num_bins = ticker.MaxNLocator(nbins=cbar_bins)
                    cb.locator = num_bins
                    cb.update_ticks()
                    # setting the title
                    axs.set_title(str(pinn.physics.residuals[r]), fontsize=14)
                    axs.axis('off')
                else:
                    axes = axs[r].tripcolor(triangles, np.squeeze(op_pred), cmap=cmap, norm=colors.CenteredNorm(clip=[cbar_limits[0], cbar_limits[-1]]))
                    cb = plt.colorbar(axes, ax=axs[r])
                    cb.ax.tick_params(labelsize=14)
                    # adjusting the number of ticks
                    num_bins = ticker.MaxNLocator(nbins=cbar_bins)
                    cb.locator = num_bins
                    cb.update_ticks()
                    # title
                    axs[r].set_title(str(pinn.physics.residuals[r]), fontsize=14)
                    axs[r].axis('off')

    return fig, axs

def tripcolor_similarity(pinn, feature_name, feat_title=None, mdata='ISSM', sim='MAE', cmap='jet', scale=1, colorbar_bins=10, elements=None):
    """tripcolor similarity, plot with ISSM triangulation
    """

    if feat_title==None:
        if type(feature_name)==list:
            raise TypeError('feat_title must be provided as input str')
        else:
            feat_title = feature_name

    # initialize figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # neural network features 
    input_names = pinn.nn.parameters.input_variables
    output_names = pinn.nn.parameters.output_variables

    # inputs
    X_ref = pinn.model_data.data[mdata].X_dict
    xref = X_ref[input_names[0]].flatten()[:,None]
    for i in range(1, len(input_names)):
        xref = np.hstack((xref, X_ref[input_names[i]].flatten()[:,None]))
    meshx = np.squeeze(xref[:, 0])
    meshy = np.squeeze(xref[:, 1])

    # predictions
    pred = pinn.model.predict(xref)

    # reference solution
    X_sol = pinn.model_data.data[mdata].data_dict
    sol = X_sol[output_names[0]].flatten()[:,None]
    for i in range(1, len(output_names)):
        sol = np.hstack((sol, X_sol[output_names[i]].flatten()[:,None]))

    # elements, if any
    if elements!=None:
        triangles = elements
        if len(triangles)!=len(meshx):
            raise ValueError('number of elements must equal number number of vertices (x, y)')
    else:
        if pinn.model_data.data[mdata].mesh_dict == {}:
            triangles = mpl.tri.Triangulation(meshx, meshy)
        else:
            if pinn.params.param_dict['data'][mdata]['data_path'].endswith('.mat'):
                elements = pinn.model_data.data[mdata].mesh_dict['elements']-1
            else:
                elements = pinn.model_data.data[mdata].mesh_dict['elements']
            triangles = mpl.tri.Triangulation(meshx, meshy, elements)

    # grab feature
    # initializing ref and pred
    ref_sol = np.zeros_like(np.squeeze(sol[:, 0:1]*scale))
    pred_sol = np.zeros_like(np.squeeze(pred[:, 0:1]*scale))
    if type(feature_name) == list:
        for feat in feature_name:
            fid = output_names.index(feat)
            ref_sol += (np.squeeze(sol[:, fid:fid+1]*scale))**2
            pred_sol += (np.squeeze(pred[:, fid:fid+1]*scale))**2
        ref_sol = np.sqrt(ref_sol)
        pred_sol = np.sqrt(pred_sol)
    else:
        fid = output_names.index(feature_name)
        ref_sol = np.squeeze(sol[:, fid:fid+1]*scale)
        pred_sol = np.squeeze(pred[:, fid:fid+1]*scale)

    [cmin, cmax] = [0.9*np.min(np.append(ref_sol, pred_sol)), 1.1*np.max(np.append(ref_sol, pred_sol))]
    data_list = [ref_sol, pred_sol]
    title_list = [ feat_title + r"$_{ref}$", feat_title + r"$_{pred}$"]

    # looping through the columns of the plot
    for c in range(3):
        if c == 2:
            if sim.upper() == 'MAE':
                diff = np.abs(ref_sol-pred_sol)
                diff_val = np.round(np.mean(diff), 2)
                title = r"|"+feat_title+r"$_{ref} - $"+feat_title+r"$_{pred}$|, MAE="+str(diff_val)
            elif sim.upper() == 'MSE':
                diff = (ref_sol-pred_sol)**2
                diff_val = np.round(np.mean(diff), 2)
                title = r"$($"+feat_title+r"$_{ref} - $"+feat_title+r"$_{pred})^2$, MSE="+str(diff_val)
            elif sim.upper() == 'RMSE':
                diff = (ref_sol-pred_sol)**2
                diff_val = np.round(np.sqrt(np.mean(diff)), 2)
                diff = np.sqrt(diff)
                title = r"$(($"+feat_title+r"$_{ref} - $"+feat_title+r"$_{pred})^2)^{1/2}$, RMSE="+str(diff_val)
            elif sim.upper() == 'SIMPLE':
                diff = ref_sol-pred_sol
                diff_val = np.round(np.mean(diff), 2)
                title = feat_title+r"$_{ref} - $"+feat_title+r"$_{pred}$, AVG. DIFF="+str(diff_val)
            else:
                print('Default similarity MAE implemented.')
                diff = np.abs(ref_sol-pred_sol)
                diff_val = np.round(np.mean(diff), 2)
                title = r"|"+feat_title+r"$_{ref} - $"+feat_title+r"$_{pred}$|, MAE="+str(diff_val)

            diff_map = np.squeeze(diff)
            clim = np.max([np.abs(np.min(diff)*0.9), np.abs(np.max(diff)*1.1)])
            axes = axs[c].tripcolor(triangles, diff_map, cmap='RdBu', norm=colors.Normalize(vmin=-1*clim, vmax=clim))
        else:
            axes = axs[c].tripcolor(triangles, data_list[c], cmap=cmap, norm=colors.Normalize(vmin=cmin, vmax=cmax))
            title = title_list[c]

        # common settings
        axs[c].set_title(title, fontsize=14)
        axs[c].axis('off')
        cb = plt.colorbar(axes, ax=axs[c])
        cb.ax.tick_params(labelsize=12)
        cb.locator = ticker.MaxNLocator(nbins=colorbar_bins)
        cb.update_ticks()

    return fig, axs

def tripcolor_residuals(pinn, cmap='RdBu', colorbar_bins=10, cbar_limits=[-5e3, 5e3]):
    """plot pde residuals with ISSM triangulation
    """
    input_names = pinn.nn.parameters.input_variables
    output_names = pinn.nn.parameters.output_variables

    # inputs
    X_ref = pinn.model_data.data['ISSM'].X_dict
    xref = X_ref[input_names[0]].flatten()[:,None]
    for i in range(1, len(input_names)):
        xref = np.hstack((xref, X_ref[input_names[i]].flatten()[:,None]))
    meshx = np.squeeze(xref[:, 0])
    meshy = np.squeeze(xref[:, 1])

    # grabbing ISSM elements/triangles
    elements = pinn.model_data.data['ISSM'].mesh_dict['elements']-1
    triangles = mpl.tri.Triangulation(meshx, meshy, elements)

    Nr = len(pinn.physics.residuals)
    fig, axs = plt.subplots(1, len(pinn.physics.residuals), figsize=(5*Nr, 4))

    pde_dict = {} # counting the number of residuals per pde
    for i in pinn.params.physics.equations.keys():
        pde_dict[i] = 0

    for r in range(Nr):
        # looping through the equations keys
        for p in pinn.params.physics.equations.keys():
            if p in pinn.physics.residuals[r]:
                pde_dict[p] += 1
                pde_pred = pinn.model.predict(xref, operator=pinn.physics.operator(p))

                op_pred = pde_pred[pde_dict[p]-1]
                if Nr <= 1:
                    axes = axs.tripcolor(triangles, np.squeeze(op_pred), cmap=cmap, norm=colors.CenteredNorm(clip=[cbar_limits[0], cbar_limits[-1]]))
                    cb = plt.colorbar(axes, ax=axs)
                    cb.ax.tick_params(labelsize=14)
                    # adjusting the number of ticks
                    num_bins = ticker.MaxNLocator(nbins=colorbar_bins)
                    cb.locator = num_bins
                    cb.update_ticks()
                    # setting the title
                    axs.set_title(str(pinn.physics.residuals[r]), fontsize=14)
                    axs.axis('off')
                else:
                    axes = axs[r].tripcolor(triangles, np.squeeze(op_pred), cmap=cmap, norm=colors.CenteredNorm(clip=[cbar_limits[0], cbar_limits[-1]]))
                    cb = plt.colorbar(axes, ax=axs[r])
                    cb.ax.tick_params(labelsize=14)
                    # adjusting the number of ticks
                    num_bins = ticker.MaxNLocator(nbins=colorbar_bins)
                    cb.locator = num_bins
                    cb.update_ticks()
                    # title
                    axs[r].set_title(str(pinn.physics.residuals[r]), fontsize=14)
                    axs[r].axis('off')

    return fig, axs

def plot_tracks(pinn, feature, filepath=None, x_name_map='x', y_name_map='y', feat_name_map=None, gdata=None, mdata='ISSM', elements=None, scale=1, cmap='jet', figsize=(10, 8), clim=None, cbar_bins=10):
    """plotting (sparse) ground truth data on top of prediction
    """
    if filepath == None and gdata == None:
        raise TypeError('gfilepath and gdata cannot both be None, one must be provided')

    if gdata != None:
        print('using gdata provided')
    else:
        print('using filepath')
        if feat_name_map==None:
            raise TypeError('feat_name_map must be provided')
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.mat'):
            df = sio.loadmat(filepath)
        else:
            raise NotImplementedError('use .csv or .mat file formats')
        xx = df[x_name_map]
        yy = df[y_name_map]
        gdata = df[feat_name_map]

    # initialize figure
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    
    # neural network features
    input_names = pinn.nn.parameters.input_variables
    output_names = pinn.nn.parameters.output_variables

    # inputs
    X_ref = pinn.model_data.data[mdata].X_dict
    xref = X_ref[input_names[0]].flatten()[:,None]
    for i in range(1, len(input_names)):
        xref = np.hstack((xref, X_ref[input_names[i]].flatten()[:,None]))
    meshx = np.squeeze(xref[:, 0])
    meshy = np.squeeze(xref[:, 1])

    # predictions
    pred = pinn.model.predict(xref)
    if type(feature)==list:
        raise TypeError('feature must be str')
    fid = output_names.index(feature)
    pred_sol = np.squeeze(pred[:, fid:fid+1]*scale)

    # triangles/elements
    if elements==None:
        if pinn.model_data.data[mdata].mesh_dict == {}:
            triangles = mpl.tri.Triangulation(meshx, meshy)
        else:
            if pinn.params.param_dict['data'][mdata]['data_path'].endswith('mat'):
                elements = pinn.model_data.data[mdata].mesh_dict['elements']-1
            else:
                elements = pinn.model_data.data[mdata].mesh_dict['elements']
            triangles = mpl.tri.Triangulation(meshx, meshy, elements)
    else:
        triangles = elements
        if len(triangles)!=len(meshx):
            raise ValueError('number of elements must equal number of (x, y) inputs')

    if clim==None:
        [cmin, cmax] = [np.min(pred_sol), np.max(pred_sol)]
    else:
        [cmin, cmax] = clim

    axes = axs.tripcolor(triangles, pred_sol, cmap=cmap, norm=colors.Normalize(vmin=cmin, vmax=cmax))
    axes = axs.scatter(xx, yy, s=2, c=gdata, cmap=cmap, vmin=cmin, vmax=cmax)
    title = feature+r"$_{pred}$ & sparse " +feature+r"$_{true}$"
    axs.set_title(title, fontsize=12)
    axs.axis('off')
    cb = plt.colorbar(axes, ax=axs)
    cb.ax.tick_params(labelsize=12)
    cb.locator = ticker.MaxNLocator(nbins=cbar_bins)
    cb.update_ticks()

    return fig, axs
