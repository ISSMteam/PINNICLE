import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata

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

def plot_solutions(pinn, path="", X_ref=None, sol_ref=None, cols=None, resolution=200, **kwargs):
    """ plot model predictions
    Args:
        path (Path, str): Path to save the figures
        X_ref (dict): Coordinates of the reference solutions, if None, then just plot the predicted solutions
        u_ref (dict): Reference solutions, if None, then just plot the predicted solutions
        cols (int): Number of columns of subplot
        resolution (int): Number of grid points per row/column for plotting
    """
    # generate Cartisian grid of X, Y
    # currently only work on 2D
    # TODO: add 1D plot
    if pinn.domain.geometry.dim == 2:
    # generate 200x200 mesh on the domain
        X, Y = np.meshgrid(np.linspace(pinn.param.nn.input_lb[0], pinn.param.nn.input_ub[0], resolution),
                np.linspace(pinn.param.nn.input_lb[1], pinn.param.nn.input_ub[1], resolution))
        X_nn = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))

        # predicted solutions
        sol_pred = pinn.model.predict(X_nn)
        plot_data = {k+"_pred":np.reshape(sol_pred[:,i:i+1], X.shape) for i,k in enumerate(pinn.param.nn.output_variables)}
        vranges = {k+"_pred":[pinn.param.nn.output_lb[i], pinn.param.nn.output_ub[i]] for i,k in enumerate(pinn.param.nn.output_variables)}

        # if ref solution is provided
        if (sol_ref is not None) and (X_ref is not None):
            # convert X_ref to np.narray if it is a dict
            if isinstance(X_ref, dict):
                X_ref = np.hstack((X_ref['x'].flatten()[:,None],X_ref['y'].flatten()[:,None]))

            if isinstance(sol_ref, np.ndarray):
                plot_data.update({k+"_ref":griddata(X_ref, sol_ref[:,i].flatten(), (X, Y), method='cubic') for i,k in enumerate(pinn.param.nn.output_variables)})
            elif isinstance(sol_ref, dict):
                plot_data.update({k+"_ref":griddata(X_ref, sol_ref[k].flatten(), (X, Y), method='cubic') for k in pinn.param.nn.output_variables})
            else:
                raise TypeError(f"Type of sol_ref ({type(sol_ref)}) is not supported ")
            
            vranges.update({k+"_ref":vranges[k+"_pred"] for k in pinn.param.nn.output_variables})
            plot_data.update({k+"_diff":(plot_data[k+"_pred"] - plot_data[k+"_ref"]) for k in pinn.param.nn.output_variables})
            vranges.update({k+"_diff":[-0.1*max(np.abs(vranges[k+"_pred"])), 0.1*max(np.abs(vranges[k+"_pred"]))] for k in pinn.param.nn.output_variables})

        # plot
        n = len(plot_data)
        if cols is None:
            cols = len(pinn.param.nn.output_variables)

        fig, axs = plt.subplots(math.ceil(n/cols), cols, figsize=(16,12))
        for ax,name in zip(axs.ravel(), plot_data.keys()):
            vr = vranges.setdefault(name, [None, None])
            im = ax.imshow(plot_data[name], interpolation='nearest', cmap='rainbow',
                    extent=[X.min(), X.max(), Y.min(), Y.max()],
                    vmin=vr[0], vmax=vr[1],
                    origin='lower', aspect='auto', **kwargs)
            ax.set_title(name)

            fig.colorbar(im, ax=ax, shrink=0.8)

        plt.savefig(path+"2Dsolution.png")

    else:
        raise ValueError("Plot is only implemented for 2D problem")
