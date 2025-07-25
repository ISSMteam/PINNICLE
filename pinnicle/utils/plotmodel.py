import matplotlib as mpl
import matplotlib.pyplot as plt
import deepxde.backend as bkd

def plotmodel(pinn, path="", filename="", **kwargs):
    """ plotmodel
    """
    pass


def plot2d(axs, X, Y, data, **kwargs):
    """ plot 2d scattered data, make a triangular mesh and plot the data on the mesh

    Args:
        axs (AxesSubplot): handler for plotting
        X (np.array): x-coordinates of the 2D plot
        Y (np.array): y-coordinates of the 2D plot
        data (np.array): data for the 2D plot, it has the same size as X and Y
    return:
        axs (AxesSubplot): axes of the subplots
    """
    # TODO: add masks to enable concave shape of the domain
    # generate triagular mesh, the plot all the 1d-array data, no matter if it has a grid or not

    triangles = mpl.tri.Triangulation(X, Y)
    im = axs.tripcolor(triangles, data, **kwargs)

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



