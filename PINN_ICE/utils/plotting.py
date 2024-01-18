import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

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

def plot_solutions():
    pass

def plot_history():
    pass
