import math
import numpy as np
import matplotlib.pyplot as plt
from deepxde.model import LossHistory
from . import save_dict_to_json, load_dict_from_json


class History:
    """ class of the training history, based on deepxde LossHistory
        only need steps and loss_train
    """
    def __init__(self, loss_history, names):
        super().__init__()
        self._loss_train = np.array(loss_history.loss_train)
        self._names = names
        
        # put history of each term is a dict
        self.history = {k:list(self._loss_train[:,i]) for i,k in enumerate(names)}
        self.history["steps"] = loss_history.steps

    def save(self, path, filename="history.json"):
        """ save training history 
        """
        save_dict_to_json(self.history, path, filename)
        
    def load(self, path, filename="history.json"):
        """ load training history from folder or path
        """
        self.history = load_dict_from_json(path, filename)

    def plot(self, path, figname="history.png", cols=4):
        """ plot the history 
        """
        # subtract "step"
        loss_keys = [k for k in self.history.keys() if k != "steps"]
        n = len(loss_keys)   

        fig, axs = plt.subplots(math.ceil(n/cols), cols, figsize=(16,12))

        for ax, name in zip(axs.ravel(), loss_keys):
            ax.plot((self.history[name]), label=name)
            ax.axes.set_yscale('log')
            ax.legend(loc="best")

        # if figname is set to nothing, then don't save the figure
        if figname != "":
            plt.savefig(path+figname)
