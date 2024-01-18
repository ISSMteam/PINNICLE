from deepxde.model import LossHistory
import numpy as np
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
        self.history['steps'] = loss_history.steps

    def save(self, path, filename="history.json"):
        """ save training history 
        """
        save_dict_to_json(self.history, path, filename)
        
    
    def load(self, path, filename="history.json"):
        """ load training history from folder or path
        """
        self.history = load_dict_from_json(path, filename)
