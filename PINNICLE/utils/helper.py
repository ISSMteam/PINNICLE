import json
import os
import mat73
import scipy.io

def is_file_ext(path, ext):
    """ check if a given path is ended by ext
    """
    if isinstance(path, str):
        if path.endswith(ext):
            return True
    return False

def save_dict_to_json(data, path, filename):
    """ Save the dict to the path as a .json file
    Args:
        data (dict): Dictionary data to save
        path (Path, str): Path to save
        filename (str): name to save, attach '.json' if needed
    """
    if not filename.endswith('.json'):
        filename += '.json'
    with open(os.path.join(path,filename), "w") as fp:
        json.dump(data, fp)

def load_dict_from_json(path, filename):
    """ Load a dict data from a .json file 
    Args:
        path (Path, str): Path to load
        filename (str): name to load, attach '.json' if needed
    Returns:
        data (dict): Dictionary data to load
    """
    if not filename.endswith('.json'):
        filename += '.json'
    f = os.path.join(path, filename)
    if os.path.isfile(f):
        with open(f, 'r') as fp:
            data = json.load(fp)
    else:
        data = {}
    return data

def load_mat(file):
    """ load .mat file, if the file is in MATLAB 7.3 format use mat73.loadmat, otherwise use scipy.io.loadmat()
    """
    try:
        data = mat73.loadmat(file)
    except TypeError:
        data = scipy.io.loadmat(file)
    return data
