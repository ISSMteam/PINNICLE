import os

"""The test_transfer_learning.py requires PyTorch specifc behavior at import time.This is a pytest config file that must run before any test are imported. 
It needs to be located at the root repo, ensuring it executes before any module level imports.
Its purpose is to set up the environment early enough to avoid import-order issues 
that occur when __init__.py imports pinnicle (in turn loading DeepXDE and PyTorch). 
Specifically, it sets the environment variables DDE_BACKEND="pytorch" to force a consistent
backend, configures MPLBACKEND="Agg" for headless plotting, and initializes PyTorch early to
prevent DLL loading errors on Windows. 
"""

# OpenMP workaround 
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# use non-GUI backend for matplotlib (needed for tests)
os.environ.setdefault("MPLBACKEND", "Agg")

# force DeepXDE to use Pytorch backend  
os.environ.setdefault("DDE_BACKEND", "pytorch")

# initialize torch DLLs BEFORE pinnicle/deepxde import
import torch
_ = torch.tensor([0.0])  # forces full DLL initialization


# Pytest hooks 
def pytest_configure():
    """Ensures environment variables above are set before any tests execute
    """
    pass
