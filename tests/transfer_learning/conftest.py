import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DDE_BACKEND", "pytorch")

import torch
_ = torch.tensor([0.0])