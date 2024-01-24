import PINN_ICE as pinn
import numpy as np
import pytest
import deepxde as dde
import os

def test_update_cid():

    expFileName = "fastflow_CF.exp"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    hp = {}
    hp["shapefile"] = os.path.join(repoPath, "dataset", expFileName)

    p = pinn.parameter.DomainParameter()
    # Check exceptions
    with pytest.raises(Exception):
        d = pinn.domain.Domain(p)

    p2 = pinn.parameter.DomainParameter(hp)
    d2 = pinn.domain.Domain(p2)
    assert type(d2.geometry) == dde.geometry.Polygon
