import pinnicle as pinn
import pytest
import deepxde as dde
import os


def test_domain():
    expFileName = "fastflow_CF.exp"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    hp = {}
    hp["shapefile"] = os.path.join(repoPath, "dataset", expFileName)

    p = pinn.parameter.DomainParameter()
    # Check no data file exceptions
    with pytest.raises(Exception):
        pinn.domain.Domain(p)

    p2 = pinn.parameter.DomainParameter(hp)
    d2 = pinn.domain.Domain(p2)
    assert type(d2.geometry) == dde.geometry.Polygon

    hp["time_dependent"] = True
    hp["start_time"] = 0
    hp["end_time"] = 1
    p2 = pinn.parameter.DomainParameter(hp)
    d2 = pinn.domain.Domain(p2)
    assert type(d2.geometry) != dde.geometry.Polygon
    assert type(d2.geometry) == dde.geometry.GeometryXTime
    assert (d2.geometry.random_points(5)).shape == (5, 3)
