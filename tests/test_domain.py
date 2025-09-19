import pinnicle as pinn
import pytest
import deepxde as dde
import os
import numpy as np


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

    # inside
    c = np.array([0.5*(d2.geometry.bbox[0]+d2.geometry.bbox[1])])
    assert d2.inside(c)
    o = np.array([d2.geometry.bbox[0]-[100,100] ,d2.geometry.bbox[1]+[100,100]])
    assert not np.any(d2.inside(o))

    # bbox
    assert np.all(d2.bbox() == d2.geometry.bbox)

    # add midpoint 
    assert len(d2.vertices) == 13
    d2._add_midpoint(d2.vertices)
    assert len(d2.vertices) == 14

def test_shapebox_domain():
    hp = {}
    hp["shapebox"] = [0,1,0,1]

    # define domain b shapebox
    p = pinn.parameter.DomainParameter(hp)
    d = pinn.domain.Domain(p)
    assert type(d.geometry) == dde.geometry.Polygon

    # inside
    c = np.array([0.5*(d.geometry.bbox[0]+d.geometry.bbox[1])])
    assert d.inside(c)
    c = np.array([[0.5,0.5]])
    assert d.inside(c)
    c = np.array([[1.5,0.5]])
    assert not d.inside(c)

def test_shapebox_margin_domain():
    hp = {}
    hp["shapebox"] = [0,1,0,1]
    hp["margin"] = 0.5

    # define domain b shapebox
    p = pinn.parameter.DomainParameter(hp)
    d = pinn.domain.Domain(p)
    assert type(d.geometry) == dde.geometry.Polygon

    # inside
    c = np.array([0.5*(d.geometry.bbox[0]+d.geometry.bbox[1])])
    assert d.inside(c)
    c = np.array([[0.5,0.5]])
    assert d.inside(c)
    c = np.array([[1.49,0.5]])
    assert d.inside(c)
    c = np.array([[1.51,0.5]])
    assert not d.inside(c)

def test_rectangle_domain():
    # dealing with rectangle domains
    expFileName = "testtile.exp"
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

    # inside
    c = np.array([0.5*(d2.geometry.bbox[0]+d2.geometry.bbox[1])])
    assert d2.inside(c)
    o = np.array([d2.geometry.bbox[0]-[100,100] ,d2.geometry.bbox[1]+[100,100]])
    assert not np.any(d2.inside(o))

    # bbox
    assert np.all(d2.bbox() == d2.geometry.bbox)

def test_time_dependent_domain():
    expFileName = "fastflow_CF.exp"
    repoPath = os.path.dirname(__file__) + "/../examples/"
    hp = {}
    hp["shapefile"] = os.path.join(repoPath, "dataset", expFileName)

    hp["time_dependent"] = True
    hp["start_time"] = 0
    hp["end_time"] = 1
    p2 = pinn.parameter.DomainParameter(hp)
    d2 = pinn.domain.Domain(p2)
    assert type(d2.geometry) != dde.geometry.Polygon
    assert type(d2.geometry) == dde.geometry.GeometryXTime
    assert (d2.geometry.random_points(5)).shape == (5, 3)

    c = np.array([0.5*(d2.geometry.geometry.bbox[0]+d2.geometry.geometry.bbox[1])])
    assert d2.inside(c)
    o = np.array([d2.geometry.geometry.bbox[1]-[100,100] ,d2.geometry.geometry.bbox[1]+[100,100]])
    assert not np.any(d2.inside(o))

    # bbox
    assert np.all(d2.bbox() == d2.geometry.geometry.bbox)

