import os
import pinnicle as pinn
import numpy as np
import deepxde as dde
from deepxde.backend import backend_name
from pinnicle.utils import data_misfit, plot_nn
import pytest

dde.config.set_default_float('float64')

inputFileName="Helheim_fastflow.mat"
expFileName = "fastflow_CF.exp"

# path for loading data and saving models
repoPath = os.path.dirname(__file__) + "/../examples/"
appDataPath = os.path.join(repoPath, "dataset")
path = os.path.join(appDataPath, inputFileName)

hp = {}
# General parameters
hp["epochs"] = 10
hp["learning_rate"] = 0.001
hp["loss_functions"] = "MSE"
hp["is_save"] = False

# NN
hp["activation"] = "tanh"
hp["initializer"] = "Glorot uniform"
hp["num_neurons"] = 10
hp["num_layers"] = 4

# data
issm = {}
issm["data_path"] = path
issm["data_size"] = {"u":10, "v":10, "s":10, "H":10, "C":None, "vel":10, "B":10}
hp["data"] = {"ISSM": issm}

# domain
hp["shapefile"] = os.path.join(repoPath, "dataset", expFileName)
hp["num_collocation_points"] = 10

# extension of the saved model:
if backend_name == "tensorflow":
    extension = "weights.h5"
elif backend_name == "pytorch":
    extension = "pt"

def test_SSA_pde_function():
    hp_local = dict(hp)
    hp_local["equations"] = {"SSA":{}}
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    y = experiment.model.predict(experiment.model_data.X['u'], operator=experiment.physics.operator("SSA"))
    assert len(y) == 2
    assert y[0].shape == (10,1)
    assert y[1].shape == (10,1)

def test_SSA_VB_pde_function():
    hp_local = dict(hp)
    hp_local["equations"] = {"SSA_VB":{}}
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    y = experiment.model.predict(experiment.model_data.X['u'], operator=experiment.physics.operator("SSA_VB"))
    assert len(y) == 2
    assert y[0].shape == (10,1)
    assert y[1].shape == (10,1)

@pytest.mark.skipif(backend_name=="jax", reason="MOLHO is not implemented for jax")
def test_MOLHO_pde_function():
    hp_local = dict(hp)
    hp_local["equations"] = {"MOLHO":{}}
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    y = experiment.model.predict(experiment.model_data.X['u'], operator=experiment.physics.operator("MOLHO"))
    assert len(y) == 4
    assert y[0].shape == (10,1)
    assert y[3].shape == (10,1)

def test_MC_pde_function():
    hp_local = dict(hp)
    hp_local["equations"] = {"MC":{}}
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    y = experiment.model.predict(experiment.model_data.X['u'], operator=experiment.physics.operator("MC"))
    assert len(y) == 1
    assert y[0].shape == (10,1)

def test_thickness_pde_function():
    hp_local = dict(hp)
    hp_local["equations"] = {"Mass transport":{}}
    hp_local["time_dependent"] = True
    hp_local["start_time"] = 0
    hp_local["end_time"] = 1
    experiment = pinn.PINN(params=hp_local)
    assert experiment.model_data.X['u'][1,2] == 0
    experiment.compile()
    y = experiment.model.predict(experiment.model_data.X['u'], operator=experiment.physics.operator("mass transport"))

    assert len(y) == 1
    assert y[0].shape == (10,1)

def test_ssashelf_pde_function():
    hp_local = dict(hp)
    hp_local["equations"] = {"SSA_SHELF":{}}
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    y = experiment.model.predict(experiment.model_data.X['u'], operator=experiment.physics.operator("SSA_SHELF"))

    assert len(y) == 2
    assert y[0].shape == (10,1)
    assert y[1].shape == (10,1)

def test_ssashelfB_pde_function():
    hp_local = dict(hp)
    hp_local["equations"] = {"SSA_SHELF_VB": {}}
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    y = experiment.model.predict(experiment.model_data.X['u'], operator=experiment.physics.operator("SSA_SHELF_VB"))

    assert len(y) == 2
    assert y[0].shape == (10,1)
    assert y[1].shape == (10,1)

def test_time_invariant_func():
    hp_local = dict(hp)
    hp_local["equations"] = {"Time_Invariant": {}}
    hp_local["time_dependent"] = True
    hp_local["start_time"] = 0
    hp_local["end_time"] = 1
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    y = experiment.model.predict(experiment.model_data.X['u'], operator=experiment.physics.operator("Time_Invariant"))

    assert len(y) == 2
    assert y[0].shape == (10,1)
    assert y[1].shape == (10,1)

def test_vel_mag():
    hp_local = dict(hp)
    hp_local["equations"] = {"SSA": {}}
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    sol = experiment.model.predict(experiment.model_data.X['u'])
    vel_sol = np.sqrt(sol[:,0]**2+sol[:,1]**2)
    def op(i,o):
        return experiment.physics.vel_mag(i,o,None)
    vel = experiment.model.predict(experiment.model_data.X['u'], operator=op)
    assert np.all(vel >=0)
    assert vel.shape == (10,1)
    assert np.all(vel.flatten() - vel_sol.flatten() < 2.0*np.finfo(float).eps)

@pytest.mark.skipif(backend_name=="jax", reason="jacobian function implemented for jax uses different syntax, skip from test for now")
def test_user_defined_grad():
    hp_local = dict(hp)
    hp_local["equations"] = {"SSA": {}}
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    def op2(i,o):
        return experiment.physics.surf_x(i,o,None)
    surfx = experiment.model.predict(experiment.model_data.X['u'], operator=op2)
    assert surfx.shape == (10,1)

    def op3(i,o):
        return experiment.physics.surf_y(i,o,None)
    surfy = experiment.model.predict(experiment.model_data.X['u'], operator=op3)
    assert surfy.shape == (10,1)

    def op4(i,o):
        return experiment.physics.user_defined_gradient('s','x')(i, o, None)
    surfx1 = experiment.model.predict(experiment.model_data.X['u'], operator=op4)
    assert np.all(surfx == surfx1)

    def op5(i,o):
        return experiment.physics.user_defined_gradient('s','y')(i, o, None)
    surfy1 = experiment.model.predict(experiment.model_data.X['u'], operator=op5)
    assert np.all(surfy == surfy1)
