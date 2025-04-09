import os
import pinnicle as pinn
import numpy as np
import deepxde as dde
from deepxde.backend import backend_name
from pinnicle.utils import data_misfit, plot_nn
import pytest

dde.config.set_default_float('float64')
#dde.config.disable_xla_jit()

weights = [7, 7, 5, 5, 3, 3, 5]

inputFileName="Helheim_fastflow.mat"
expFileName = "fastflow_CF.exp"

# path for loading data and saving models
repoPath = os.path.dirname(__file__) + "/../examples/"
appDataPath = os.path.join(repoPath, "dataset")
path = os.path.join(appDataPath, inputFileName)
yts =3600*24*365
loss_weights = [10**(-w) for w in weights]
loss_weights[2] = loss_weights[2] * yts*yts
loss_weights[3] = loss_weights[3] * yts*yts

hp = {}
# General parameters
hp["epochs"] = 10
hp["loss_weights"] = loss_weights
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

# domain
hp["shapefile"] = os.path.join(repoPath, "dataset", expFileName)
hp["num_collocation_points"] = 9000

# physics
SSA = {}
SSA["scalar_variables"] = {"B":1.26802073401e+08}
hp["equations"] = {"SSA":SSA}

# extension of the saved model:
if backend_name == "tensorflow":
    extension = "weights.h5"
elif backend_name == "pytorch":
    extension = "pt"

def test_compile_no_data():
    hp_local = dict(hp)
    issm["data_size"] = {}
    hp_local["data"] = {"ISSM":issm}
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    assert experiment.loss_names == ['fSSA1', 'fSSA2']
    assert experiment.params.nn.output_variables == ['u', 'v', 's', 'H', 'C']
    assert experiment.params.nn.output_lb[0]<0.0
    assert experiment.params.nn.output_ub[0]>0.0
    assert experiment.params.nn.output_lb[1]<0.0
    assert experiment.params.nn.output_ub[1]>0.0

def test_add_loss():
    hp_local = dict(hp)
    # additional loss
    vel_loss = {}
    vel_loss['name'] = "vel log"
    vel_loss['function'] = "VEL_LOG"
    vel_loss['weight'] = 1.0
    hp_local["additional_loss"] = {"vel":vel_loss}
    issm["data_size"] = {"u":4000, "v":4000, "s":4000, "H":4000, "C":None}
    hp_local["data"] = {"ISSM": issm}
    experiment = pinn.PINN(params=hp_local)
    assert len(experiment.training_data) == 5
    assert type(experiment.training_data[-1]) == dde.icbc.boundary_conditions.PointSetBC
    assert len(experiment.loss_names) == 7
    assert len(experiment.params.training.loss_weights) == 7
    assert experiment.params.training.loss_functions == ["MSE"]*7

    issm["data_size"] = {"u":4000, "v":4000, "s":4000, "H":4000, "C":None, "vel":4000}
    hp_local["data"] = {"ISSM": issm}
    experiment = pinn.PINN(params=hp_local)
    assert len(experiment.training_data) == 6
    assert type(experiment.training_data[-1]) == dde.icbc.boundary_conditions.PointSetOperatorBC
    assert len(experiment.loss_names) == 8
    assert len(experiment.params.training.loss_weights) == 8
    assert len(experiment.params.training.loss_functions) == 8
    assert experiment.params.training.loss_functions == ["MSE"]*7 + [data_misfit.get("VEL_LOG")]

    vel_loss['function'] = "MAPE"
    hp_local["additional_loss"] = {"vel":vel_loss}
    experiment = pinn.PINN(params=hp_local)
    assert experiment.params.training.loss_functions == ["MSE"]*7 + [data_misfit.get("MAPE")]

def test_save_and_load_setting(tmp_path):
    experiment = pinn.PINN(params=hp)
    experiment.save_setting(path=tmp_path)
    assert experiment.params.param_dict == experiment.load_setting(path=tmp_path)
    assert os.path.isdir(f"{tmp_path}/pinn/")
    experiment2 = pinn.PINN(loadFrom=tmp_path)
    assert experiment.params.param_dict == experiment2.params.param_dict

def test_update_parameters():
    experiment = pinn.PINN(params=hp)
    experiment.update_parameters({})
    assert experiment.params.param_dict == hp
    experiment.update_parameters({"add_param": 1})
    assert experiment.params.param_dict["add_param"] == 1
    experiment.update_parameters({"add_param": 2})
    assert experiment.params.param_dict["add_param"] == 2

def test_train_only_data(tmp_path):
    hp_local = dict(hp)
    hp_local["is_parallel"] = False
    hp_local["is_save"] = False
    hp_local["num_collocation_points"] = 100
    issm["data_size"] = {"u":100, "v":100, "s":100, "H":100}
    hp_local["num_neurons"] = [4,10];
    hp_local["data"] = {"ISSM": issm}
    dummy = {}
    dummy["output"] = ['v', 'H']
    hp_local["equations"] = {"DUMMY":dummy}
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    experiment.train()
    assert experiment.loss_names == ['v', 'H']

def test_train(tmp_path):
    hp_local = dict(hp)
    hp_local["is_save"] = False
    hp_local["num_collocation_points"] = 100
    issm["data_size"] = {"u":100, "v":100, "s":100, "H":100, "C":None}
    hp_local["data"] = {"ISSM": issm}
    hp_local["equations"] = {"SSA":SSA}
    experiment = pinn.PINN(params=hp_local)
    experiment.compile(decay=("inverse time", 5, 0.3))
    experiment.train()
    assert experiment.loss_names == ['fSSA1', 'fSSA2', 'u', 'v', 's', 'H', 'C']

def test_train_decay(tmp_path):
    hp_local = dict(hp)
    hp_local["is_save"] = False
    hp_local["num_collocation_points"] = 100
    issm["data_size"] = {"u":None, "v":100, "s":100, "H":100, "C":None}
    hp_local["data"] = {"ISSM": issm}
    hp_local["equations"] = {"SSA":SSA}
    hp_local["decay_steps"] = 5
    hp_local["decay_rate"]= 0.3
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    experiment.train()
    assert experiment.loss_names == ['fSSA1', 'fSSA2', 'u', 'v', 's', 'H', 'C']

def test_fft_training(tmp_path):
    hp_local = dict(hp)
    hp_local['fft'] = True
    hp_local["is_save"] = False
    hp_local["num_collocation_points"] = 10
    issm["data_size"] = {"u":10, "v":10, "s":10, "H":10, "C":None}
    hp_local["data"] = {"ISSM": issm}
    hp_local["equations"] = {"SSA":SSA}
    experiment = pinn.PINN(params=hp_local)
    experiment.save_setting(path=tmp_path)
    assert experiment.params.param_dict == experiment.load_setting(path=tmp_path)
    assert experiment.params.nn.B is None
    assert os.path.isdir(f"{tmp_path}/pinn/")
    experiment2 = pinn.PINN(loadFrom=tmp_path)
    assert experiment.params.param_dict == experiment2.params.param_dict
    assert len(experiment2.params.nn.B) == 2
    assert len(experiment2.params.nn.B[1]) == 10    

@pytest.mark.skipif(backend_name in ["jax"], reason="save model is not implemented in deepxde for jax")
def test_train_PFNN(tmp_path):
    hp_local = dict(hp)
    hp_local["is_parallel"] = True
    hp_local["is_save"] = False
    hp_local["num_collocation_points"] = 10
    issm["data_size"] = {"u":10, "v":10, "s":10, "H":10, "C":None, "vel":10}
    hp_local["num_neurons"] = [4,10];
    hp_local["data"] = {"ISSM": issm}
    # additional loss
    vel_loss = {}
    vel_loss['name'] = "vel log"
    vel_loss['function'] = "VEL_LOG"
    vel_loss['weight'] = 1.0
    hp_local["additional_loss"] = {"vel":vel_loss}
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    experiment.train()
    assert experiment.loss_names == ['fSSA1', 'fSSA2', 'u', 'v', 's', 'H', 'C', "vel log"]
    assert experiment.params.nn.num_layers == 2
    if backend_name == "pytorch":
        assert len(experiment.model.net.layers) == 3
        assert len(experiment.model.net.layers[0]) == 5
        assert len(experiment.model.net.layers[1]) == 5
        assert len(experiment.model.net.layers[2]) == 5
        assert len(list(experiment.model.net.parameters())) == 30
    else:
        assert len(experiment.model.net.layers) == 5*(2+1)
        assert len(experiment.model.net.trainable_weights) == 30

@pytest.mark.skipif(backend_name in ["jax"], reason="save model is not implemented in deepxde for jax")
def test_save_and_load_train(tmp_path):
    hp_local = dict(hp)
    hp_local["save_path"] = str(tmp_path)
    hp_local["is_save"] = True
    hp_local["is_plot"] = True
    hp_local["num_collocation_points"] = 10
    issm["data_size"] = {"u":10, "v":10, "s":10, "H":10, "C":None, "vel":10}
    hp_local["data"] = {"ISSM": issm}
    hp_local["is_parallel"] = False
    # additional loss
    vel_loss = {}
    vel_loss['name'] = "vel log"
    vel_loss['function'] = "VEL_LOG"
    vel_loss['weight'] = 1.0
    hp_local["additional_loss"] = {"vel":vel_loss}
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    experiment.train()
    assert experiment.loss_names == ['fSSA1', 'fSSA2', 'u', 'v', 's', 'H', 'C', "vel log"]
    assert os.path.isfile(f"{tmp_path}/pinn/model-{hp_local['epochs']}.{extension}")
    experiment_load = pinn.PINN(params=hp_local)
    experiment_load.load_model(path=tmp_path, epochs=hp_local['epochs'])
    assert np.all(experiment_load.model.predict(experiment.model_data.X['u'])==experiment.model.predict(experiment.model_data.X['u']))

@pytest.mark.skipif(backend_name in ["jax"], reason="save model is not implemented in deepxde for jax")
def test_train_with_callbacks(tmp_path):
    hp_local = dict(hp)
    hp_local["save_path"] = str(tmp_path)
    hp_local["is_save"] = True
    hp_local["num_collocation_points"] = 100
    issm["data_size"] = {"u":100, "v":100, "s":100, "H":100, "C":None, "vel":100}
    hp_local["data"] = {"ISSM": issm}
    # additional loss
    vel_loss = {}
    vel_loss['name'] = "vel log"
    vel_loss['function'] = "VEL_LOG"
    vel_loss['weight'] = 1.0
    hp_local["additional_loss"] = {"vel":vel_loss}
    # callbacks
    hp_local["min_delta"] = 1e10
    hp_local["period"] = 5
    hp_local["patience"] = 8
    hp_local["checkpoint"] = True
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    experiment.train()
    assert experiment.loss_names == ['fSSA1', 'fSSA2', 'u', 'v', 's', 'H', 'C', "vel log"]
    assert os.path.isfile(f"{tmp_path}/pinn/model-1.{extension}")
    assert os.path.isfile(f"{tmp_path}/pinn/model-9.{extension}")
    assert not os.path.isfile(f"{tmp_path}/pinn/model-{hp_local['epochs']}.{extension}")

def test_only_callbacks(tmp_path):
    hp_local = dict(hp)
    hp_local["save_path"] = str(tmp_path)
    hp_local["num_collocation_points"] = 100
    issm["data_size"] = {"u":100, "v":100, "s":100, "H":100, "C":None, "vel":100}
    hp_local["data"] = {"ISSM": issm}
    hp_local["min_delta"] = 1e10
    hp_local["period"] = 5
    hp_local["patience"] = 8
    hp_local["checkpoint"] = True
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    callbacks = experiment.update_callbacks()
    assert callbacks is not None
    assert len(callbacks) == 3

@pytest.mark.skipif(backend_name=="jax", reason="plot_prediection is not implemented for jax")
def test_plot(tmp_path):
    hp_local = dict(hp)
    hp_local["save_path"] = str(tmp_path)
    hp_local["is_save"] = True
    issm["data_size"] = {"u":10, "v":10, "s":10, "H":10, "C":None}
    hp_local["data"] = {"ISSM": issm}
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    assert experiment.plot_predictions(X_ref=experiment.model_data.data["ISSM"].X_dict, 
                                       sol_ref=experiment.model_data.data["ISSM"].data_dict, 
                                       resolution=10) is None
    X_ref = np.hstack((experiment.model_data.data["ISSM"].X_dict['x'].flatten()[:,None], 
                       experiment.model_data.data["ISSM"].X_dict['y'].flatten()[:,None]))
    assert experiment.plot_predictions(X_ref=X_ref, 
                                       sol_ref=experiment.model_data.data["ISSM"].data_dict, 
                                       resolution=10, absvariable=['C']) is None
    X, Y, im_data, axs = plot_nn(experiment, experiment.model_data.data["ISSM"].data_dict, resolution=10);
    assert X.shape == (10,10)
    assert Y.shape == (10,10)
    assert len(im_data) == 5
    assert im_data['u'].shape == (10,10) 

    thickness = {}
    hp_local["equations"] = {"Mass transport":thickness}
    hp_local["num_collocation_points"] = 10
    hp_local["time_dependent"] = True
    hp_local["start_time"] = 0
    hp_local["end_time"] = 1
    experiment = pinn.PINN(params=hp_local)
    experiment.compile()
    y = experiment.model.predict(experiment.model_data.X['u'], operator=experiment.physics.operator("mass transport"))
    assert experiment.plot_predictions(X_ref=experiment.model_data.data["ISSM"].X_dict,
            sol_ref=experiment.model_data.data["ISSM"].data_dict,
            resolution=10) is None

