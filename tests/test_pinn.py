import os
import pinnicle as pinn
import numpy as np
import deepxde as dde
from pinnicle.utils import data_misfit, plot_nn, plot_similarity

dde.config.set_default_float('float64')
dde.config.disable_xla_jit()

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
hp["num_layers"] = 6

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

def test_compile_no_data():
    issm["data_size"] = {}
    hp["data"] = {"ISSM":issm}
    experiment = pinn.PINN(params=hp)
    experiment.compile()
    assert experiment.loss_names == ['fSSA1', 'fSSA2']
    assert experiment.params.nn.output_variables == ['u', 'v', 's', 'H', 'C']
    assert experiment.params.nn.output_lb[0]<0.0
    assert experiment.params.nn.output_ub[0]>0.0
    assert experiment.params.nn.output_lb[1]<0.0
    assert experiment.params.nn.output_ub[1]>0.0

def test_add_loss():
    # additional loss
    vel_loss = {}
    vel_loss['name'] = "vel log"
    vel_loss['function'] = "VEL_LOG"
    vel_loss['weight'] = 1.0
    hp["additional_loss"] = {"vel":vel_loss}
    issm["data_size"] = {"u":4000, "v":4000, "s":4000, "H":4000, "C":None}
    hp["data"] = {"ISSM": issm}
    experiment = pinn.PINN(params=hp)
    assert len(experiment.training_data) == 5
    assert type(experiment.training_data[-1]) == dde.icbc.boundary_conditions.PointSetBC
    assert len(experiment.loss_names) == 7
    assert len(experiment.params.training.loss_weights) == 7
    assert experiment.params.training.loss_functions == ["MSE"]*7

    issm["data_size"] = {"u":4000, "v":4000, "s":4000, "H":4000, "C":None, "vel":4000}
    hp["data"] = {"ISSM": issm}
    experiment = pinn.PINN(params=hp)
    assert len(experiment.training_data) == 6
    assert type(experiment.training_data[-1]) == dde.icbc.boundary_conditions.PointSetOperatorBC
    assert len(experiment.loss_names) == 8
    assert len(experiment.params.training.loss_weights) == 8
    assert len(experiment.params.training.loss_functions) == 8
    assert experiment.params.training.loss_functions == ["MSE"]*7 + [data_misfit.get("VEL_LOG")]

    vel_loss['function'] = "MAPE"
    hp["additional_loss"] = {"vel":vel_loss}
    experiment = pinn.PINN(params=hp)
    assert experiment.params.training.loss_functions == ["MSE"]*7 + [data_misfit.get("MAPE")]

def test_save_and_load_setting(tmp_path):
    experiment = pinn.PINN(params=hp)
    experiment.save_setting(path=tmp_path)
    assert experiment.params.param_dict == experiment.load_setting(path=tmp_path)
    experiment2 = pinn.PINN(loadFrom=tmp_path)
    assert experiment.params.param_dict == experiment2.params.param_dict

#def test_train(tmp_path):
#    hp["save_path"] = str(tmp_path)
#    hp["is_save"] = True
#    hp["num_collocation_points"] = 100
#    issm["data_size"] = {"u":100, "v":100, "s":100, "H":100, "C":None, "vel":100}
#    hp["data"] = {"ISSM": issm}
#    experiment = pinn.PINN(params=hp)
#    experiment.compile()
#    experiment.train()
#    assert experiment.loss_names == ['fSSA1', 'fSSA2', 'u', 'v', 's', 'H', 'C', "vel log"]
#    assert os.path.isfile(f"{tmp_path}/pinn/model-{hp['epochs']}.ckpt.index")
#
#def test_train_with_callbacks(tmp_path):
#    hp["save_path"] = str(tmp_path)
#    hp["is_save"] = True
#    hp["num_collocation_points"] = 100
#    issm["data_size"] = {"u":100, "v":100, "s":100, "H":100, "C":None, "vel":100}
#    hp["data"] = {"ISSM": issm}
#    hp["min_delta"] = 1e10
#    hp["period"] = 5
#    hp["patience"] = 8
#    hp["checkpoint"] = True
#    experiment = pinn.PINN(params=hp)
#    experiment.compile()
#    experiment.train()
#    assert experiment.loss_names == ['fSSA1', 'fSSA2', 'u', 'v', 's', 'H', 'C', "vel log"]
#    assert os.path.isfile(f"{tmp_path}/pinn/model-1.ckpt.index")
#    assert os.path.isfile(f"{tmp_path}/pinn/model-9.ckpt.index")
#    assert not os.path.isfile(f"{tmp_path}/pinn/model-{hp['epochs']}.ckpt.index")

def test_only_callbacks(tmp_path):
    hp["save_path"] = str(tmp_path)
    hp["num_collocation_points"] = 100
    issm["data_size"] = {"u":100, "v":100, "s":100, "H":100, "C":None, "vel":100}
    hp["data"] = {"ISSM": issm}
    hp["min_delta"] = 1e10
    hp["period"] = 5
    hp["patience"] = 8
    hp["checkpoint"] = True
    experiment = pinn.PINN(params=hp)
    experiment.compile()
    callbacks = experiment.update_callbacks()
    assert callbacks is not None
    assert len(callbacks) == 3


def test_plot(tmp_path):
    hp["save_path"] = str(tmp_path)
    hp["is_save"] = True
    issm["data_size"] = {"u":4000, "v":4000, "s":4000, "H":4000, "C":None}
    hp["data"] = {"ISSM": issm}
    experiment = pinn.PINN(params=hp)
    experiment.compile()
    assert experiment.plot_predictions(X_ref=experiment.model_data.data["ISSM"].X_dict, 
                                       sol_ref=experiment.model_data.data["ISSM"].data_dict, 
                                       resolution=10) is None
    X_ref = np.hstack((experiment.model_data.data["ISSM"].X_dict['x'].flatten()[:,None], 
                       experiment.model_data.data["ISSM"].X_dict['y'].flatten()[:,None]))
    assert experiment.plot_predictions(X_ref=X_ref, 
                                       sol_ref=experiment.model_data.data["ISSM"].data_dict, 
                                       resolution=10) is None
    X, Y, im_data, axs = plot_nn(experiment, experiment.model_data.data["ISSM"].data_dict, resolution=10);
    assert X.shape == (10,10)
    assert Y.shape == (10,10)
    assert len(im_data) == 5
    assert im_data['u'].shape == (10,10) 

def test_similarity(tmp_path):
    hp["save_path"] = str(tmp_path)
    hp["is_save"] = False
    issm["data_size"] = {"u":100, "v":100, "s":100, "H":100, "C":None}
    hp["data"] = {"ISSM": issm}
    experiment = pinn.PINN(params=hp)
    experiment.compile()
    # plot_similarity(pinn, feature_name, sim='MAE', cmap='jet', scale=1, cols=[0, 1, 2])
    # default
    fig, axs = plot_similarity(experiment, feature_name='s')
    assert (fig is not None) and (np.size(axs) == 3)
    fig, axs = plot_similarity(experiment, feature_name='H', cols=[0])
    assert (fig is not None) and (np.size(axs) == 1)
    fig, axs = plot_similarity(experiment, feature_name="u", sim="mae", cols=[2])
    assert (fig is not None) and (np.size(axs) == 1) 
    fig, axs = plot_similarity(experiment, feature_name="v", sim="Mse", cols=[2, 1])
    assert (fig is not None) and (np.size(axs) == 2) 
    fig, axs = plot_similarity(experiment, feature_name="C", sim="rmse", cols=[0, 2, 1])
    assert (fig is not None) and (np.size(axs) == 3) 
    fig, axs = plot_similarity(experiment, feature_name="H", sim="SIMPLE")
    assert (fig is not None) and (np.size(axs) == 3) 
