# export DDE_BACKEND=pytorch in terminal before running pytest
import os

# MUST be set before importing torch/deepxde/pinnicle
#os.environ.setdefault("DDE_BACKEND", "pytorch")

import pinnicle as pinn
from deepxde.backend import backend_name, torch
import pytest

if backend_name == "pytorch":
    os.environ.setdefault("MPLBACKEND", "Agg")

"""Purpose of Tests:
    - Test 1 checks:
        1. Weights are loaded correctly
        2. Some paramaters are frozen
        3. Some parameters remain trainable
## Test 1 confirms transfer learning setup is working correctly ##
    - Test 2 checks:
        1. Frozen parameters do not change after training
        2. Trainable parameters do change
## Test 2 confirms freezing is actually enforced during training ##
"""

# path to dataset
repoPath = os.path.dirname(__file__) + "/../examples/"
appDataPath = os.path.join(repoPath, "dataset")
path = os.path.join(appDataPath, "Helheim_fastflow.mat")

# count how many parameters are frozen and how many are trainable
def count_frozen(net):
    params = list(net.parameters())
    n_frozen = sum(1 for p in params if p.requires_grad is False)
    n_trainable = sum(1 for p in params if p.requires_grad is True)
    return n_frozen, n_trainable

# helper to inspect parameter (debugging)
def debug_param_names(net):
    return [(name, p.requires_grad, tuple(p.shape)) for name, p in net.named_parameters()]

# forward pass to ensure network is initialized
def materialize_net(experiment):
    x = torch.zeros((1, len(experiment.params.nn.input_variables)), dtype=torch.float32)
    _ = experiment.model.net(x)
    
# Base hyperparameter configuration 
def base_hp(tmp_path):
    hp = {}
    hp["shapebox"] = [-1e7, 1e7, -1e7, 1e7]

    hp["input_variables"] = ["x", "y"]
    hp["output_variables"] = ["u"]
    hp["num_layers"] = 2
    hp["num_neurons"] = 8
    hp["activation"] = "tanh"
    hp["output_lb"] = [-1e4]
    hp["output_ub"] = [1e4]

    hp["equations"] = {"SSA": {"input": ["x", "y"]}}
    hp["X_map"] = {"x": "x", "y": "y"}

    hp["data"] = {
        "dummy": {
            "data_path": path,
            "data_size": {"u": 10},
            "source": "ISSM Light",
        }
    }

    hp["epochs"] = 1
    hp["is_save"] = False
    hp["save_path"] = str(tmp_path)

    # default transfer learning configuration
    hp["transfer_learning"] = {
        "enabled": True,
        "weights_path": "",
        "strict": False,
        "prefix_strip": ["net.", "model.", "module."],
        "freeze": {
            "mode": "fraction",
            "fraction": 0.8,
            "first_n": 0,   # not selected mode. set to zero.
            "target": "linears",
        },
    }
    return hp

# Phase 1: train a model, save the weights
def build_phase1_and_save_weights(tmp_path):
    hp = base_hp(tmp_path)
    hp["transfer_learning"] = dict(hp["transfer_learning"])
    hp["transfer_learning"]["enabled"] = False  # disable transfer learning for phase 1
    hp["transfer_learning"]["weights_path"] = ""

    experiment = pinn.PINN(params=hp)
    experiment.compile()
    materialize_net(experiment)

    with torch.no_grad():
        for p in experiment.model.net.parameters():
            p.fill_(0.1234)

    weights_path = os.path.join(str(tmp_path), "p1_state.pt")
    torch.save(experiment.model.net.state_dict(), weights_path)
    return hp, weights_path

# Phase 2: load saved weights and apply transfer learning
def build_phase2_with_tl(hp, weights_path):
    hp2 = dict(hp)
    hp2["transfer_learning"] = dict(hp["transfer_learning"])
    hp2["transfer_learning"]["enabled"] = True
    hp2["transfer_learning"]["weights_path"] = str(weights_path)

    experiment = pinn.PINN(params=hp2)
    experiment.compile()
    materialize_net(experiment)
    return hp2, experiment

# Verify weights are loaded and freezing is applied
@pytest.mark.skipif(backend_name!="pytorch", reason="transfer learning only supports pytorch")
def test_transfer_learning_freeze_and_load(tmp_path):
    hp, weights_path = build_phase1_and_save_weights(tmp_path)  # build phase 1 model and save the weights
    hp2, experiment = build_phase2_with_tl(hp, weights_path)    # build phase 2 model with transfer learning    

    tl_enabled = hp2.get("transfer_learning", {}).get("enabled", False)
    n_frozen, n_trainable = count_frozen(experiment.model.net)  # count frozen vs. trainable parameters

    if not tl_enabled:  # nothing should freeze if transfer learning is not enabled
        assert n_frozen == 0
        return

    if n_frozen == 0:   # ensure some parameters are frozen
        names = debug_param_names(experiment.model.net)
        raise AssertionError(
            "Transfer learning was enabled but no parameters were frozen.\n"
            f"Frozen={n_frozen}, Trainable={n_trainable}\n"
            f"First params:\n{names[:10]}"
        )

    # ensure some paramters remain trainable
    assert n_trainable > 0, (
        "Expected some parameters to remain trainable in fraction mode "
        f"but all were frozen. Frozen={n_frozen}, Trainable={n_trainable}"
    )

    # verify weights were correctly loaded
    saved_sd = torch.load(weights_path, map_location="cpu")
    model_sd = experiment.model.net.state_dict()

    saved_first = next(iter(saved_sd.values()))
    model_first = next(iter(model_sd.values()))

    assert torch.allclose(saved_first, model_first), (
        "Expected weights in phase 2 to match exactly weights in phase 1."
    )

# Ensure fozen parameters do not update during training
@pytest.mark.skipif(backend_name!="pytorch", reason="transfer learning only supports pytorch")
def test_frozen_params_do_not_update(tmp_path):
    hp, weights_path = build_phase1_and_save_weights(tmp_path)
    _, experiment = build_phase2_with_tl(hp, weights_path)

    # save parameter values before training step
    before = {
        name: p.detach().clone()
        for name, p in experiment.model.net.named_parameters()
    }

    # optimizer only updates trainable parameters
    optimizer = torch.optim.Adam(
        [p for p in experiment.model.net.parameters() if p.requires_grad],
        lr=1e-3,
    )

    # dummy training
    x = torch.randn(8, len(experiment.params.nn.input_variables))
    y = torch.randn(8, 1)

    optimizer.zero_grad()
    pred = experiment.model.net(x)
    loss = torch.mean((pred - y) ** 2)
    loss.backward()
    optimizer.step()

    # save parameter values after update
    after = {
        name: p.detach().clone()
        for name, p in experiment.model.net.named_parameters()
    }

    # check the behavior of frozen vs. trainable paramters
    saw_frozen = False
    saw_trainable = False
    trainable_changed = False

    for name, p in experiment.model.net.named_parameters():
        changed = not torch.allclose(before[name], after[name])

        if p.requires_grad:
            saw_trainable = True
            if changed:
                trainable_changed = True
        else:
            saw_frozen = True
            assert not changed, f"Frozen parameter changed after optimizer step: {name}"
            
    # final checks
    assert saw_frozen, "Expected at least one frozen parameter."
    assert saw_trainable, "Expected at least one trainable parameter."
    assert trainable_changed, "No trainable parameters changed after optimizer step."
