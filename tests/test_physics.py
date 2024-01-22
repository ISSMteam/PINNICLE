import PINN_ICE as pinn
import numpy as np
from PINN_ICE.physics import *
import pytest

def test_update_cid():
    ssa = SSA2DUniformB(1)

    # input id
    i_var = ["y", "x"]
    assert ssa.input_var["x"] == 0
    ssa.update_id(global_input_var=i_var)
    assert ssa.input_var["x"] == 1

    # output id
    o_var = ["u", "v", "H", "s", "a", "C"]
    assert ssa.output_var["H"] == 3
    ssa.update_id(global_output_var=o_var)
    assert ssa.output_var["H"] == 2

    # Check exceptions
    with pytest.raises(Exception):
        g_var = ["u", "v", "H", "a", "C"]
        ssa.update_id(global_output_var=g_var)

def test_Physics_SSA():
    hp = {}
    hp["equations"] = ["SSA"]
    hp["scalar_variables"] = {"B":1.26802073401e+08}
    phy = Physics(PhysicsParameter(hp))

    assert phy.input_var == {'x': 0, 'y': 1}
    assert phy.output_var == {'u': 0, 'v': 1, 's': 2, 'H': 3, 'C': 4}
    assert phy.residuals == ['fSSA1', 'fSSA2']

def test_Physics_SSA_MOLHO():
    hp = {}
    hp["equations"] = ["SSA", "MOLHO"]
    hp["scalar_variables"] = {"B":1.26802073401e+08}
    phy = Physics(PhysicsParameter(hp))

    assert phy.input_var == {'x': 0, 'y': 1}
    assert phy.output_var == {'u': 0, 'v': 1, 's': 2, 'H': 3, 'C': 4, 'u_base': 5, 'v_base': 6}
    assert phy.residuals == ['fSSA1', 'fSSA2', 'fMOLHO1', 'fMLHO2']
