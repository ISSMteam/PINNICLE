import PINN_ICE as pinn
import numpy as np
from PINN_ICE.physics import *
import pytest

def test_update_cid():
    ssa = SSA2DUniformB(1)

    # input id
    i_var = ["y", "x"]
    assert ssa.local_input_var["x"] == 0
    ssa.update_id(global_input_var=i_var)
    assert ssa.local_input_var["x"] == 1

    # output id
    o_var = ["u", "v", "H", "s", "a", "C"]
    assert ssa.local_output_var["H"] == 3
    ssa.update_id(global_output_var=o_var)
    assert ssa.local_output_var["H"] == 2

    # Check exceptions
    with pytest.raises(Exception):
        g_var = ["u", "v", "H", "a", "C"]
        ssa.update_id(global_output_var=g_var)

def test_Physics_SSA():
    hp = {}
    hp["equations"] = ["SSA"]
    hp["scalar_variables"] = {"B":1.26802073401e+08}
    phy = Physics(PhysicsParameter(hp))

    assert phy.input_var == ['x', 'y']
    assert phy.output_var == ['u', 'v', 's', 'H', 'C']
    assert phy.residuals == ['fSSA1', 'fSSA2']
    assert len(phy.output_lb) == 5
    assert len(phy.output_ub) == 5
    assert len(phy.data_weights) == 5
    assert len(phy.pde_weights) == 2

def test_Physics_MOLHO():
    hp = {}
    hp["equations"] = ["MOLHO"]
    hp["scalar_variables"] = {"B":1.26802073401e+08}
    phy = Physics(PhysicsParameter(hp))

    assert phy.input_var == ['x', 'y']
    assert phy.output_var == ['u', 'v',  'u_base', 'v_base', 's', 'H', 'C']
    assert phy.residuals == ['fMOLHO 1', 'fMOLHO 2', 'fMOLHO base 1', 'fMOLHO base 2']
    assert phy.equations[0].local_output_var == {'u': 0, 'v': 1, 'u_base': 2, 'v_base': 3, 's': 4, 'H': 5, 'C': 6}
    assert len(phy.output_lb) == 7
    assert len(phy.output_ub) == 7
    assert len(phy.data_weights) == 7
    assert len(phy.pde_weights) == 4

def test_Physics_SSA_MOLHO():
    hp = {}
    hp["equations"] = ["SSA", "MOLHO"]
    hp["scalar_variables"] = {"B":1.26802073401e+08}
    phy = Physics(PhysicsParameter(hp))

    hp["equations"] = ["MOLHO"]
    phy2 = Physics(PhysicsParameter(hp))

    assert phy.input_var == ['x', 'y']
    assert phy.output_var == ['u', 'v', 's', 'H', 'C', 'u_base', 'v_base']
    assert phy.residuals == ['fSSA1', 'fSSA2', 'fMOLHO 1', 'fMOLHO 2', 'fMOLHO base 1', 'fMOLHO base 2']
    assert phy.equations[1].local_output_var == {'u': 0, 'v': 1, 'u_base': 5, 'v_base': 6, 's': 2, 'H': 3, 'C': 4}
    assert len(phy.output_lb) == 7
    assert len(phy.output_ub) == 7
    assert phy.output_lb[5] == phy2.output_lb[2]
    assert len(phy.data_weights) == 7
    assert len(phy.pde_weights) == 6
