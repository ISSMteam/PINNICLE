import pinnicle as pinn
from pinnicle.physics import *
from pinnicle.parameter import PhysicsParameter
import pytest

def test_Constants():
    SSA = {}
    SSA["scalar_variables"] = {"B":1.26802073401e+08}
    hp = {}
    hp["equations"] = {"SSA":SSA}
    phy = Physics(PhysicsParameter(hp))
    assert phy.equations[0].rhoi == 917.0
    assert phy.equations[0].rhow == 1023.0
    assert phy.equations[0].g == 9.81
    assert phy.equations[0].yts == 3600.0*24*365
    assert phy.equations[0].variable_lb.keys() == phy.equations[0].variable_ub.keys()
    assert all([phy.equations[0].variable_lb[k] < phy.equations[0].variable_ub[k] for k in phy.equations[0].variable_lb])

def test_update_cid():
    p = SSAEquationParameter({"scalar_variables":{"B":1}})
    ssa = SSA(p)

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
    SSA = {}
    SSA["scalar_variables"] = {"B":1.26802073401e+08}
    hp = {}
    hp["equations"] = {"SSA":SSA}
    phy = Physics(PhysicsParameter(hp))

    assert phy.input_var == ['x', 'y']
    assert phy.output_var == ['u', 'v', 's', 'H', 'C']
    assert phy.residuals == ['fSSA1', 'fSSA2']
    assert len(phy.output_lb) == 5
    assert len(phy.output_ub) == 5
    assert len(phy.data_weights) == 5
    assert len(phy.pde_weights) == 2

def test_Physics_SSAVB():
    SSA = {}
    SSA["scalar_variables"] = {"n":3}
    hp = {}
    hp["equations"] = {"SSA_VB":SSA}
    phy = Physics(PhysicsParameter(hp))

    assert phy.input_var == ['x', 'y']
    assert phy.output_var == ['u', 'v', 's', 'H', 'C','B']
    assert phy.residuals == ['fSSA_VB1', 'fSSA_VB2']
    assert len(phy.output_lb) == 6
    assert len(phy.output_ub) == 6
    assert len(phy.data_weights) == 6
    assert len(phy.pde_weights) == 2

def test_Physics_MC():
    hp = {}
    hp["equations"] = {"MC":{}}
    phy = Physics(PhysicsParameter(hp))

    assert phy.input_var == ['x', 'y']
    assert phy.output_var == ['u', 'v', 'a', 'H']
    assert phy.residuals == ['fMC']
    assert len(phy.output_lb) == 4
    assert len(phy.output_ub) == 4
    assert len(phy.data_weights) == 4
    assert len(phy.pde_weights) == 1

def test_Physics_Thickness():
    hp = {}
    hp["equations"] = {"Mass transport":{}}
    phy = Physics(PhysicsParameter(hp))

    assert phy.input_var == ['x', 'y', 't']
    assert phy.output_var == ['u', 'v', 'a', 'H']
    assert phy.residuals == ['fMass transport']
    assert len(phy.output_lb) == 4
    assert len(phy.output_ub) == 4
    assert len(phy.data_weights) == 4
    assert len(phy.pde_weights) == 1

def test_Physics_MOLHO():
    MOLHO = {}
    MOLHO["scalar_variables"] = {"B":1.26802073401e+08}
    hp = {}
    hp["equations"] = {"MOLHO":MOLHO}
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
    SSA = {}
    SSA["scalar_variables"] = {"B":1.26802073401e+08}
    MOLHO = {}
    MOLHO["scalar_variables"] = {"B":1.26802073401e+08}
    hp = {}
    hp["equations"] = {"SSA":SSA, "MOLHO":MOLHO}
    phy = Physics(PhysicsParameter(hp))

    hp["equations"] = {"MOLHO":MOLHO}
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

def test_Physics_Exception():
    hp = {}
    hp["equations"] = {"Not defined":{}}

    with pytest.raises(Exception):
        Physics(PhysicsParameter(hp))

def test_update_Physics_SSA():
    SSA = {}
    SSA["scalar_variables"] = {"B":1.26802073401e+08}
    hp = {}
    hp["equations"] = {"SSA":SSA}
    phy = Physics(PhysicsParameter(hp))
    assert phy.input_var == ['x', 'y']

    SSA["input"] = ['x']
    SSA["output"] = ['a','b','c']
    SSA["output_lb"] =[-1.0,-1.0,-1.0]
    SSA["output_ub"] =[1.0,1.0,1.0]
    SSA["data_weights"] = [10.0, 1.0, 10.0]
    SSA["pde_weights"] = [1.0, 1.0 ]
    hp["equations"] = {"SSA":SSA}
    phy = Physics(PhysicsParameter(hp))
    assert phy.input_var == ['x']
    assert phy.output_var == ['a', 'b', 'c']
    assert phy.output_lb == [-1, -1, -1]
    assert phy.output_ub == [1, 1, 1]
    assert phy.data_weights == [10.0, 1.0, 10.0]
    assert phy.pde_weights == [1.0, 1.0]

    SSA["pde_weights"] = [1.0]
    hp["equations"] = {"SSA":SSA}
    with pytest.raises(Exception):
        phy = Physics(PhysicsParameter(hp))

    SSA["pde_weights"] = 1.0
    hp["equations"] = {"SSA":SSA}
    with pytest.raises(Exception):
        phy = Physics(PhysicsParameter(hp))

def test_operator():
    SSA = {}
    SSA["scalar_variables"] = {"B":1.26802073401e+08}

    hp = {}
    hp["equations"] = {"MC":{}, "SSA":SSA, "SSA_VB":{}, "MOLHO":{}, "Mass transport":{}, "Time_Invariant":{}}
    phy = Physics(PhysicsParameter(hp))
    
    assert phy.operator('mc')
    assert phy.operator('Mc')
    assert phy.operator('SSA')
    assert phy.operator('ssa')
    assert phy.operator('SSA_VB')
    assert phy.operator('ssa_vb')
    assert phy.operator('molho')
    assert phy.operator('MOLHO')
    assert phy.operator('MASS TRANSPORT')
    assert phy.operator('mass transport')
    assert phy.operator('Time_Invariant')
    assert phy.operator('TIME_INVARIANT')

def test_Physics_dummy():
    dummy = {}
    dummy["output"] = ['u', 's', 'C']
    hp = {}
    hp["equations"] = {"DUMMY":dummy}
    phy = Physics(PhysicsParameter(hp))

    assert phy.input_var == ['x', 'y']
    assert phy.output_var == ['u', 's',  'C']
    assert phy.residuals == []
    assert phy.equations[0].local_output_var == {'u': 0, 's': 1, 'C': 2}
    assert len(phy.output_lb) == 3
    assert len(phy.output_ub) == 3
    assert len(phy.data_weights) == 3
    assert len(phy.pde_weights) == 0
