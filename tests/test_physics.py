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
