import PINN_ICE as pinn
import numpy as np
from PINN_ICE.physics import *
import pytest

def test_update_cid():
    ssa = SSA2DUniformB(1)

    # input id
    i_var = ["y", "x"]
    assert ssa.input_id == [0, 1]
    ssa.update_id(global_input_var=i_var)
    assert ssa.input_id == [1, 0]

    # output id
    o_var = ["u", "v", "H", "s", "a", "C"]
    assert ssa.output_id == [0, 1, 2, 3, 4]
    ssa.update_id(global_output_var=o_var)
    assert ssa.output_id == [0, 1, 3, 2, 5]

    # Check exceptions
    with pytest.raises(Exception):
        g_var = ["u", "v", "H", "a", "C"]
        ssa.update_id(global_output_var=g_var)
