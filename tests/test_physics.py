import PINN_ICE as pinn
import numpy as np
from PINN_ICE.physics import *
import pytest

def test_update_cid():
    ssa = SSA2DUniformB(1)
    g_var = ["u", "v", "H", "s", "a", "C"]
    ssa.update_cid(g_var)
    assert ssa.cid == [0, 1, 3, 2, 5]
    with pytest.raises(Exception):
        g_var = ["u", "v", "H", "a", "C"]
        ssa.update_cid(g_var)
