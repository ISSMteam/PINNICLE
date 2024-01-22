import deepxde as dde
from ..parameters import PhysicsParameter
from . import stressbalance
import itertools


class Physics:
    """ Physics: 
    """
    def __init__(self, parameters=PhysicsParameter()):
        self.parameters = parameters

        # add all physics to a list
        self.physics = [self._add_physics(eq, parameters) for eq in parameters.equations] 

        # update input, output variable list for nn

        # update residual list
        self.residuals = list(itertools.chain.from_iterable([p.residuals for p in self.physics]))

    def _add_physics(self, eq, parameters):
        if eq == "SSA":
            return stressbalance.SSA2DUniformB(parameters.scalar_variables["B"])
        if eq == "MOLHO":
            return stressbalance.MOLHO(parameters.scalar_variables["B"])

    def equations(self, nn_input_var, nn_output_var):
        """ a wrapper of all the equations used in the PINN
        """
        eq = []
        for p in self.physics:
            eq += p.pde(nn_input_var, nn_output_var) 
        return eq
