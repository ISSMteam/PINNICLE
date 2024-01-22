import deepxde as dde
from ..parameters import PhysicsParameter
from . import stressbalance
import itertools


class Physics:
    """ Physics: 
    """
    def __init__(self, parameters=PhysicsParameter()):
        self.parameters = parameters

        # add all physics 
        self.physics = [self._add_physics(eq, parameters) for eq in parameters.equations] 

        # update input, output variable list for nn
        self.input_var = self._update_global_variables([p.input_var for p in self.physics])
        self.output_var = self._update_global_variables([p.output_var for p in self.physics])

        # update the index in the nn

        # update residual list
        self.residuals = list(itertools.chain.from_iterable([p.residuals for p in self.physics]))

    def _add_physics(self, eq, parameters):
        """ add physics to the model
        """
        if eq == "SSA":
            return stressbalance.SSA2DUniformB(parameters.scalar_variables["B"])
        if eq == "MOLHO":
            return stressbalance.MOLHO(parameters.scalar_variables["B"])
        # TODO: add mass conservation
        
        raise ValueError(f"Unknown equations {eq} found. Please define the physics first!")
        
    def _update_global_variables(self, local_var_list):
        """ Update global variables based on a list of local varialbes,
            find all unqiue keys, then assign ids to them, from 0 to n
        """
        # merge all dict
        global_var = {}
        for d in local_var_list:
            global_var.update(d)

        # assign ids
        return {k:i for i,k in enumerate(global_var.keys())}

    def equations(self, nn_input_var, nn_output_var):
        """ a wrapper of all the equations used in the PINN
        """
        eq = []
        for p in self.physics:
            eq += p.pde(nn_input_var, nn_output_var) 
        return eq
