import deepxde as dde
from ..parameter import PhysicsParameter
from . import stressbalance
import itertools


class Physics:
    """ Physics: 
    """
    def __init__(self, parameters=PhysicsParameter()):
        self.parameters = parameters

        # add all physics 
        self.equations = [self._add_equations(eq) for eq in self.parameters.equations] 

        # update (global) input, output variable list from local_input_var and local_output_var of each equations
        self.input_var = self._update_global_variables([p.local_input_var for p in self.equations])
        self.output_var = self._update_global_variables([p.local_output_var for p in self.equations])

        # update the index in each of physics
        for p in self.equations:
            p.update_id(self.input_var, self.output_var)

        # find the min and max of the lb and ub of the output_var among all physics
        self.output_lb = []
        self.output_ub = []
        self.data_weights = []
        for k in self.output_var:
            self.output_lb.append(min([p.output_lb[k] for p in self.equations if k in p.output_lb]))
            self.output_ub.append(max([p.output_ub[k] for p in self.equations if k in p.output_ub]))
            self.data_weights.append(max([p.data_weights[k] for p in self.equations if k in p.data_weights]))

        # update residual list
        self.residuals = list(itertools.chain.from_iterable([p.residuals for p in self.equations]))
        self.pde_weights = list(itertools.chain.from_iterable([p.pde_weights for p in self.equations]))

    def _add_equations(self, eq):
        """ add equations to the model
        """
        equation = None
        if eq == "SSA":
            equation = stressbalance.SSA2DUniformB
        elif eq == "MOLHO":
            equation = stressbalance.MOLHO
        # TODO: add mass conservation

        # TODO: if eq is a class, directly add it to the physics
        if equation is not None:
            return equation(self.parameters.equations[eq])
        else:
            raise ValueError(f"Unknown equations {eq} found. Please define the physics first!")
        
    def _update_global_variables(self, local_var_list):
        """ Update global variables based on a list of local varialbes,
            find all unqiue keys, then put in one single List
        """
        # merge all dict, get all unique keys
        global_var = {}
        for d in local_var_list:
            global_var.update(d)

        return list(global_var.keys())

    def pdes(self, nn_input_var, nn_output_var):
        """ a wrapper of all the equations used in the PINN
        """
        eq = []
        for p in self.equations:
            eq += p.pde(nn_input_var, nn_output_var) 
        return eq
