from ..parameter import PhysicsParameter
from . import EquationBase
import itertools


class Physics:
    """ All the physics in used as constraint in the PINN
    """
    def __init__(self, parameters=PhysicsParameter()):
        self.parameters = parameters

        # add all physics 
        self.equations = [EquationBase.create(eq, parameters=self.parameters.equations[eq])  for eq in self.parameters.equations] 

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

    def vel_mag(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call

        Args: 
            nn_input_var:  input tensor to the nn
            nn_output_var: output tensor from the nn
            X:  NumPy array of the inputs
        """
        uid = self.output_var.index('u')
        vid = self.output_var.index('v')
        vel = (nn_output_var[:,uid:uid+1]**2.0 + nn_output_var[:,vid:vid+1]**2.0) ** 0.5
        return vel

    def operator(self, pid):
        """ grab the pde operator

        Args:
            pid : pde operator id
        """
        pcount = 0
        for p in self.equations:
            if pcount != pid:
                pcount += 1
            else:
                return p.pde


