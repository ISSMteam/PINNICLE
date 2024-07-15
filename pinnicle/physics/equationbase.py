from abc import ABC, abstractmethod
from deepxde.backend import backend_name
from ..parameter import EquationParameter
from . import Constants

class EquationBase(ABC, Constants):
    """ base class of all the equations
    """
    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls._EQUATION_TYPE] = cls

    @classmethod
    def create(cls, equation_type,  **kwargs):
        if equation_type not in cls.subclasses:
            raise ValueError(f"Equation type {format(equation_type)} is not defined")
        return cls.subclasses[equation_type](**kwargs)

    def __init__(self, parameters=EquationParameter()):
        # load constants first
        Constants.__init__(self)

        # get the setting parameters 
        self.parameters = parameters

        # update parameters in the equation accordingly
        self.update_parameters(self.parameters)

        # update scalar variables
        self.update_scalars(self.parameters.scalar_variables)

        # set pde
        if backend_name in ["tensorflow.compat.v1", "tensorflow", "paddle", "pytorch"]:
            self.pde = self._pde
        elif backend_name == "jax":
            self.pde = self._pde_jax
        else:
            raise ValueError(f"Backeend {backend_name} is not defined")

    def get_input_list(self):
        """ get the List of names of input variables
        """
        return list(self.local_input_var.keys())

    def get_output_list(self):
        """ get the List of names of output variables
        """
        return list(self.local_output_var.keys())

    def update_id(self, global_input_var=None, global_output_var=None):
        """ update component id, always remeber to call this in compiling the model

        Args:
            global_input_var: List of input_variables to nn, these variables are 
                shared across all the physics
            global_output_var: List of output_variables from nn, these variables 
                are shared across all the physics
        """
        if global_input_var is not None:
            self.local_input_var = {o:global_input_var.index(o) for o in self.local_input_var}
        if global_output_var is not None:
            self.local_output_var = {o:global_output_var.index(o) for o in self.local_output_var}

    def update_parameters(self, parameters):
        """ update attributes of the class using EquationParameter
        """
        # Dict of dependent and independent variables of the model, the values are
        # the global component id in the Physics, these two dicts are maps from local 
        # to global, current indices are temporary, they will be updated after all equations are set
        self.local_input_var = {k:i for i,k in enumerate(parameters.input)}        # x, y, z, t, etc.
        self.local_output_var = {k:i for i,k in enumerate(parameters.output)}      # u, v, s, H, etc.

        # lower and upper bounds of the output in [SI] unit, with keys of the variable name
        self.output_lb = {k: parameters.output_lb[i] for i,k in enumerate(parameters.output)}
        self.output_ub = {k: parameters.output_ub[i] for i,k in enumerate(parameters.output)}

        # weights to scale the data misfit to 1 in [SI]
        self.data_weights = {k: parameters.data_weights[i] for i,k in enumerate(parameters.output)}

        # residuals name list
        self.residuals = parameters.residuals
        # pde weights
        self.pde_weights = parameters.pde_weights

    def update_scalars(self, scalar_variables: dict):
        """ update scalars in the equations
        """
        if isinstance(scalar_variables, dict):
            for key, value in scalar_variables.items():
                setattr(self, key, value)
    @abstractmethod
    def _pde(self, nn_input_var, nn_output_var):
        """ pde function used in deepxde
        """
        return
