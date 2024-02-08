from abc import ABC, abstractmethod
from .utils import data_misfit


class ParameterBase(ABC):
    """ Abstract class of parameters in the experiment
    """
    def __init__(self, param_dict):
        self.param_dict = param_dict

        # default
        self.set_default()

        # update parameters
        self.set_parameters(self.param_dict)

        # check consistency
        self.check_consisteny()

    def __str__(self):
        """ display all attributes except 'param_dict'
        """
        return "\t" + type(self).__name__ + ": \n" + \
                ("\n".join(["\t\t" + k + ":\t" + str(self.__dict__[k]) for k in self.__dict__ if k != "param_dict"]))+"\n"

    @abstractmethod
    def set_default(self):
        """ set default values
        """
        pass

    @abstractmethod
    def check_consisteny(self):
        """ check consistency of the parameter data
        """
        pass

    def _add_parameters(self, pdict: dict):
        """ add all the keys from pdict to the class, with their values
        """
        if isinstance(pdict, dict):
            for key, value in pdict.items():
                setattr(self, key, value)

    def set_parameters(self, pdict: dict):
        """ find all the keys from pdict which are avalible in the class, update the values
        """
        if isinstance(pdict, dict):
            for key, value in pdict.items():
                # only update attribute the key
                if hasattr(self, key):
                    setattr(self, key, value)

    def has_keys(self, keys):
        """ if all the keys are in the class, return true, otherwise return false
        """
        if isinstance(keys, dict) or isinstance(keys, list):
            return all([hasattr(self, k) for k in keys])
        else:
            return False


class DomainParameter(ParameterBase):
    """
    parameters of domain
    """
    def __init__(self, param_dict={}):
        super().__init__(param_dict)

    def set_default(self):
        # shape file to define the outer boundary of the domain
        self.shapefile = None
        # number of collocation points used in the domain
        self.num_collocation_points = 0

    def check_consisteny(self):
        pass


class DataParameter(ParameterBase):
    """
    parameters of data
    """
    def __init__(self, param_dict={}):
        super().__init__(param_dict)

    def set_default(self):
        """ default settings
        """
        # file path
        self.data_path = ""
        # length of each data in used, leave no data variable(sol) empty or set to None
        self.data_size = {}
        # source of the data
        self.source = "ISSM"

    def check_consisteny(self):
        if self.source != "ISSM":
            raise ValueError(f"Data loader of {self.source} is not implemented")


class NNParameter(ParameterBase):
    """
    parameters of nn
    """
    def __init__(self, param_dict={}):
        super().__init__(param_dict)

    def set_default(self):
        """
        default values:
        """
        # nn architecture
        self.input_variables = []
        self.output_variables = []
        self.num_neurons = 0
        self.num_layers = 0
        self.activation = "tanh"
        self.initializer = "Glorot uniform"

        #  scaling parameters
        self.input_lb = None
        self.input_ub = None
        self.output_lb = None
        self.output_ub = None

    def set_parameters(self, pdict: dict):
        super().set_parameters(pdict)
        self.input_size = len(self.input_variables)
        self.output_size = len(self.output_variables)

    def check_consisteny(self):
        # input size of nn equals to dependent in physics
        if self.input_size != len(self.input_variables):
            raise ValueError("'input_size' does not match the number of 'input_variables'")
        # out size of nn equals to variables in physics
        if self.output_size != len(self.output_variables):
            raise ValueError("'output_size' does not match the number of 'output_variables'")
        pass

    def is_input_scaling(self):
        """
        if the input boundaries are provided
        """
        if (self.input_lb is not None) and (self.input_ub is not None):
            return True
        else:
            return False

    def is_output_scaling(self):
        """
        if the output boundaries are provided
        """
        if (self.output_lb is not None) and (self.output_ub is not None):
            return True
        else:
            return False


class PhysicsParameter(ParameterBase):
    """ parameter of physics
    """
    def __init__(self, param_dict={}):
        super().__init__(param_dict)
        self.setup_equations()

    def set_default(self):
        # name(s) and parameters of the equations
        self.equations = {}

    def check_consisteny(self):
        pass

    def setup_equations(self):
        """ translate the input dict to subclass of EquationParameter(), and save back to the values in self.equations
        """
        self.equations = {k:EquationParameter.create(k, param_dict = self.equations[k]) for k in self.equations}

    def __str__(self):
        """
        display all equations
        """
        return "\t" + type(self).__name__ + ": \n" + \
                ("\n".join(["\t\t" + k + ":\n" + str(self.equations[k]) for k in self.equations]))+"\n"

class EquationParameter(ParameterBase):
    """ parameter of equations
    """
    subclasses = {}
    def __init__(self, param_dict={}):
        super().__init__(param_dict)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls._EQUATION_TYPE] = cls

    @classmethod
    def create(cls, equation_type,  **kwargs):
        if equation_type not in cls.subclasses:
            raise ValueError(f"Equation type {format(equation_type)} is not defined")
        return cls.subclasses[equation_type](**kwargs)

    def set_default(self):
        # list of input names
        self.input = []
        # list of output names
        self.output = []
        # lower and upper bound of output 
        self.output_lb = []
        self.output_ub = []
        # weights of each output 
        self.data_weights = []
        # names of residuals
        self.residuals = [] 
        # pde weights
        self.pde_weights = []
        # scalar variables: name:value
        self.scalar_variables = {}

    def check_consisteny(self):
        if (len(self.output)) != (len(self.output_lb)):
            raise ValueError("Size of 'output' does not match the size of 'output_lb'")
        if (len(self.output)) != (len(self.output_ub)):
            raise ValueError("Size of 'output' does not match the size of 'output_ub'")
        if any([l>=u for l,u in zip(self.output_lb, self.output_ub)]):
            raise ValueError("output_lb is not smaller than output_ub")
        if (len(self.output)) != (len(self.data_weights)):
            raise ValueError("Size of 'output' does not match the size of 'data_weights'")

        # check the pde weights
        if isinstance(self.pde_weights, list):
            if len(self.pde_weights) != len(self.residuals):
                raise ValueError("Length of pde_weights does not match the length of residuals")
        else:
            raise ValueError("pde_weights is not a list")
        
    def set_parameters(self, pdict: dict):
        """ overwrite the default function, so that for 'scalar_parameters', only update the dict
        """
        if isinstance(pdict, dict):
            for key, value in pdict.items():
                # only update attribute the key
                if hasattr(self, key):
                    # only update the dictionary, not overwirte
                    if isinstance(value, dict):
                        old_dict = getattr(self, key)
                        old_dict.update(value)
                        setattr(self, key, old_dict)
                    else:
                        setattr(self, key, value)

    def __str__(self):
        """
        display all attributes except 'param_dict'
        """
        return ("\n".join(["\t\t\t" + k + ":\t" + str(self.__dict__[k]) for k in self.__dict__ if k != "param_dict"]))+"\n"


class TrainingParameter(ParameterBase):
    """ parameter of training
    """
    def __init__(self, param_dict={}):
        super().__init__(param_dict)

        if self.additional_loss:
            self.update_parameters()

    def set_default(self):
        self.epochs = 0
        self.optimizer = "adam"
        self.loss_functions = "MSE"
        self.additional_loss = {} 
        self.learning_rate = 0
        self.loss_weights = []
        self.save_path = ""
        self.is_save = True
        self.is_plot = False

    def check_consisteny(self):
        pass

    def update_parameters(self):
        """ convert dict to class LossFunctionParameter
        """
        self.additional_loss = {k:LossFunctionParameter(self.additional_loss[k]) for k in self.additional_loss}

class LossFunctionParameter(ParameterBase):
    """ parameter of customize loss function
    """
    def __init__(self, param_dict={}):
        super().__init__(param_dict)

    def set_default(self):
        self.name = ""
        self.function = "MSE"
        self.weight = 1.0

    def check_consisteny(self):
        data_misfit.get(self.function)

class Parameters(ParameterBase):
    """ parameters of the pinn, including domain, data, nn, and physics
    """
    def __init__(self, param_dict={}):
        super().__init__(param_dict)

        self.update_parameters()

    def __str__(self):
        return "Parameters: \n" + str(self.training) + str(self.domain) + str(self.data) + str(self.nn) + str(self.physics)

    def set_default(self):
        self.training = TrainingParameter()
        self.domain = DomainParameter()
        self.data = DataParameter()
        self.nn = NNParameter()
        self.physics = PhysicsParameter()

    def set_parameters(self, param_dict):
        self.training = TrainingParameter(param_dict)
        self.domain = DomainParameter(param_dict)
        self.data = DataParameter(param_dict)
        self.nn = NNParameter(param_dict)
        self.physics = PhysicsParameter(param_dict)

    def check_consisteny(self):
        # length of training.loss_weights equals to equations+datasize
        #if (any(x not in self.nn.output_variables for x in self.data.datasize)):
        # TODO:    raise ValueError("names in 'datasize' does not match the name in 'output_variables'")
        pass

    def update_parameters(self):
        """
        update parameters according to the input
        """
        # set the size of variables not given in data to None
        for x in self.nn.output_variables:
            if x not in self.data.datasize:
                self.data.datasize[x] = None

        # set component id in data according to the order in physics
        pass
