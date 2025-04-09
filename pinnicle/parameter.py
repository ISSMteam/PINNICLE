from abc import ABC, abstractmethod
from .utils import data_misfit


class ParameterBase(ABC):
    """ Abstract class of parameters in the experiment
    """
    def __init__(self, param_dict):
        self.param_dict = param_dict

        # set default parameters 
        self.set_default()

        # set parameters from param_dict if given
        self.set_parameters(self.param_dict)

        # make some necessary update of the parameters after loading from param_dict
        self.update()

        # check consistency
        self.check_consistency()

    @abstractmethod
    def set_default(self):
        """ set default values
        """
        pass

    @abstractmethod
    def check_consistency(self):
        """ check consistency of the parameter data
        """
        pass

    def _add_parameters(self, pdict: dict):
        """ add all the keys from pdict to the class, with their values
        """
        if isinstance(pdict, dict):
            for key, value in pdict.items():
                setattr(self, key, value)

    def has_keys(self, keys):
        """ if all the keys are in the class, return true, otherwise return false
        """
        if isinstance(keys, dict) or isinstance(keys, list):
            return all([hasattr(self, k) for k in keys])
        else:
            return False

    def set_parameters(self, pdict: dict):
        """ find all the keys from pdict which are avalible in the class, update the values
        """
        if isinstance(pdict, dict):
            for key, value in pdict.items():
                # only update attribute the key
                if hasattr(self, key):
                    setattr(self, key, value)

    def __str__(self):
        """ display all attributes except 'param_dict'
        """
        return "\t" + type(self).__name__ + ": \n" + \
                ("\n".join(["\t\t" + k + ":\t" + str(self.__dict__[k]) for k in self.__dict__ if k != "param_dict"]))+"\n"

    def update(self):
        """ after set_parameter, make some necessary update of the parameters
        """
        pass


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
        # static or time dependent problem
        self.time_dependent = False
        # start and end time
        self.start_time = 0
        self.end_time = 0

    def check_consistency(self):
        """ need to provide start and end time if solving a time dependent problem
        """
        if self.time_dependent:
            if self.start_time >= self.end_time:
                raise ValueError(f"'start_time' at {self.start_time} is ahead of 'end_time' at {self.end_time}")


class DataParameter(ParameterBase):
    """ list of all data used
    """
    def __init__(self, param_dict={}):
        super().__init__(param_dict)

    def set_default(self):
        """ default parameters
        """
        self.data = {}

    def check_consistency(self):
        pass

    def __str__(self):
        """
        display all data
        """
        return "\t" + type(self).__name__ + ": \n" + \
                ("\n".join(["\t\t" + k + ":\n" + str(self.data[k]) for k in self.data]))+"\n"

    def update(self):
        """ convert dict to class SingleDataParameter
        """
        if self.data:
            self.data = {k:SingleDataParameter(self.data[k]) for k in self.data}


class SingleDataParameter(ParameterBase):
    """ parameters of a single data file
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
        # name map k->v, k is the variable name in the PINN, v is the variable name in the data file
        self.name_map = {}
        # X name map k->v, k is the input names in the PINN, v is the coordinates name in the data file
        self.X_map = {}
        # source of the data
        self.source = "ISSM"
        # default time point, None means not in used
        self.default_time = None

    def check_consistency(self):
        if self.source not in ["ISSM", "mat", "h5"]:
            raise ValueError(f"Data loader of {self.source} is not implemented")

    def __str__(self):
        """
        display all attributes except 'param_dict'
        """
        return ("\n".join(["\t\t\t" + k + ":\t" + str(self.__dict__[k]) for k in self.__dict__ if k != "param_dict"]))+"\n"

    def update(self):
        """ update name_map according to data_size
        """
        # if the X coordinates names are not given, then use default
        if not self.X_map:
            self.X_map["x"] = "x"
            self.X_map["y"] = "y"
            self.X_map["t"] = "t"

        # every variable in data_size need to be loaded from the data loader
        for k in self.data_size:
            # names in data_size, if not given in name_map, then use the same name for key and value
            if k not in self.name_map:
                self.name_map[k] = k


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

        # fourier feature transform
        self.fft = False
        self.num_fourier_feature = 10
        self.sigma = 1.0
        self.B = None

        # parallel neural network
        self.is_parallel = False

        #  scaling parameters
        self.input_lb = None
        self.input_ub = None
        self.output_lb = None
        self.output_ub = None

    def check_consistency(self):
        if self.fft:
            if self.input_size != self.num_fourier_feature*2:
                raise ValueError("'input_size' does not match the number of fourier feature")
            if self.B is not None:
                if not isinstance(self.B, list):
                    raise TypeError("'B' matrix need to be input in a list")
                if len(self.B[0]) != self.num_fourier_feature:
                    raise ValueError("Number of columns of 'B' matrix does not match the number of fourier feature")
        else:
            # input size of nn equals to dependent in physics
            if self.input_size != len(self.input_variables):
                raise ValueError("'input_size' does not match the number of 'input_variables'")
        # out size of nn equals to variables in physics
        if self.output_size != len(self.output_variables):
            raise ValueError("'output_size' does not match the number of 'output_variables'")
        pass

    def is_input_scaling(self):
        """
        if the input boundaries are provided, or fourier feature transform is used
        """
        if ((self.input_lb is not None) and (self.input_ub is not None)) or self.fft:
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

    def set_parameters(self, pdict: dict):
        super().set_parameters(pdict)
        self.input_size = len(self.input_variables)
        self.output_size = len(self.output_variables)
        # num_eurons is list 
        if isinstance(self.num_neurons, list):
            self.num_layers = len(self.num_neurons)

    def update(self):
        """ update the input_size for fourier feature transform
        """
        if self.fft:
            self.input_size = self.num_fourier_feature*2

class PhysicsParameter(ParameterBase):
    """ parameter of physics
    """
    def __init__(self, param_dict={}):
        super().__init__(param_dict)
        self.setup_equations()

    def set_default(self):
        # name(s) and parameters of the equations
        self.equations = {}

    def check_consistency(self):
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

    def check_consistency(self):
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


    def set_default(self):
        # number of epochs
        self.epochs = 0
        # optimization method
        self.optimizer = "adam"
        # general loss function
        self.loss_functions = "MSE"
        # additional loss functions, specified as a dict
        self.additional_loss = {} 
        # learning rate
        self.learning_rate = 0.001
        # decay steps
        self.decay_steps = 0
        # decay rate
        self.decay_rate = 0.0
        # list of the weights
        self.loss_weights = []
        # setting the callbacks
        self.has_callbacks = False
        # dde.callbacks.EarlyStopping(min_delta=min_delta, patience=patience)
        self.min_delta = None
        self.patience = None
        # dde.callbacks.PDEPointResampler(period=period)
        self.period = None
        # dde.callbacks.ModelCheckpoint(filepath, verbose=1, save_better_only=True)
        self.checkpoint = False
        # path to save the results
        self.save_path = "./"
        # if save the results and history
        self.is_save = True
        # if plot the results and history, and save figures
        self.is_plot = False

    def check_consistency(self):
        pass

    def check_callbacks(self):
        """ check if any of the following variable is given from setting
        """
        # EarlyStopping
        if self.has_EarlyStopping():
            return True
        # PDEPointResampler
        if self.has_PDEPointResampler():
            return True
        # ModelCheckpoint
        if self.has_ModelCheckpoint():
            return True
        # otherwise
        return False

    def has_EarlyStopping(self):
        """ check if param has the min_delta or patience for early stopping
        """
        has_es = False
        if self.min_delta is not None:
            has_es = True

        if self.patience is not None:
            has_es = True

        # update the setting with partially None
        if has_es:
            if self.min_delta is None:
                self.min_delta = 0
            if self.patience is None:
                self.patience = 0

        return has_es

    def has_ModelCheckpoint(self):
        """ check if param has checkpoint=True for checkpointing
        """
        return self.checkpoint

    def has_PDEPointResampler(self):
        """ check if param has the period for resampler
        """
        if self.period is None:
            return False
        else:
            return True

    def update(self):
        """ convert dict to class LossFunctionParameter
        """
        # update additional loss
        if self.additional_loss:
            self.additional_loss = {k:LossFunctionParameter(self.additional_loss[k]) for k in self.additional_loss}

        #  add callback setttings if given any of them
        self.has_callbacks = self.check_callbacks()


class LossFunctionParameter(ParameterBase):
    """ parameter of customize loss function
    """
    def __init__(self, param_dict={}):
        super().__init__(param_dict)

    def set_default(self):
        # name of the loss term, should avoid using existing names in the system, 
        # TODO: make sure this name is not in used
        self.name = ""
        # loss functions
        self.function = "MSE"
        # weight of this loss function
        self.weight = 1.0

    def check_consistency(self):
        data_misfit.get(self.function)


class Parameters(ParameterBase):
    """ parameters of the pinn, including domain, data, nn, and physics
    """
    def __init__(self, param_dict={}):
        super().__init__(param_dict)

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

    def check_consistency(self):
        # length of training.loss_weights equals to equations+datasize
        #if (any(x not in self.nn.output_variables for x in self.data.datasize)):
        # TODO:    raise ValueError("names in 'datasize' does not match the name in 'output_variables'")
        pass

    def update(self):
        """
        update parameters according to the input
        """
        # set the size of variables not given in data to None
        for x in self.nn.output_variables:
            if x not in self.data.datasize:
                self.data.datasize[x] = None

        pass
