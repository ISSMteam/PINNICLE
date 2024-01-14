from abc import ABC, abstractmethod

class parameterBase(ABC): 
    """
    Abstract class of parameters in the experiment
    """
    def __init__(self, param_dict):
        self.param_dict = param_dict

        # default
        self.set_default()

        # update parameters
        self.set_parameters(self.param_dict)

        # check consistency
        self.check_consisteny()

    @abstractmethod
    def set_default(self):
        """
        set default values
        """
        pass

    @abstractmethod
    def check_consisteny(self):
        """
        check consistency of the parameter data
        """
        pass

    def _add_parameters(self, pdict: dict):
        """
        add all the keys from pdict to the class, with their values
        """
        if isinstance(pdict, dict):
            for key, value in pdict.items():
                setattr(self, key, value)
    
    def set_parameters(self, pdict: dict):
        """
        find all the keys from pdict which are avalible in the class, update the values
        """
        if isinstance(pdict, dict):
            for key, value in pdict.items():
                # only update attribute the key
                if hasattr(self, key):
                    setattr(self, key, value)

    def has_keys(self, keys):
        """
        if all the keys are in the class, return true, otherwise return false
        """
        if isinstance(keys, dict) or isinstance(keys, list): 
            return all([hasattr(self,k) for k in keys])
        else:
            return False

class domain_parameter(parameterBase):
    """
    parameters of domain
    """
    def __init__(self, param_dict={}):
        super().__init__(param_dict)

    def set_default(self):
        # shape file to define the outer boundary of the domain
        self.shapefile = None

    def check_consisteny(self):
        pass

class data_parameter(parameterBase):
    """
    parameters of data
    """
    def __init__(self, param_dict={}):
        super().__init__(param_dict)

    def set_default(self):
        # name list of the data used in PINN
        self.name = []
        # length of each data in used
        self.size = []

    def check_consisteny(self):
        if len(self.name) == len(self.size):
            pass
        else:
            raise SyntaxError("The length of datanames does not match datalength!")


class nn_parameter(parameterBase):
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
        self.input_size = 2
        self.output_size = 0
        self.num_neurons = 0
        self.num_layers = 0
        self.activation = "tanh"
        self.initializer = "Glorot uniform"

        #  scaling parameters
        self.input_lb = None
        self.input_ub = None
        self.output_lb = None
        self.output_ub = None

    def check_consisteny(self):
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

