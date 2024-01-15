class Data:
    """
    class of data for training the PINN
    """
    def __init__(self, X={}, sol={}):
        # input to PINN
        self.X = X
        # reference solution of the output of PINN
        self.sol = sol
