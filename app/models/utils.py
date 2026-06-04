# Imports

# EarlyStopper class
class EarlyStopper:
    """
    Class used to stop the training phase if the weights start to converge.

    Attributes:
        patience (int): Number of epochs needed without any improvement to stop the training process.
        min_delta (float): Maximum variation considered as stable (without improvement).
        min_validation_loss (float): The lowest validation loss achieved. 
        _counter (int): Counter for the number of epochs with no improvement.
    """
    def __init__(self, patience: int = 1, min_delta: float = 0.0):
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.min_validation_loss: float = float('inf')

        self._counter: int = 0

    def early_stop(self, validation_loss: float) -> bool:
        """
        Decide if the training process needs to be stopped.

        Args:
            validation_loss (float): The current validation loss achieved for the current epoch.

        Returns:
            bool: Whether the training process needs to be stopped or not.
        """

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self._counter = 0

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self._counter += 1

            if self._counter >= self.patience:
                return True
            
        return False