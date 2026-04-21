# Imports

# EarlyStopper class
class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0.0):
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.min_validation_loss: float = float('inf')

        self._counter: int = 0

    def early_stop(self, validation_loss: float) -> bool:

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self._counter = 0

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self._counter += 1

            if self._counter >= self.patience:
                return True
            
        return False