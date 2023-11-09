import torch
import numpy as np
from utils.others import check_same_shape, convert_to_tensor


class Huber:
    """
    Computes the huber loss between `y_true` and `y_pred`.

    Args:
        y_true: Tensor of Ground truth values. shape = [batch_size, d0, .., dN]
        y_pred: Tensor of Predicted values. shape = [batch_size, d0, .., dN]
        delta: A float, the point where the Huber loss function changes from a
                quadratic to linear. default: `1.0`

    Returns:
        Tensor of Huber loss
    """

    def __call__(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, delta: float = 1.0
    ) -> torch.Tensor:
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)

        check_same_shape(y_pred, y_true)

        abs_error = torch.abs(y_pred - y_true)
        quadratic = torch.clamp(abs_error, max=delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic.pow(2) + delta * linear
        return loss.mean()
    
    def __str__(self):
        return "Huber Loss"


class LogCoshError:
    """
    Computes Logarithm of the hyperbolic cosine of the prediction error.

    Args:
        y_true: Tensor of Ground truth values. shape = [batch_size, d0, .., dN]
        y_pred: Tensor of Predicted values. shape = [batch_size, d0, .., dN]

    Returns:
        Tensor of Logcosh error
    """

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)

        check_same_shape(y_pred, y_true)

        diff = y_pred - y_true
        return torch.mean(torch.log((torch.exp(diff) + torch.exp(-1.0 * diff)) / 2.0))
    
    def __str__(self):
        return "LogCosh Error"
    

class MeanAbsoluteError:
    def __call__(self, tensor1, tensor2):
        """
        Arguments
        ---------
        x : torch.Tensor
        y : torch.Tensor
        """
        return torch.mean(torch.abs(convert_to_tensor(tensor1) - convert_to_tensor(tensor2)))
    
    def __str__(self):
        return "Mean Absoulute Error"
    
    
class MeanIoU:
    def __init__(self):
        self.epsilon = 1e-10

    def __call__(self, tensor1, tensor2):
        tensor1 = convert_to_tensor(tensor1)
        tensor2 = convert_to_tensor(tensor2)
        # if single dimension
        if len(tensor1.shape) == 1 and len(tensor2.shape) == 1:
            inter = torch.sum(torch.squeeze(tensor1 * tensor2))
            union = torch.sum(torch.squeeze(tensor1 + tensor2)) - inter
        else:
            inter = torch.sum(
                torch.sum(torch.squeeze(tensor1 * tensor2, axis=3), axis=2), axis=1
            )
            union = (
                torch.sum(
                    torch.sum(torch.squeeze(tensor1 + tensor2, axis=3), axis=2), axis=1
                )
                - inter
            )
        return torch.mean((inter + self.epsilon) / (union + self.epsilon))
    
    def __str__(self):
        return "MeanIou Loss"
    


class MeanSquaredError:
    def __call__(self, tensor1, tensor2):
        """
        Arguments
        ---------
        x : torch.Tensor
        y : torch.Tensor
        """
        tensor1 = convert_to_tensor(tensor1)
        tensor2 = convert_to_tensor(tensor2)
        return torch.mean((tensor1 - tensor2) ** 2)
    
    def __str__(self):
        return "Mean Squared Error"
    

class MeanSquaredLogarithmicError:
    """
    Computes the mean squared logarithmic error between `y_true` and `y_pred`.

    Args:
        y_true: Tensor of Ground truth values. shape = [batch_size, d0, .., dN]
        y_pred: Tensor of Predicted values. shape = [batch_size, d0, .., dN]

    Returns:
        Tensor of mean squared logarithmic error
    """

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)

        check_same_shape(y_pred, y_true)

        squared_log = torch.pow(torch.log1p(y_pred) - torch.log1p(y_true), 2)

        return torch.mean(squared_log)
    
    def __str__(self):
        return "Mean Squared Logarithmic Error"
    

class RootMeanSquaredError:
    def __call__(self, tensor1, tensor2):
        """
        Returns the root mean squared error (RMSE) of two tensors.

        Arguments
        ---------
        x : torch.Tensor
        y : torch.Tensor
        """
        tensor1 = convert_to_tensor(tensor1)
        tensor2 = convert_to_tensor(tensor2)
        return torch.sqrt(torch.mean((tensor1 - tensor2) ** 2))
    
    def __str__(self):
        return "Root Mean Squared Error"
    
    
def corrcoef(tensor1, tensor2):
    """
    Arguments
    ---------
    x : torch.Tensor
    y : torch.Tensor
    """
    xm = tensor1.sub(torch.mean(tensor1))
    ym = tensor2.sub(torch.mean(tensor2))
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


class RSquared:
    def __init__(self):
        self.corrcoef = corrcoef

    def __call__(self, tensor1, tensor2):
        """
        Arguments
        ---------
        x : torch.Tensor
        y : torch.Tensor
        """
        tensor1 = convert_to_tensor(tensor1)
        tensor2 = convert_to_tensor(tensor2)
        return (corrcoef(tensor1, tensor2)) ** 2
    
    def __str__(self):
        return "RSquared Error"
    
    
class StdDeviation:
    def __call__(self, tensor1, tensor2):
        tensor1 = convert_to_tensor(tensor1)
        tensor2 = convert_to_tensor(tensor2)
        return torch.Tensor(np.array([np.sqrt(np.var(tensor1.numpy()-tensor2.numpy()))]))[0]
    
    def __str__(self):
        return "Standard Deviation of Error"
