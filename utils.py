import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_f(x, device=DEVICE):
    """Send numpy ndarray to GPU.

    Converts numpy ndarray to a PyTorch float tensor, and sends it to the GPU.

    Args:
        x (np.ndarray): Numpy ndarray to move to GPU.
        device (torch.device): PyTorch device which input data is move to.

    Returns:
        torch.Tensor: PyTorch representation of input, located on the GPU.
    """
    if isinstance(x, torch.Tensor):
        return x.float().to(device)

    return torch.tensor(x).float().to(device)


def to_np(x):
    """Convert GPU based tensor to numpy array.

    Detaches a PyTorch tensor from the computation graph, moves it to the CPU,
    and converts the array to a numpy ndarray.

    Args:
        x (torch.Tensor): PyTorch tensor to convert to numpy ndarray.

    Returns:
        np.ndarray: Numpy representation of input tensor.
    """
    return x.detach().cpu().numpy()


class RunningAverageMeter(object):
    """Compute and stores the average and current value.

    This implementation was taken from the original Neural ODE
    repository: https://github.com/rtqichen/torchdiffeq.
    """

    def __init__(self, momentum=0.99):
        """Initialize RunningAverageMeter.

        Args:
            momentum (float, optional): Momentum coefficient. Defaults to 0.99.
        """
        self.momentum = momentum
        self.val = None
        self.avg = 0

    def reset(self):
        """Reset running average to zero."""
        self.val = None
        self.avg = 0

    def update(self, val):
        """Update running average with new value.

        Args:
            val (float): New value.
        """
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
