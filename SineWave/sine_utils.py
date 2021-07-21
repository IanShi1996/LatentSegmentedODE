import sys
import os
from torch.utils.data import Dataset

sys.path.append(os.path.abspath('..'))
from segment import segment, get_changepoints
from utils import gpu_f


class SineSet(Dataset):
    """Custom PyTorch dataset for sine wave data."""
    def __init__(self, data, time):
        """Initialize SineSet.

        Args:
            data (torch.Tensor): Dataset.
            time (torch.Tensor): Timepoints.
        """
        self.data = data
        self.time = time

        self.lengths = [len(d) for d in data]

    def __len__(self):
        """Return length of dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get data and timepoints by index."""
        return self.data[idx], self.time, self.lengths[idx]


def run_sine_segmentation(data, model, n_samp, min_seg, K, n_dec, noise_var):
    """Segmentation main loop for Sine Wave data.
    
    Runs segmentation and persists results.

    Args:
        data (dict): Dict containing data, timepoints, and true changepoints.
        model (nn.Module): PyTorch Latent NODE model.
        n_samp (int): Number of MC samples to use to estimate log_px.
        min_seg (int): Minimum length of valid segment for PELT.
        K (float): Relaxation term on PELT penalty.
        n_dec: Decimal precision used for parallel estimation step.
        noise_var (float, optional): Fixed variance for likelihood calculation.
        
    Returns:
        (list, list): List of predicted changepoints and score matrices.
    """
    pred_all = []
    scores_all = []

    for pack in data:
        d = gpu_f(pack[0])
        tp = gpu_f(pack[1])
        true_cp = pack[2]

        scores = segment(d, tp, model, n_samp, min_seg, K, n_dec, noise_var)
        pred_cp = get_changepoints(scores)

        scores_all.append(scores)
        pred_all.append(pred_cp)

        print('True cp:', true_cp, flush=True)
        print('Predicted cp:', pred_cp, '\n', flush=True)

    return pred_all, scores_all
