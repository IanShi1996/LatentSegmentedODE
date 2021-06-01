import sys, os
from torch.utils.data import Dataset

sys.path.append(os.path.abspath('..'))
from segment import segment, get_changepoints
from utils import gpu_f, to_np

class SineSet(Dataset):
    def __init__(self, data, time):
        self.data = data
        self.time = time
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.time
    
def run_sine_segmentation(data, model, n_samp, min_seg_len, K, n_decimal, noise_var):
    """Segmentation main loop for Sine Wave data.
    
    Runs segmentation and persists results.

    Args:
        data (dict): Dict containing data, timepoints, and true changepoints.
        model (nn.Module): PyTorch Latent NODE model.
        n_samp (int): Number of MC samples to use to estimate logpx.
        min_seg_len (int): Minimum length of valid segment for PELT.
        K (float): Relaxation term on PELT penalty.
        n_decimal: Decimal precision used for parallel estimation step.
        noise_var (float, optional): Fixed variance for likelihood calculation.
        
    Returns:
        (list, list): List of predicted changepoints and score matrices.
    """
    pred_all = []
    scores_all = []

    for pack in data:
        d = pack[0]
        tp = pack[1]
        true_cp = pack[2]

        scores = segment(gpu_f(d), gpu_f(tp), model, n_samp, min_seg_len, K, n_decimal, noise_var)
        pred_cp = get_changepoints(scores)

        scores_all.append(scores)
        pred_all.append(pred_cp)

        print('True cp:', true_cp, flush=True)
        print('Predicted cp:', pred_cp, '\n', flush=True)

    return pred_all, scores_all