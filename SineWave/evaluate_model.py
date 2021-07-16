import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

from sine_utils import run_sine_segmentation

sys.path.append(os.path.abspath('..'))
from model import LatentODEBuilder

np.random.seed(2547)

device = torch.device('cuda:0')

# Disables autograd, reduces memory usage
torch.autograd.set_grad_enabled(False)

des = "Evaluate trained model on Sine Wave data."
parser = argparse.ArgumentParser(description=des)
parser.add_argument("--model_file", type=str)
parser.add_argument("--data_file", type=str)
parser.add_argument("--n_samp", type=int, default=100)
parser.add_argument("--min_seg_len", type=int, default=10)
parser.add_argument("--K", type=float, default=100)
parser.add_argument("--n_dec", type=int, default=2)
parser.add_argument("--noise_var", type=float, default=1)
args = parser.parse_args()

data_root = Path("./Data/Test")
model_root = Path("./Models")
output_root = Path("./Results")

save_data = torch.load(model_root / Path(args.model_file))

model_args = save_data['model_args']
model = LatentODEBuilder(**model_args).build_latent_ode().to(device)
model.load_state_dict(save_data['model_state_dict'])


raw_data = torch.load(data_root / Path(args.data_file))
data = raw_data['data']

test_args = {
    'n_samp': args.n_samp,
    'min_seg_len': args.min_seg_len,
    'K': args.K,
    'n_decimal': args.n_dec,
    'noise_var': args.noise_var
}

pred_all, scores_all = run_sine_segmentation(data, model, **test_args)

results = {
    'pred_all': pred_all,
    'model_path': model_root / Path(args.model_file),
    'data_path': data_root / Path(args.data_file),
    'test_args': test_args,
}

output_root.mkdir(parents=True, exist_ok=True)
output_fn = "results_{}_{}_{}_{}".format(args.model_file, args.data_file,
                                         args.n_samp, args.min_seg_len)

torch.save(results, output_root / Path(output_fn))
