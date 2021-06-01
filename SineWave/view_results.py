import argparse
import sys, os

from pathlib import Path

import torch
torch.autograd.set_grad_enabled(False)

import numpy as np

sys.path.append(os.path.abspath('..'))
from metrics import randindex, f1_score, hausdorff, annotation_error, evaluate_metrics
from utils import to_np, gpu_f

device = torch.device('cuda:0')

parser = argparse.ArgumentParser(description="Evaluate Sine Wave results.")
parser.add_argument('--results_file', type=str)
args = parser.parse_args()

results_root = Path("./Results/")
results_path = Path(args.results_file)

results_data = torch.load(results_root / results_path)

test_dataset = torch.load(results_data['data_path'])['data']

data = [d[0] for d in test_dataset]
tps = [d[1] for d in test_dataset]
true_cps = [d[2]. astype(int) for d in test_dataset]
pred_cps = results_data['pred_all']

rr_pred = [list(pred_cps[i]) + [len(tps[i])] for i in range(len(pred_cps))]
rr_true = [list(true_cps[i]) + [len(tps[i])] for i in range(len(true_cps))]

metrics = [randindex, f1_score, hausdorff, annotation_error]
seg_node_data = evaluate_metrics(metrics, rr_pred, rr_true)

for k, v in seg_node_data.items():
    print('{}: {}'.format(k, np.mean(v)))
    
    