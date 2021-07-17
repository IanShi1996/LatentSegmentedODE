import argparse
import os
import sys

from datetime import datetime
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import Adamax

sys.path.append(os.path.abspath('..'))

from augment import aug_add_noise, aug_subsample, aug_crop_start
from models import MODEL_CONSTR_DICT, MODEL_TRAIN_DICT
from segsine_utils import SegmentedSineSet, HybridSineSet

np.random.seed(2547)
torch.manual_seed(2547)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--data_file', type=str, required=True)
parser.add_argument('--data_type', type=str, required=True, choices=['h', 's'])
parser.add_argument('--model_type', type=str, required=True,
                    choices=['gru', 'gruode', 'latode', 'segode'])
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.999)
parser.add_argument('--del_t', action="store_true")

parser.add_argument('--rec_dim', type=int, default=16)
parser.add_argument('--rec_gru_units', type=int, default=200)
parser.add_argument('--rec_node_units', type=int, default=100)
parser.add_argument('--rec_node_layers', type=int, default=2)
parser.add_argument('--rec_node_act', type=str, default='Tanh')
parser.add_argument('--rec_out_units', type=int, default=100)

parser.add_argument('--node_dim', type=int, default=8)
parser.add_argument('--latent_node_units', type=int, default=100)
parser.add_argument('--latent_node_layers', type=int, default=2)
parser.add_argument('--latent_node_act', type=str, default='Tanh')

parser.add_argument('--dec_units', type=int, default=100)
parser.add_argument('--dec_layers', type=int, default=1)
parser.add_argument('--dec_act', type=str, default='ReLU')

parser.add_argument('--s_min', type=float, default=0.5)
parser.add_argument('--s_frac', type=float, default=0.75)
parser.add_argument('--samp_prob', type=float, default=0.5)

parser.add_argument('--aug_crop', action='store_true')
parser.add_argument('--aug_samp', action='store_true')
parser.add_argument('--aug_noise', action='store_true')

parser.add_argument('--crop_min', type=int, default=30)
parser.add_argument('--sample_min', type=int, default=50)
parser.add_argument('--aug_noise_var', type=float, default=0.01)

parser.add_argument('--kl_burn_max', type=int, default=1)
parser.add_argument('--l_std', type=float, default=0.1)

args = parser.parse_args()

data_path = Path("./Data/") / Path(args.data_file)
data = torch.load(data_path)

dataset = HybridSineSet if args.data_type == 'h' else SegmentedSineSet

train_data = dataset(data['train_set'])
val_data = dataset(data['val_set'])

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data))

model_args = {
    'type': args.model_type,
    'obs_dim': 1,
    'rec_dim': args.rec_dim,
    'rec_gru_units': args.rec_gru_units,
    'dec_units': args.dec_units,
    'dec_layers': args.dec_layers,
    'dec_act': args.dec_act,
}

if args.model_type in ['gru']:
    model_args['del_t'] = args.del_t

if args.model_type in ['gruode', 'latsegode', 'latode']:
    model_args['rec_node_units'] = args.rec_node_units
    model_args['rec_node_layers'] = args.rec_node_layers
    model_args['rec_node_act'] = args.rec_node_act

if args.model_type in ['latode', 'latsegode']:
    model_args['node_dim'] = args.node_dim
    model_args['rec_out_units'] = args.rec_out_units
    model_args['latent_node_units'] = args.latent_node_units
    model_args['latent_node_layers'] = args.latent_node_layers
    model_args['latent_node_act'] = args.latent_node_act
    model_args['dec_type'] = 'NN'

model_constructor = MODEL_CONSTR_DICT[args.model_type]
model = model_constructor(model_args).build_model().to(device)

if args.model_type in ['gru', 'gruode']:
    aug_args = {
        's_min': args.s_min,
        's_frac': args.s_frac,
        'samp_prob': args.samp_prob,
    }
else:
    aug_args = {
        'crop_min': args.crop_min,
        'sample_min': args.sample_min,
        'aug_noise_var': args.aug_noise_var
    }

aug_methods = []
if hasattr(args, 'aug_crop') and args.aug_crop:
    aug_methods.append(aug_crop_start)
if hasattr(args, 'aug_samp') and args.aug_samp:
    aug_methods.append(aug_subsample)
if hasattr(args, 'aug_noise') and args.aug_noise:
    aug_methods.append(aug_add_noise)

train_args = {
    'max_epochs': args.max_epochs,
    'lr': args.lr,
    'lr_decay': args.lr_decay,
    'batch_size': args.batch_size,
    'l_std': args.l_std,
    'kl_burn_max': args.kl_burn_max,
    'aug_methods': aug_methods,
    'aug_args': aug_args
}

print(model_args, flush=True)
print(train_args, flush=True)

optimizer = Adamax(model.parameters(), args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

train_loop_class = MODEL_TRAIN_DICT[args.model_type]
train_loop = train_loop_class(model, train_loader, val_loader, device)
train_loop.train(optimizer, train_args, scheduler)

output_root = Path("./Models")
output_root.mkdir(parents=True, exist_ok=True)

model_name = "segsine_{}".format(args.model_type)
model_name += "_{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'data_path': data_path,
    'model_args': model_args,
    'train_args': train_args,
}, output_root / Path(model_name))
