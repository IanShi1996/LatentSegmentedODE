import argparse
import os
import sys

from datetime import datetime
from pathlib import Path

import numpy as np

import torch
from torch.optim import Adamax
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath('..'))

from augment import aug_add_noise, aug_subsample, aug_crop_start
from models import LatentODEBuilder
from train import TrainLoopAE
from utils import gpu_f
from sine_utils import SineSet

np.random.seed(2547)
torch.manual_seed(2547)

device = torch.device('cuda:0')

parser = argparse.ArgumentParser(description="Trains model on Sine Wave data.")
parser.add_argument('--data_file', type=str)
args = parser.parse_args()

data_path = Path("./Data/Train") / Path(args.data_file)
generator = torch.load(data_path)['generator']

train_time, train_data = generator.get_train_set()
val_time, val_data = generator.get_val_set()

train_data = train_data.reshape(len(train_data), -1, 1)
val_data = val_data.reshape(len(val_data), -1, 1)

train_data_tt = gpu_f(train_data)
train_time_tt = gpu_f(train_time)

val_data_tt = gpu_f(val_data)
val_time_tt = gpu_f(val_time)

train_dataset = SineSet(train_data_tt, train_time_tt)
val_dataset = SineSet(val_data_tt, val_time_tt)


model_args = {
    'obs_dim': 1,
    'rec_dim': 8,
    'node_dim': 4,

    'rec_gru_units': 100,
    'rec_node_units': 100,
    'rec_node_layers': 2,
    'rec_node_act': 'Tanh',
    'rec_out_units': 100,

    'latent_node_units': 100,
    'latent_node_layers': 2,
    'latent_node_act': 'Tanh',

    'dec_type': 'NN',
    'dec_units': 100,
    'dec_layers': 2,
    'dec_act': 'ReLU',
}

aug_args = {
    'crop_min': 20,
    'sample_min': 20,
    'noise_var': 0.03,
}

train_args = {
    'aligned_data': True,
    'lr': 5e-3,
    'batch_size': 512,
    'l_std': 1,
    'kl_burn_max': 50,
    'clip_norm': 5,
    'model_atol': 1e-4,
    'model_rtol': 1e-3,
    'aug_methods': [aug_subsample],
    'aug_args': aug_args,
}

total_iters = 14000
iters_per_epoch = generator.n_traj[0] // train_args['batch_size']
train_args['max_epochs'] = total_iters // iters_per_epoch

target_decay = 0.98 ** 240
train_args['lr_decay'] = target_decay ** (1 / train_args['max_epochs'])

train_loader = DataLoader(train_dataset, batch_size=train_args['batch_size'],
                          shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=len(val_dataset))

model = LatentODEBuilder(model_args).build_model().to(device)

parameters = (model.parameters())
optimizer = Adamax(parameters, lr=train_args['lr'])
scheduler = ExponentialLR(optimizer, train_args['lr_decay'])

train_loop = TrainLoopAE(model, train_loader, val_loader, device)
train_loop.train(optimizer, train_args, scheduler)

output_root = Path("./Models")
output_root.mkdir(parents=True, exist_ok=True)

model_name = "sine_lode_" + "_".join(args.data_file.split("_")[2:])
model_name += "_{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'data_path': data_path,
    'model_args': model_args,
    'train_args': train_args,
    'aug_args': aug_args,
}, output_root / Path(model_name))
