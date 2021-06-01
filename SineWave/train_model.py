import argparse
import os
import sys

from datetime import datetime
from pathlib import Path

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath('..'))

from augment import aug_add_noise, aug_subsample, aug_crop_start
from model import LatentODEBuilder
from train import TrainingLoop
from utils import gpu_f, to_np, RunningAverageMeter
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

batch_size = 512

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=len(val_dataset))

model_args = {
    'obs_dim': 1,
    'rec_latent_dim': 8,
    'node_latent_dim': 4,
    
    'rec_gru_unit': 100,
    'rec_node_hidden': 100,
    'rec_node_layer': 2,
    'rec_node_act': 'Tanh',
    
    'latent_node_hidden': 100,
    'latent_node_layer': 2,
    'latent_node_act': 'Tanh',
    
    'dec_type': 'NN',
    'dec_hidden': 100,
    'dec_layer': 2,
    'dec_act': 'ReLU',
}

model = LatentODEBuilder(**model_args).build_latent_ode().to(device)

main = TrainingLoop(model, train_loader, val_loader)

total_iters = 14000
iters_per_epoch = generator.n_traj[0] // batch_size
n_epochs = total_iters // iters_per_epoch

target_decay = 0.98 ** 240
adjusted_decay = target_decay ** (1 / n_epochs)

parameters = (model.parameters())
optimizer = optim.Adamax(parameters, lr=5e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, adjusted_decay)

aug_args = {
    'crop_min': 20,
    'sample_min': 20,
    'noise_var': 0.03,
}

train_args = {
    'max_epoch': n_epochs,
    'l_std': 1,
    'kl_burn': 50,
    'clip_norm': 5,
    'model_atol': 1e-4,
    'model_rtol': 1e-3,
    'aug_methods': [aug_add_noise, aug_subsample, aug_crop_start],
    'aug_args': aug_args,
}

main.train(optimizer, train_args, scheduler)

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