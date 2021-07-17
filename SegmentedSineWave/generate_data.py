import argparse
from pathlib import Path
import uuid

import torch

from segsine_simulate import generate_resampled_set, generate_test_set

parser = argparse.ArgumentParser(description="Generates Sine Wave data.")
parser.add_argument('--amp_range_min', type=int, default=-8)
parser.add_argument('--amp_range_max', type=int, default=8)
parser.add_argument('--freq_range_min', type=int, default=2)
parser.add_argument('--freq_range_max', type=int, default=4)
parser.add_argument('--samp_range_min', type=int, default=50)
parser.add_argument('--samp_range_max', type=int, default=150)
parser.add_argument('--end_range_min', type=int, default=3)
parser.add_argument('--end_range_max', type=int, default=5)
parser.add_argument('--min_amp_delta', type=float, default=2.5)
parser.add_argument('--min_freq_delta', type=float, default=0.25)
parser.add_argument('--noise', type=float, default=0.025)
parser.add_argument('--extrap_len', type=float, default=100)
parser.add_argument('--extrap_time', type=float, default=5)
parser.add_argument('--max_cp', type=float, default=2)
parser.add_argument('--n_traj_train', type=float, default=7500)
parser.add_argument('--n_traj_val', type=float, default=50)
parser.add_argument('--n_traj_test', type=float, default=50)
args = parser.parse_args()

data_args = {
    'amp_range': (args.amp_range_min, args.amp_range_max),
    'freq_range': (args.freq_range_min, args.freq_range_max),
    'samp_range': (args.samp_range_min, args.samp_range_max),
    'end_range': (args.end_range_min, args.end_range_max),
    'min_amp_delta': args.min_amp_delta,
    'min_freq_delta': args.min_freq_delta,
    'noise': args.noise,
    'extrap_len': args.extrap_len,
    'extrap_time': args.extrap_time,
    'max_cp': args.max_cp,
    'n_traj_train': args.n_traj_train,
    'n_traj_val': args.n_traj_val,
    'n_traj_test': args.n_traj_test,
}

train_set = generate_resampled_set(data_args['n_traj_train'],
                                   data_args['samp_range'],
                                   data_args['end_range'][1],
                                   data_args['amp_range'],
                                   data_args['freq_range'],
                                   data_args['max_cp'],
                                   data_args['min_amp_delta'],
                                   data_args['min_freq_delta'],
                                   data_args['noise'])

val_set = generate_resampled_set(data_args['n_traj_val'],
                                 data_args['samp_range'],
                                 data_args['end_range'][1],
                                 data_args['amp_range'],
                                 data_args['freq_range'],
                                 data_args['max_cp'],
                                 data_args['min_amp_delta'],
                                 data_args['min_freq_delta'],
                                 data_args['noise'])

test_set = generate_test_set(data_args['n_traj_test'],
                             data_args['samp_range'],
                             data_args['end_range'],
                             data_args['amp_range'],
                             data_args['freq_range'],
                             data_args['max_cp'],
                             data_args['min_amp_delta'],
                             data_args['min_freq_delta'],
                             data_args['noise'],
                             data_args['extrap_len'],
                             data_args['extrap_time'])

out_dir = Path("./Data")
out_dir.mkdir(parents=True, exist_ok=True)

path_name = 'segsine_data_{}'.format(uuid.uuid4())

out_path = out_dir / Path(path_name)

torch.save({
    'train_set': train_set,
    'val_set': val_set,
    'test_set': test_set,
    'data_args': data_args,
}, out_path)
