import argparse
from pathlib import Path

import torch

from sine_simulate import SineSetGenerator, generate_test_set


parser = argparse.ArgumentParser(description="Generates Sine Wave data.")
parser.add_argument('--n_train_traj', type=int, default=10000)
parser.add_argument('--n_test_traj', type=int, default=25)
parser.add_argument('--n_train_samp', type=int, default=100)
parser.add_argument('--n_test_samp', type=int, default=100)
parser.add_argument('--length', type=float, default=7)
parser.add_argument('--noise', type=float, default=0.025)
parser.add_argument('--amp_min', type=float, default=3)
args = parser.parse_args()

amp_range = (-10, 10)
freq_range = (1, 4)

# Parameters which control training dataset
train_data_params = {
    'n_traj': (args.n_train_traj, 256, 1),
    'n_samp': (args.n_train_samp, args.n_train_samp, 1),
    'amp': amp_range,
    'freq': freq_range,
    'phase': True,
    'start': 0,
    'end': args.length,
    'noise': args.noise,
    'tp_generation': ('U', 'U', 'U'),
}

train_generator = SineSetGenerator(train_data_params)

# Parameters which control test segmented dataset
test_generator_params = {
    'amp_range': amp_range,
    'freq_range': freq_range,
    'phase_flag': True,
    'cp_min': args.n_test_samp * 0.25,
    'amp_min': args.amp_min,
    'noise': args.noise
}

test_max_cp = 2

test_data = generate_test_set(test_generator_params, test_max_cp,
                              args.n_test_traj, args.n_test_samp, args.length)

train_data = {
    'generator': train_generator,
    'data_params': train_data_params,
}

test_data = {
    'generator_params': test_generator_params,
    'data': test_data
}

train_path = Path("./Data/Train")
test_path = Path("./Data/Test")

train_path.mkdir(parents=True, exist_ok=True)
test_path.mkdir(parents=True, exist_ok=True)

train_file_name = "sine_train_{}_{}_{}_{}".format(args.n_train_traj,
                                                  args.n_train_samp,
                                                  args.length,
                                                  args.noise)

test_file_name = "sine_test_{}_{}_{}_{}".format(args.n_test_samp, args.length,
                                                args.noise, args.amp_min)

train_file_path = train_path / Path(train_file_name)
test_file_path = test_path / Path(test_file_name)

if not train_file_path.exists():
    torch.save(train_data, train_file_path)

if not test_file_path.exists():
    torch.save(test_data, test_file_path)
