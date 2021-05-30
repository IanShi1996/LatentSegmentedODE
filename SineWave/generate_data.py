from datetime import datetime
from pathlib import Path

import torch

from simulate_sine import SineSetGenerator, generate_test_set

timestamp = datetime.now()

# Parameters which control training dataset
train_data_params = {
    'n_traj': (30000, 200, 200),
    'n_samp': (200, 100, 100),
    'amp': (-10, 10),
    'freq': (1, 4),
    'phase': True,
    'start': 0,
    'end': 7,
    'noise': 0.025,
    'tp_generation': ('U', 'U', 'U'),
}

train_generator = SineSetGenerator(train_data_params)

# Parameters which control test segmented dataset
test_generator_params = {
    'amp_range': (-8, 8),
    'freq_range': (1, 4),
    'phase_flag': True,
    'cp_min': 50,
    'amp_min': 3
}

test_n_traj = 25
test_max_cp = 2

test_data = generate_test_set(test_generator_params, test_max_cp, test_n_traj)

train_data = {
    'generator': train_generator,
    'data_params': train_data_params,
}

test_data = {
    'generator_params': test_generator_params,
    'data': test_data
}

Path("./Data/Train").mkdir(parents=True, exist_ok=True)
Path("./Data/Test").mkdir(parents=True, exist_ok=True)

torch.save(train_data, "./Data/Train/sine_data_{}".format(timestamp))
torch.save(test_data, "./Data/Test/sine_piecewise_{}".format(timestamp))
