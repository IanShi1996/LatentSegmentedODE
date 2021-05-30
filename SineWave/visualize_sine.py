import numpy as np
import matplotlib.pyplot as plt

import torch

from utils import gpu_f, to_np


def visualize_trajectory(data, ts, model, ax):
    """Visualize trajectory reconstructions by Latent NODE model.

    Given an input of ground truth data, visualizes reconstructed trajectory
    made by latent model. Ground truth trajectories are solid, while
    predictions are dashed.

    Data should be in shape of BxLxD where:
        B = number of trajectories
        L = length of time series
        D = input features

    Args:
        data (np.ndarray): Input data to visualize.
        ts (np.ndarray): Time points of observation for data points.
        model (nn.Module): PyTorch model to evaluate.
        ax (matplotlib.axes.Axes): Matplotlib axes to plot results.
    """
    out = to_np(model.get_prediction(gpu_f(data), gpu_f(ts)))

    for i in range(len(data)):
        ax.plot(ts, data[i], c='red', alpha=0.8)
        ax.plot(ts, out[i].squeeze(), c='orange', alpha=0.9, linestyle='--')


def plot_training_sine(model, data, tps, n_plot):
    """Plot trajectories in individual subplots.

    Used as callback during training loop.

    Args:
        model (nn.Module): PyTorch model used to get prediction.
        data (torch.Tensor or np.ndarray): Data.
        tps (torch.Tensor or np.ndarray): Time points.
        n_plot (int): Number of subplots.
    """
    fig, axes = plt.subplots(1, n_plot, figsize=(6 * n_plot, 5))

    ind = np.random.randint(0, len(data), n_plot)

    if isinstance(data, torch.Tensor):
        data = to_np(data)
    data = data[ind]

    if isinstance(tps, torch.Tensor):
        tps = to_np(tps)
    tps = tps[ind]

    for i in range(n_plot):
        d = data[i][np.newaxis, :, :]
        visualize_trajectory(d, tps[i], model, ax=axes[i])
