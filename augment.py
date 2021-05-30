import numpy as np
import torch


def aug_add_noise(data, tps, args):
    """Augment data by injecting gaussian noise.

    This augmentation should serve to improve the stability of the Latent ODE.
    Small perturbations in the input should not affect ODE dynamics.

    TODO: Maybe also add augmentation to latent initial state? Injected noise
    into data dimension may be resolved by encoder, so directly inject into
    NODE.

    Args:
        data (torch.Tensor): Data.
        tps (torch.Tensor): Time.
        args (dict): Augmentation parameters.
            args['noise_var'] (float): Specifies variance of random noise.

    Returns:
        torch.Tensor, torch.Tensor: Augmented data and time.
    """
    noise = torch.randn(data.size()) * args['noise_var']
    data += noise.to(data.device)

    return data, tps


def aug_subsample(data, time, args):
    """Augment data and time by subsampling trajectories.

    This augmentation should improve generalization abilities of the encoder.
    When data points sparse, the trajectory should still map to similar
    latent initial states.

    Args:
        data (torch.Tensor): Data.
        time (torch.Tensor): Time.
        args (dict): Augmentation parameters.
            args['sample_min'] (int): Minimum length of augmented sequence.

    Returns:
        torch.Tensor, torch.Tensor: Augmented data and time.
    """
    if time.shape[1] <= args['sample_min']:
        return data, time

    n_samples = np.random.randint(args['sample_min'], time.shape[1])
    sample_ind = np.random.choice(time.shape[1], n_samples, replace=False)
    sample_ind = np.sort(sample_ind)

    time = time[:, sample_ind]
    data = data[:, sample_ind, :]

    return data, time


def aug_crop_start(data, tps, args):
    """Augment data and time points by truncating beginning of sequence.

    This augmentation should boost generalization for different initial
    conditions with the same parameters.

    Args:
        data (torch.Tensor): Data.
        tps (torch.Tensor): Time points.
        args (dict): Augmentation parameters.
            args['sample_min'] (int): Minimum length of augmented sequence.

    Returns:
        (torch.Tensor, torch.Tensor): Augmented data and time.
    """
    if tps.shape[1] <= args['crop_min']:
        return data, tps

    crop_ind = np.random.randint(args['crop_min'], tps.shape[1])

    tps = tps[:, -crop_ind:]
    data = data[:, -crop_ind:, :]

    return data, tps


def augment(data, tps, methods, args):
    """Perform augmentation by randomly selecting method from list.

    Args:
        data (torch.Tensor): Data.
        tps (torch.Tensor): Time points.
        methods (list of functions): Specific augmentations methods to apply.
        args (dict): Method-specific arguments which are passed through.

    Returns:
        (torch.Tensor, torch.Tensor): Augmented data and time points.
    """
    method_map = {i: methods[i] for i in range(len(methods))}

    select = np.random.choice(np.arange(len(methods)), len(methods))
    select = list(set(select))

    for ind in select:
        data, tps = method_map[ind](data, tps, args)

    return data, tps
