import numpy as np
import torch


def aug_add_noise(data, tps, lengths, args):
    """Augment data by injecting gaussian noise.

    This augmentation should serve to improve the stability of the Latent ODE.
    Small perturbations in the input should not affect ODE dynamics.

    TODO: Maybe also add augmentation to latent initial state? Injected noise
    into data dimension may be resolved by encoder, so directly inject into
    NODE.

    Args:
        data (torch.Tensor): Data.
        tps (torch.Tensor): Time.
        lengths (torch.Tensor): Segment lengths.
        args (dict): Augmentation parameters.
            args['noise_var'] (float): Specifies variance of random noise.

    Returns:
        torch.Tensor, torch.Tensor: Augmented data and time.
    """
    noise = torch.randn(data.size()) * args['aug_noise_var']
    data += noise.to(data.device)

    return data, tps, lengths


def aug_subsample(data, time, lengths, args):
    """Augment data and time by subsampling trajectories.

    This augmentation should improve generalization abilities of the encoder.
    When data points sparse, the trajectory should still map to similar
    latent initial states.

    Args:
        data (torch.Tensor): Data.
        time (torch.Tensor): Time.
        lengths (torch.Tensor): Segment lengths.
        args (dict): Augmentation parameters.
            args['sample_min'] (int): Minimum length of augmented sequence.

    Returns:
        torch.Tensor, torch.Tensor, torch.Tensor: Augmented data, time, lengths.
    """
    if time.shape[1] <= args['sample_min']:
        return data, time, lengths

    n_samples = np.random.randint(args['sample_min'], time.shape[1])
    sample_ind = np.random.choice(time.shape[1], n_samples, replace=False)
    sample_ind = np.sort(sample_ind)

    time = time[:, sample_ind]
    data = data[:, sample_ind, :]

    if lengths is not None:
        for i in range(len(lengths)):
            lengths[i] = len(np.where(sample_ind < lengths[i].item())[0])

    return data, time, lengths


def aug_crop_start(data, tps, lengths, args):
    """Augment data and time points by truncating beginning of sequence.

    This augmentation should boost generalization for different initial
    conditions with the same parameters.

    Args:
        data (torch.Tensor): Data.
        tps (torch.Tensor): Time points.
        lengths (torch.Tensor): Segment lengths.
        args (dict): Augmentation parameters.
            args['sample_min'] (int): Minimum length of augmented sequence.

    Returns:
        torch.Tensor, torch.Tensor, torch.Tensor: Augmented data, time, lengths.
    """
    if tps.shape[1] <= args['crop_min']:
        return data, tps, lengths

    crop_ind = np.random.randint(args['crop_min'], tps.shape[1])

    tps = tps[:, -crop_ind:]
    data = data[:, -crop_ind:, :]

    if lengths is not None:
        lengths = lengths - crop_ind

    return data, tps, lengths


def augment(data, tps, lengths, methods, args):
    """Perform augmentation by randomly selecting method from list.

    Args:
        data (torch.Tensor): Data.
        tps (torch.Tensor): Time points.
        lengths (torch.Tensor): Segment lengths. Set to None if N/A.
        methods (list of functions): Specific augmentations methods to apply.
        args (dict): Method-specific arguments which are passed through.

    Returns:
        torch.Tensor, torch.Tensor, torch.Tensor: Augmented data, time, lengths.
    """
    method_map = {i: methods[i] for i in range(len(methods))}

    select = np.random.choice(np.arange(len(methods)), len(methods))
    select = list(set(select))

    for ind in select:
        data, tps, lengths = method_map[ind](data, tps, lengths, args)

    return data, tps, lengths
