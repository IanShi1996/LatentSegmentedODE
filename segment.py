import torch
import numpy as np
from scipy.special import logsumexp
from utils import to_np


def generate_epsilon(n_samples, latent_dim):
    """Generate gaussian noise.

    Args:
        n_samples (int): Number of independent samples to take.
        latent_dim (int): Number of dimensions for noise.

    Returns:
        torch.Tensor: Tensor containing gaussian noise.
    """
    return torch.randn((n_samples, latent_dim))


def reparameterize(epsilon, qz0_mean, qz0_logvar):
    """Reparameterize using mean and variance with given noise.

    Args:
        epsilon (torch.Tensor): Noise from zero one gaussian.
        qz0_mean (torch.Tensor): Latent mean vector.
        qz0_logvar (torch.Tensor): Latent log variance vector.

    Returns:
        torch.Tensor: Reparameterized latent initial states.
    """
    return epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean


def parallel_estimate_logpx(data, tp, seg_len, cps, model, n_samp, precision,
                            noise_var=1, fixed_eps=None):
    """Estimate log marginal likelihood through parallel MC sampling.

    Computes Monte Carlo importance sampling estimation of log p(x).
    Several observations are exploited to allow combination of segments into
    one batch:

    1. Encoder output is deterministic, meaning each input trajectory needs
       one computation to obtain the variational posterior.
    2. ODE solutions scales well as number of timepoints solved increases.
       Thus, we can combine all segmentations by first scaling the timepoints
       such that the first timepoint is 0, and then finding the set union of
       all timepoints for all trajectories. These time deltas are rounded
       to a decimal precision to save compute.

    All these steps result in a ~30x speed up compared to the sequential
    implementation.

    Further speed-up can likely be obtained by optimizing encoder masking and
    score computation, but gains are marginal at this point.

    The fixed variance used to calculate reconstruction error can be modified.
    Performance seems to be best when this is set to match the variance used
    during model training.

    Empirical testing shows that the K bound can be lowered if a fixed noise
    is used. This is optional.

    Args:
        data (torch.Tensor): Tensor of input data points.
        tp (torch.Tensor): Global tensor of timepoints.
        seg_len (int): Length of segment being considered
        cps (list of int): All changepoints being considered.
        model (nn.Module): Latent NODE model used to evaluate scores.
        n_samp (int): Number of MC samples to take.
        precision (int): Decimal precision used to calculate time deltas.
        noise_var (float): Fixed variance used to calculate loss.
        fixed_eps (torch.Tensor): Noise used for reparameterization.

    Returns:
        list of float: Estimated log p(x) of segment.
    """
    mask = get_mask_array(seg_len, cps).to(data.device)

    seg_data = data[:, cps[0]:seg_len, :]
    batch_seg = torch.cat(len(cps) * [seg_data])

    seg_tp = tp[cps[0]:seg_len]
    seg_tp = seg_tp - seg_tp[0]

    qz0_mean, qz0_logvar = model.get_latent_initial_state(batch_seg, seg_tp,
                                                          mask=mask)

    qz0_mean = qz0_mean.repeat_interleave(n_samp, 0)
    qz0_logvar = qz0_logvar.repeat_interleave(n_samp, 0)

    if fixed_eps is not None:
        eps_repeat = torch.cat([fixed_eps] * int(qz0_mean.size(0) / n_samp))
        z0 = reparameterize(eps_repeat, qz0_mean, qz0_logvar)
        eps = fixed_eps
    else:
        z0, eps = model.reparameterize(qz0_mean, qz0_logvar)

    qz0_var = torch.exp(.5 * qz0_logvar)

    const = torch.from_numpy(np.array([2. * np.pi])).float().to(data.device)
    const = torch.log(const) + np.log(noise_var)

    seg_tps, tp_union_tt = get_tp_map(tp, seg_len, cps, precision)

    pred_z = model.generate_from_latent(z0, tp_union_tt)

    tp_union = to_np(tp_union_tt)
    pred_z = recover_tps(pred_z, cps, seg_tps, tp_union, n_samp, precision)

    mc_ests = []

    for i in range(len(pred_z)):
        pred_x = model.dec(pred_z[i])
        data_seg = data[:, cps[i]:seg_len, :]

        # Compute data likelihood: p(x|z)
        likelihood = -.5 * (const + (data_seg - pred_x) ** 2. / noise_var)
        likelihood = likelihood.sum(-1).sum(-1)

        # Compute variation posterior: q(z|x)
        if fixed_eps is not None:
            qz = torch.sum(-0.5 * eps ** 2 -
                           torch.log(qz0_var[i*n_samp:i*n_samp+n_samp]), -1)
        else:
            qz = torch.sum(-0.5 * eps[i*n_samp:i*n_samp+n_samp] ** 2 -
                           torch.log(qz0_var[i*n_samp:i*n_samp+n_samp]), -1)

        # Compute parameter prior: p(z)
        pz = torch.sum(-0.5 * z0[i*n_samp:i*n_samp+n_samp] ** 2, -1)

        logpx = likelihood + pz - qz

        mc_est = logsumexp(to_np(logpx)) - np.log(n_samp)
        mc_ests.append(-mc_est)

    return mc_ests


def segment(data, tp, model, n_samp, min_seg_len, K, n_decimal,
            noise_var=1, fixed_noise=False, latent_dim=None):
    """Perform segmentation using OPTSEG or PELT algorithm.

    Computes segmentation of a time series using an optimal partitioning
    algorithm. All possible segmentations are evaluated using dynamic
    programming. Non-optimal changepoints can be discarded under a pruning
    condition. The optimal segmentation and pruned exact linear time (PELT)
    method are adapted from the PELT paper: https://arxiv.org/pdf/1101.1438.pdf

    The PELT prune condition requires there exist a K such that:
        C(X_s:t) + C(X_t:T) + K <= C(X_s:T)
    This condition is true for K=0 when the cost function assumes i.i.d. data,
    but fails for the Segmented NODE model. Thus, K needs to be empirically
    specified. Increasing K will consider more segments, increasing runtime.

    This function can be converted to optimal segmentation by specifying a very
    high K term, such as np.inf, disallowing discarding of changepoints.

    Previously, a penalization parameter for approximating the BIC existed,
    but has been removed as taking MC estimates of log p(x) allows segmentation
    without penalization, due to the Bayesian Occam's Razor effect:

    https://www.cs.princeton.edu/courses/archive/fall09/
    cos597A/papers/MacKay2003-Ch28.pdf

    Setting the minimum segmentation length will ignore segments less than the
    length, providing marginal speedup.

    The scoring function has been switched to the parallel implementation.

    Empirically, using a fixed point noise has shown to result in better
    performance.

    Args:
        data (torch.Tensor): Input data.
        tp (torch.Tensor): Observation times of input data.
        model (nn.Module): Latent NODE used to perform reconstructions.
        n_samp (int): Number of MC samples used to estimate score.
        min_seg_len (int): Minimum length of segment to consider.
        K (float): Term used to relax pruning condition.
        n_decimal (int): Decimal precision used in timepoint delta calculation.
        noise_var (float): Fixed variance used to calculate loss.
        fixed_noise (boolean): Whether fixed noise should be used.
        latent_dim (int): Dim. of latent. Only required is fixed noise is used.

    Returns:
        np.ndarray: Matrix containing all segment scores.
    """
    length = data.shape[1]

    score = np.zeros((length, length))
    score.fill(np.inf)

    min_score = np.zeros(length)
    valid_cp = [0]

    if fixed_noise:
        eps = generate_epsilon(n_samp, latent_dim).to(data.device)
    else:
        eps = None

    for seg_len in range(1, length):
        segment_score = np.ones(length) * np.inf
        if seg_len % 10 == 0:
            print(seg_len, end=' ')

        seg_cp = [cp for cp in valid_cp if cp < (seg_len - min_seg_len)]

        if len(seg_cp) == 0:
            min_score[seg_len] = np.inf
            valid_cp.append(seg_len)
            continue

        mc_est = parallel_estimate_logpx(data, tp, seg_len, seg_cp, model,
                                         n_samp, n_decimal, noise_var, eps)

        for i in range(len(mc_est)):
            cp = seg_cp[i]

            l_seg_score = min_score[cp]
            r_seg_score = mc_est[i]

            if (seg_len - cp) < min_seg_len:
                r_seg_score = np.inf

            segment_score[cp] = l_seg_score + r_seg_score

        score[seg_len, :len(segment_score)] = segment_score
        min_score[seg_len] = min(segment_score)

        for cp in seg_cp:
            if score[seg_len, cp] - K >= min_score[seg_len]:
                valid_cp.remove(cp)

        valid_cp.append(seg_len)

    return score


def get_changepoints(scores):
    """Recover optimal changepoints from segmentation score matrix.

    Uses the score matrix from PELT segmentation to recover the optimal
    segmentation.

    Args:
        scores (np.ndarray): Matrix containing segment scores from PELT.

    Returns:
        np.ndarray: Location of optimal changepoints.
    """
    changepoints = []
    curr_cp = len(scores) - 1

    while curr_cp > 0:
        curr_cp = np.argmin(scores[curr_cp])
        changepoints.append(curr_cp)

    changepoints = np.sort(changepoints)[1:]
    return changepoints


def get_mask_array(seg_len, cps):
    """Return binary array used to mask input to Latent NODE encoders.

    The binary array contains 1 when hidden state should be updated, and 0
    otherwise. The array is set based on the desired changepoints to evaluate.

    Args:
        seg_len (int): Length of overall segment.
        cps (list of int): All changepoints being considered.

    Returns:
        torch.Tensor: Binary masking array.
    """
    mask = torch.zeros((len(cps), seg_len - cps[0]))

    for i in range(len(cps)):
        mask[i, cps[i] - cps[0]:] += 1

    return mask


def get_tp_map(tp, seg_len, cps, n_decimal):
    """Compute all unique time deltas.

    Computes all time trajectories as if they started from zero. This is done
    by computing the set of all possible time deltas, and which is combined
    to allow for fast sequential evaluation via ode solve. These deltas
    can then be reconstructed according to their input hidden state.

    Time deltas are truncated to n decimal places for computational speedup.
    Empirically, 2 decimal places are sufficient for accurate results.

    Args:
        tp (torch.Tensor): Tensor containing global timepoints.
        seg_len (int): Length of segment being considered.
        cps (list of int): List of changepoints under consideration
        n_decimal (int): Number of decimals to round time deltas to.

    Returns:
        list of torch.Tensor, torch.Tensor: All time deltas associated with
            specific data trajectory, and unique sorted set of all timepoints.
    """
    seg_tps = []

    for i in range(len(cps)):
        seg_tp = tp[cps[i]:seg_len]
        seg_tp = seg_tp - seg_tp[0]
        seg_tp = torch.round(seg_tp * 10 ** n_decimal) / (10 ** n_decimal)
        seg_tps.append(seg_tp)

    tp_union = torch.unique(torch.cat(seg_tps))

    return seg_tps, tp_union


def recover_tps(out, cps, seg_tps, tp_union, n_samp, n_decimal):
    """Reconstructs latent trajectories from parallel odesolve output.

    Maps odesolve output at particular timepoint to its original trajectory by
    using the list of time deltas previously constructed in get_tp_map.

    Args:
        out (torch.Tensor): Tensor of odesolve results.
        cps (list of int): Changepoints under consideration.
        seg_tps (list of torch.Tensor): Time deltas per original trajectory.
        tp_union (torch.Tensor): List of unique time deltas.
        n_samp (int): Number of MC samples per trajectory.
        n_decimal (int): Decimal precision used to store time deltas.

    Returns:
        list of torch.Tensor: Reconstructed latent trajectories.
    """
    data = []

    tp_union = np.round(tp_union, n_decimal)
    for i in range(len(cps)):
        seg_np = np.round(to_np(seg_tps[i]), n_decimal)

        idx_array = [np.where(tp == tp_union)[0][0] for tp in seg_np]
        seg_data = out[i*n_samp:i*n_samp+n_samp, idx_array, :]
        data.append(seg_data)
    return data
