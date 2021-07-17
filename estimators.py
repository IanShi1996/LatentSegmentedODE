import numpy as np
import torch


def log_normal_pdf(x, mean, logvar):
    """Compute log pdf of data under gaussian specified by parameters.

    Implementation taken from: https://github.com/rtqichen/torchdiffeq.

    Args:
        x (torch.Tensor): Observed data points.
        mean (torch.Tensor): Mean of gaussian distribution.
        logvar (torch.Tensor): Log variance of gaussian distribution.

    Returns:
        torch.Tensor: Log probability of data under specified gaussian.
    """
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)

    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    """Compute analytic KL divergence between two gaussian distributions.

    Computes analytic KL divergence between two multivariate Gaussians which
    are parameterized by the given mean and variances. All inputs must have
    the same dimension.

    Implementation taken from: https://github.com/rtqichen/torchdiffeq.

    Args:
        mu1 (torch.Tensor): Mean of first gaussian distribution.
        lv1 (torch.Tensor): Log variance of first gaussian distribution.
        mu2 (torch.Tensor): Mean of second gaussian distribution.
        lv2 (torch.Tensor): Log variance of second gaussian distribution.

    Returns:
        torch.Tensor: Analytic KL divergence between given distributions.
    """
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    log_std1 = lv1 / 2.
    log_std2 = lv2 / 2.

    kl = log_std2 - log_std1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def reparameterize(qz0_mean, qz0_logvar):
    """Generate latent initial state from latent parameters.

    Use the reparameterization trick to enable back prop with low variance.

    Args:
        qz0_mean (torch.Tensor): Mean of latent distribution.
        qz0_logvar (torch.Tensor): Log variance of latent distribution

    Returns:
        (torch.Tensor, torch.Tensor): Latent state and noise sample.
    """
    epsilon = torch.randn(qz0_mean.size(), device=qz0_mean.device)
    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

    return z0, epsilon


def get_elbo(x, pred_x, q_mean, q_var, noise_std=0.5, kl_weight=1.):
    """Compute the ELBO.

    Computes the evidence lower bound (ELBO) for a given prediction,
    ground truth, and latent initial state.

    Supports KL annealing, where the KL term can gradually be increased
    during training, as described in: https://arxiv.org/abs/1903.10145.

    Args:
        x (torch.Tensor): Input data.
        pred_x (torch.Tensor): Data reconstructed by Latent ODE.
        q_mean (torch.Tensor): Latent initial state means.
        q_var (torch.Tensor): Latent initial state variances.
        noise_std (float, optional): Variance of gaussian pdf.
        kl_weight (float, optional): Weight for KL term.

    Returns:
        torch.Tensor: ELBO score.
    """
    noise_std_ = torch.zeros(pred_x.size(), device=x.device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_)

    log_px = log_normal_pdf(x, pred_x, noise_logvar).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(q_mean.size(), device=x.device)
    analytic_kl = normal_kl(q_mean, q_var, pz0_mean, pz0_logvar).sum(-1)

    return torch.mean(-log_px + kl_weight * analytic_kl, dim=0)
