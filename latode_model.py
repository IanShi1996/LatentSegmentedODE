import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from baseline_models import FCNN
from estimators import reparameterize


class ODEFCNN(nn.Module):
    """Fully connected MLP for use as ODE function in torchdiffeq.

    Attributes:
        fcnn (nn.Module): Feedforward neural network.
    """

    def __init__(self, input_dim, n_hidden, n_layer, act_type, output_dim=None):
        """Initialize fully connected neural network.

        Args:
            input_dim (int): Dimension of input data.
            n_hidden (int): Number of hidden units in NN.
            n_layer (int): Number of layers in NN.
            act_type (string): Type of activation to use between layers.
            output_dim (int): Dimension of NN output; defaults to input_dim

        Raises:
            ValueError: Thrown when activation function is unknown.
        """
        super().__init__()
        self.fcnn = FCNN(input_dim, n_hidden, n_layer, act_type, output_dim)

    def forward(self, t, x):
        """Compute forward pass.

        Time input is necessary for use in torchdiffeq framework, but unused.

        Args:
            t (torch.Tensor): Time points of observation.
            x (torch.Tensor): Data observations.

        Returns:
            torch.Tensor: Output of forward pass.
        """
        return self.fcnn.forward(x)


class EncoderGRUODE(nn.Module):
    """GRU with hidden dynamics represented by Neural ODE.

    Implements the GRU-ODE model in: https://arxiv.org/abs/1907.03907.
    Observations are encoded by a GRU. Between observations, the hidden
    state is evolved using a Neural ODE.

    Attributes:
        gru (nn.Module): GRU unit used to encode input data.
        node (nn.Module): Neural ODE used to evolve hidden dynamics.
        out (nn.Module): NN mapping from hidden state to output.
        latent_dim (int): Dimension of latent state.
    """

    def __init__(self, latent_dim, rec_gru, rec_node, rec_output):
        """Initialize GRU-ODE model.

        This module is intended for use as the encoder of a Latent ODE.

        Args:
            latent_dim (int): Dimension of latent state.
            rec_gru (nn.Module): GRU used to encoder input data.
            rec_node (nn.Module): NODE used to evolve state between GRU units.
            rec_output (nn.Module): Final linear layer.
        """
        super().__init__()

        self.gru = rec_gru
        self.node = rec_node
        self.out = rec_output
        self.latent_dim = latent_dim

    def forward(self, x, tps, mask=None):
        """Compute forward pass of GRU-ODE.

        Expects input of shape (B x T x D) and time observation of shape (T).
        Supports input masked by 2D binary array of shape (B x T).

        ODE dynamics are solved using euler's method. Other solvers decrease
        performance and increase runtime.

        Masked runs require additional tracking. The hidden state should
        not evolve until data is seen, and should not evolve after the last
        data point. Currently we track this with an array, but it is not
        memory efficient. Simple alternatives such as tracking last seen
        time point doesn't work since we need an identical input into the ode
        solve.

        Args:
            x (torch.Tensor): Data observations.
            tps (torch.Tensor): Time points.
            mask (torch.Tensor): 2D masking array.

        Returns:
            torch.Tensor: Output representing latent parameters.
        """
        h = torch.zeros(x.shape[0], self.latent_dim * 2).to(x.device)

        # Insert dummy time point which is discarded later.
        tps = torch.cat(((tps[0]-0.01).unsqueeze(0), tps), 0)

        h_arr, r_fill_mask = None, None

        if mask is not None:
            h_arr = torch.zeros(x.shape[0], x.shape[1], h.size(1)).to(x.device)
            r_fill_mask = self.right_fill_mask(mask)

        for i in range(x.shape[1]):
            if i != 0:
                h_ode = odeint(self.node, h, tps[i:i+2], method="euler")[1]

                if mask is not None:
                    curr_r_mask = r_fill_mask[:, i].view(-1, 1)
                    h_ode = h_ode * curr_r_mask + h * (1 - curr_r_mask)
            else:
                # Don't evolve hidden state prior to first observation
                h_ode = h

            h_rnn = self.gru(x[:, i, :], h_ode)

            if mask is not None:
                curr_mask = mask[:, i].view(-1, 1)
                h = h_rnn * curr_mask + h_ode * (1 - curr_mask)

                h_arr[:, i, :] = h
            else:
                h = h_rnn

        if mask is not None:
            ind = self.get_last_tp(mask)
            h = torch.zeros(x.shape[0], self.latent_dim * 2).to(x.device)
            for i, traj in enumerate(h_arr):
                h[i] = traj[ind[i]]
        out = self.out(h)
        return out

    @staticmethod
    def right_fill_mask(mask):
        """Return mask will all non-leading zeros filled with ones."""
        mask = mask.detach().clone()
        for i, row in enumerate(mask):
            seen = False
            for j, mp in enumerate(row):
                if seen:
                    mask[i][j] = 1
                elif mp == 1:
                    mask[i][j] = 0
                    seen = True
        return mask

    @staticmethod
    def get_last_tp(mask):
        """Return index of last observation per trajectory in masked batch."""
        mask_rev = torch.flip(mask, [1])

        ind = []
        for mask_row in mask_rev:
            count = 0
            while mask_row[count] == 0:
                count += 1
            ind.append(mask.size(1) - 1 - count)

        return ind


class NeuralODE(nn.Module):
    """Neural Ordinary Differential Equation.

    Implements Neural ODEs as described by: https://arxiv.org/abs/1806.07366.
    ODE solve uses a semi-norm. See: https://arxiv.org/abs/2009.09457.

    Attributes:
        nodef (nn.Module): NN which approximates ODE function.
    """

    def __init__(self, input_dim, n_hidden, n_layer, act_type):
        """Initialize Neural ODE.

        Args:
            input_dim (int): Dimension of input data.
            n_hidden (int): Number of hidden units in NN.
            n_layer (int): Number of layers in NN.
            act_type (string): Type of activation to use between layers.
        """
        super().__init__()

        self.nodef = ODEFCNN(input_dim, n_hidden, n_layer, act_type)

    def forward(self, z0, ts, rtol=1e-3, atol=1e-4):
        """Compute forward pass of NODE.

        Args:
            z0 (torch.Tensor): Initial state of ODE.
            ts (torch.Tensor): Time points of observations.
            rtol (float, optional): Relative tolerance of ode solver.
            atol (float, optional): Absolute tolerance of ode solver.

        Returns:
            torch.Tensor: Result of ode solve from initial state.
        """
        z = odeint(self.nodef, z0, ts, rtol=rtol, atol=atol, method='dopri5',
                   adjoint_options=dict(norm="seminorm"))
        return z.permute(1, 0, 2)


class LatentODE(nn.Module):
    """Latent ODE.

    Implements Latent ODE described in https://arxiv.org/abs/1907.03907.
    Model consists of an GRU-ODE encoder, Neural ODE, and NN Decoder which
    is configured and trained as a VAE.

    Attributes:
        dec (nn.Module): Decoder module.
        enc (nn.Module): Encoder module.
        nodef (nn.Module): Neural ODE module.
    """

    def __init__(self, enc, nodef, dec):
        """Initialize Latent ODE.

        Args:
            dec (nn.Module): Decoder module.
            enc (nn.Module): Encoder module.
            nodef (nn.Module): Neural ODE module.
        """
        super().__init__()

        self.enc = enc
        self.nodef = nodef
        self.dec = dec

    def get_latent_params(self, x, ts, mask=None):
        """Compute latent parameters.

        Allows masking via a 2D binary array of shape (B x T).

        Args:
            x (torch.Tensor): Data points.
            ts (torch.Tensor): Time points of observations.
            mask (torch.Tensor, optional): Masking array.

        Returns:
            torch.Tensor, torch.Tensor: Latent mean and logvar parameters.
        """
        obs = torch.flip(x, [1])
        rev_ts = torch.flip(ts, [0])

        if mask is not None:
            mask = torch.flip(mask, [1])

        out = self.enc.forward(obs, rev_ts, mask)

        qz0_mean = out[:, :out.size(1) // 2]
        qz0_logvar = out[:, out.size(1) // 2:]

        return qz0_mean, qz0_logvar

    def generate_latent_traj(self, z0, ts, rtol=1e-3, atol=1e-4):
        """Generate a latent trajectory from a latent initial state.

        Args:
            z0 (torch.Tensor): Latent initial state.
            ts (torch.Tensor): Time points of observations.
            rtol (float, optional): NODE ODE solver relative tolerance.
            atol (float, optional): NODE ODE solver absolute tolerance.

        Returns:
            torch.Tensor: Latent trajectory.
        """
        return self.nodef.forward(z0, ts, rtol, atol)

    def forward(self, x, ts, mask=None, rtol=1e-3, atol=1e-4):
        """Compute forward pass of Latent ODE.

        Args:
            x (torch.Tensor): Input data.
            ts (torch.Tensor): Time points of observations.
            mask (torch.Tensor, optional): Masking array.
            rtol (float, optional): NODE ODE solver relative tolerance.
            atol (float, optional): NODE ODE solver absolute tolerance.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
                Reconstructed data, latent mean, latent logvar, sampled noise.
        """
        qz0_mean, qz0_logvar = self.get_latent_params(x, ts, mask)
        z0, epsilon = reparameterize(qz0_mean, qz0_logvar)

        pred_z = self.generate_latent_traj(z0, ts, rtol, atol)
        pred_x = self.dec(pred_z)

        return pred_x, qz0_mean, qz0_logvar, epsilon

    def predict(self, x, ts, mask=None, rtol=1e-3, atol=1e-4):
        """Retrieve prediction from Latent ODE output.

        Args:
            x (torch.Tensor): Data points.
            ts (torch.Tensor): Time points of observations.
            mask (torch.Tensor, optional): Masking array.
            rtol (float, optional): NODE ODE solver relative tolerance.
            atol (float, optional): NODE ODE solver absolute tolerance.

        Returns:
            torch.Tensor: Reconstructed data points.
        """
        return self.forward(x, ts, mask, rtol, atol)[0]

    def initialize_normal(self, std=0.1):
        """Initialize linear layers with normal distribution.

        Args:
            std (float): Standard deviation of normal distribution.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=std)
                nn.init.constant_(module.bias, val=0)
