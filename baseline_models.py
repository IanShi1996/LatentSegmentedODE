import torch
import torch.nn as nn
import numpy as np

from torchdiffeq import odeint_adjoint as odeint


class Swish(nn.Module):
    """Swish activation function.

    Implements swish activation function: https://arxiv.org/pdf/1710.05941.pdf.
    Claimed by NODE authors to perform well in NODEs.
    """

    def __init__(self):
        """Initialize swish activation function."""
        super(Swish, self).__init__()

    def forward(self, x, beta=1):
        """Compute swish forward pass.

        Args:
            x (torch.Tensor): Input data.
            beta (float, optional): Scaling factor. Defaults to 1.

        Returns:
            torch.Tensor: Data with swish non-linearity applied.
        """
        return x * torch.sigmoid(beta * x)


ACTIVATION_DICT = {
    'Swish': Swish,
    'Tanh': nn.Tanh,
    'ReLU': nn.ReLU,
    'Softplus': nn.Softplus
}


class FCNN(nn.Module):
    """Generic fully connected MLP.

    Attributes:
        act (nn.Module): Activation function to use between layers.
        fc_in (nn.Module): Linear layer mapping input to hidden state.
        fc_out (nn.Module): Linear layer mapping hidden state to output.
        fc_hidden (nn.ModuleList): Hidden layers.
    """

    def __init__(self, input_dim, n_hidden, n_layer, act_type, output_dim=None):
        """Initialize NN representing ODE function.

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

        output_dim = input_dim if output_dim is None else output_dim

        self.fc_in = nn.Linear(input_dim, n_hidden)
        self.fc_out = nn.Linear(n_hidden, output_dim)

        layers = [nn.Linear(n_hidden, n_hidden) for _ in range(n_layer-1)]
        self.fc_hidden = nn.ModuleList(layers)

        try:
            self.act = ACTIVATION_DICT[act_type]()
        except KeyError:
            raise ValueError("Unsupported activation function.")

    def forward(self, x):
        """Compute forward pass.

        Args:
            x (torch.Tensor): Data observations.

        Returns:
            torch.Tensor: Output of forward pass.
        """
        h = self.fc_in(x)
        h = self.act(h)

        for layer in self.fc_hidden:
            h = layer(h)
            h = self.act(h)

        out = self.fc_out(h)
        return out


class GRU(nn.Module):
    """Gated Recurrent Unit.

    Implementation is borrowed from https://github.com/YuliaRubanova/latent_ode
    which in turn uses http://www.wildml.com/2015/10/recurrent-neural-network-
    tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
    """

    def __init__(self, input_dim, latent_dim, n_units=100):
        """Initialize GRU.

        Args:
            input_dim (int): Dimension of input.
            latent_dim (int): Dimension of latent state.
            n_units (int, optional): Number of GRU units.
        """
        super(GRU, self).__init__()

        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        self.init_network(self.update_gate)

        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        self.init_network(self.reset_gate)

        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim))
        self.init_network(self.new_state_net)

    def forward(self, x, h):
        """Compute GRU forward pass.

        Args:
            x (torch.Tensor): Input date for specific time point.
            h (torch.Tensor): Previous hidden state.
        Returns:
            torch.Tensor: Updated hidden state.
        """
        input_concat = torch.cat([h, x], -1)

        update_gate = self.update_gate(input_concat)
        reset_gate = self.reset_gate(input_concat)

        concat = torch.cat([h * reset_gate, x], -1)

        new_state = self.new_state_net(concat)

        new_y = (1 - update_gate) * new_state + update_gate * h

        return new_y

    @staticmethod
    def init_network(net):
        """Initialize network using normal distribution.

        Args:
            net (nn.Module): NN to initialize.
        """
        for module in net.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.1)
                nn.init.constant_(module.bias, val=0)


class EncoderAR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, tp, lens, args):
        args = args['aug_args']
        samp_ind = self.generate_samp_mask(x, args['s_min'], args['s_frac'])

        hid_arr = self._forward_impl(x, tp, lens, samp_ind, args['samp_prob'])
        out = self.select_by_length(hid_arr, np.array(lens))
        return out

    def predict(self, x, tp, lens, args):
        args = args['aug_args']
        samp_ind = self.generate_samp_mask(x, args['s_min'], args['s_frac'])

        hid_arr = self._forward_impl(x, tp, lens, samp_ind, 0)
        out = self.select_by_length(hid_arr, np.array(lens))
        return out

    def _forward_impl(self, x, tp, lens, samp_ind, samp_prob):
        raise NotImplementedError

    @staticmethod
    def generate_samp_mask(x, extrap_min, samp_frac):
        # Generate mask for training.
        # Points after extrapolation region index are all sampled.
        minimum_ind = int(x.size(1) * extrap_min)
        extrap_ind = np.random.randint(minimum_ind, x.size(1))

        valid_inds = np.arange(extrap_ind)
        n_inds = int(samp_frac * extrap_ind)

        samp_inds = np.sort(np.random.choice(valid_inds, n_inds, replace=False))

        return samp_inds

    @staticmethod
    def select_by_length(hid_array, lengths):
        # Used to select output from GRU output array.
        mask = torch.zeros(hid_array.size()).bool().to(hid_array.device)
        for i in range(len(mask)):
            mask[i, :lengths[i], :] = 1
        return hid_array.masked_select(mask).view(-1, hid_array.size(2))

    @staticmethod
    def get_longest_tp(tps, lens):
        ind = np.argmax(np.array(lens))
        return tps[ind]


class EncoderGRU(EncoderAR):
    def __init__(self, hidden_dim, rec_gru, rec_output, delta_t=True):
        super().__init__()

        self.gru = rec_gru
        self.out = rec_output
        self.hidden_dim = hidden_dim
        self.delta_t = delta_t

    def _forward_impl(self, x, tp, lens, samp_ind, samp_prob):
        seq_len = max(lens)

        if self.delta_t:
            delta = self.generate_delta(tp)
            x = torch.cat([x, delta], dim=2)

        h = torch.zeros(x.shape[0], self.hidden_dim).to(x.device)
        h_arr = torch.zeros(x.shape[0], seq_len, h.size(1)).to(x.device)

        for i in range(seq_len):
            if i not in samp_ind and np.random.uniform(0, 1) >= samp_prob:
                prev_out = self.out(h.unsqueeze(1)).view(x.shape[0], -1)

                if self.delta_t:
                    # Append time delta
                    prev_out = torch.cat([prev_out, x[:, i, -2:-1]], dim=-1)
                h = self.gru(prev_out, h)
            else:
                h = self.gru(x[:, i, :], h)

            h_arr[:, i, :] = h

        return self.out(h_arr)

    @staticmethod
    def generate_delta(tp):
        tp_start = torch.Tensor([0] * tp.size(0)).unsqueeze(1).float()
        tp_start = tp_start.to(tp.device)
        offset = torch.cat((tp_start, tp), dim=1)[:, :-1]
        delta = tp - offset
        return delta.unsqueeze(-1)


class EncoderGRUODE(EncoderAR):
    def __init__(self, latent_dim, rec_gru, rec_node, rec_output):
        super().__init__()

        self.gru = rec_gru
        self.node = rec_node
        self.out = rec_output
        self.latent_dim = latent_dim

    def _forward_impl(self, x, tp, lens, samp_ind, samp_prob):
        seq_len = max(lens)

        h = torch.zeros(x.size(0), self.latent_dim).to(x.device)
        h_arr = torch.zeros(x.size(0), seq_len, h.size(1)).to(x.device)

        tp = self.get_longest_tp(tp, lens)
        tp = torch.cat(((tp[0] - 0.01).unsqueeze(0), tp))

        for i in range(seq_len):
            h_ode = odeint(self.node, h, tp[i:i + 2])[1]
            if i not in samp_ind and np.random.uniform(0, 1) >= samp_prob:
                prev_out = self.out(h.unsqueeze(1)).view(x.size(0), -1)
                h = self.gru(prev_out, h_ode)
            else:
                h = self.gru(x[:, i, :], h_ode)
            h_arr[:, i, :] = h

        return self.out(h_arr)
