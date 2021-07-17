import torch.nn as nn

from baseline_models import GRU, FCNN, EncoderGRU, EncoderGRUODE
from latode_model import ODEFCNN, NeuralODE, LatentODE
from latode_model import EncoderGRUODE as EncoderLatODE

from train import TrainLoopAR, TrainLoopAE


class GRUBuilder:
    """Construct GRU."""

    def __init__(self, args):
        """Initialize all sub-modules for GRU.

        Args:
            args['obs_dim'] (int): Dimension of input data.
            args['rec_dim'] (int): Dimension of GRU hidden state.
            args['rec_gru_units'] (int): Number of units in GRU.
            args['dec_units'] (int): Number of hidden units in output NN.
            args['dec_layers'] (int): Number of layers in output NN.
            args['dec_act'] (string): Type of activation in output NN.
            args['del_t'] (boolean): Whether delta t is used as an input to GRU.

        Raises:
            ValueError: Thrown when decoder type is unsupported.
        """
        input_dim = args['obs_dim'] + int(args['del_t'])

        self.gru = GRU(input_dim, args['rec_dim'], args['rec_gru_units'])
        self.out = FCNN(args['rec_dim'], args['dec_units'], args['dec_layers'],
                        args['dec_act'], args['obs_dim'])

        self.rec_dim = args['rec_dim']
        self.del_t = args['del_t']

    def build_model(self):
        """Construct GRU with provided components.

        Returns:
            EncoderGRU: Constructed GRU.
        """
        return EncoderGRU(self.rec_dim, self.gru, self.out, self.del_t)


class GRUODEBuilder:
    """Construct GRUODE."""

    def __init__(self, args):
        """Initialize all sub-modules for GRUODE.

        Args:
            args['obs_dim'] (int): Dimension of input data.
            args['rec_dim'] (int): Dimension of encoder latent state.
            args['rec_gru_units'] (int): Number of units in encoder GRU.
            args['rec_node_units'] (int): Number of units in encoder NODE.
            args['rec_node_layers'] (int): Number of layers in encoder NODE.
            args['rec_node_act'] (str): Activations used by encoder NODE.
            args['dec_type'] (str): Type of decoder network to use.
            args['dec_units'] (int): Number of hidden units in decoder NN.
            args['dec_layers'] (int): Number of layers in decoder NN.
            args['dec_act'] (string): Activation function used in decoder NN.

        Raises:
            ValueError: Thrown when decoder type is unsupported.
        """
        self.gru = GRU(args['obs_dim'], args['rec_dim'], args['rec_gru_units'])

        self.node = ODEFCNN(args['rec_dim'], args['rec_node_units'],
                            args['rec_node_layers'], args['rec_node_act'])

        self.out = FCNN(args['rec_dim'], args['dec_units'], args['dec_layers'],
                        args['dec_act'], args['obs_dim'])

        self.rec_dim = args['rec_dim']

    def build_model(self):
        """Construct GRUODE with provided components.

        Returns:
            EncoderGRUODE: Constructed GRUODE.
        """
        return EncoderGRUODE(self.rec_dim, self.gru, self.node, self.out)


class LatentODEBuilder:
    """Construct Latent ODE."""

    def __init__(self, args):
        """Initialize all sub-modules for Latent ODE.

        Notes on hyper-parameter selection:

        The dimensionality of the encoder latent state should be >2x larger
        than decoder latent state. The decoder latent state should be close to
        the dimension of observed data.

        Args:
            args['obs_dim'] (int): Dimension of input data.
            args['rec_dim'] (int): Dimension of encoder latent state.
            args['node_dim'] (int): Dimension of node latent state.
            args['rec_gru_units'] (int): Number of units in encoder GRU.
            args['rec_node_units'] (int): Number of units in encoder NODE.
            args['rec_node_layers'] (int): Number of layers in encoder NODE.
            args['rec_node_act'] (str): Activations used by encoder NODE.
            args['rec_out_units'] (int): Number of units in rec output NN.
            args['latent_node_units'] (int): Number of units in latent NODE.
            args['latent_node_layers'] (int): Number of layers in latent NODE.
            args['latent_node_act'] (str): Activations used by latent NODE.
            args['dec_type'] (str): Type of decoder network to use.
            args['dec_units'] (int): Number of hidden units in decoder NN.
            args['dec_layers'] (int): Number of layers in decoder NN.
            args['dec_act'] (string): Activation function used in decoder NN.

        Raises:
            ValueError: Thrown when decoder type is unsupported.
        """
        enc_gru = GRU(args['rec_dim'] * 2, args['obs_dim'],
                      args['rec_gru_units'])

        enc_node = ODEFCNN(args['rec_dim'] * 2, args['rec_node_units'],
                           args['rec_node_layers'], args['rec_node_act'])

        enc_out = nn.Sequential(
            nn.Linear(args['rec_dim'] * 2, args['rec_out_units']),
            nn.Tanh(),
            nn.Linear(args['rec_out_units'], args['node_dim'] * 2)
        )

        self.enc = EncoderLatODE(args['rec_dim'], enc_gru, enc_node, enc_out)

        self.latent_node = NeuralODE(args['node_dim'],
                                     args['latent_node_units'],
                                     args['latent_node_layers'],
                                     args['latent_node_act'])

        if args['dec_type'] == 'Linear':
            self.dec = nn.Linear(args['node_latent_dim'], args['obs_dim'])
        elif args['dec_type'] == 'NN':
            self.dec = FCNN(args['node_dim'], args['dec_units'],
                            args['dec_layers'], args['dec_act'],
                            output_dim=args['obs_dim'])
        else:
            raise ValueError("Unknown or unsupported decoder type.")

    def build_model(self):
        """Construct Latent ODE with provided components.

        Returns:
            LatentODE: Constructed Latent ODE.
        """
        return LatentODE(self.enc, self.latent_node, self.dec)


MODEL_CONSTR_DICT = {
    'gru': GRUBuilder,
    'gruode': GRUODEBuilder,
    'latode': LatentODEBuilder,
    'latsegode': LatentODEBuilder,
}

MODEL_TRAIN_DICT = {
    'gru': TrainLoopAR,
    'gruode': TrainLoopAR,
    'latode': TrainLoopAE,
    'latsegode': TrainLoopAE
}
