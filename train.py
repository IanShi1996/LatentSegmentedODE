import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from augment import augment
from estimators import get_elbo
from utils import RunningAverageMeter


class TrainLoopBase:
    """Generic training loop for experiments."""

    def __init__(self, model, train_loader, val_loader, device, plot_func=None,
                 loss_meters=None, elbo_meters=None, loss_hists=None,
                 elbo_hists=None):
        """Initialize main training loop.

        Dataloaders should return tuple of data and time points with shapes:
            ((B x L x D), (B x L)) where
            B = Batch size, L = Number of observations, D = Data dimension.
        TODO: Move above to subclass docs.

        Args:
            model (nn.Module): Model to train.
            device (torch.device): GPU to use.
            train_loader (torch.utils.data.Dataloader): Training dataloader.
            val_loader (torch.utils.data.Dataloader): Validation dataloader.
            plot_func (function): Function used to plot predictions.
            loss_meters (RunningAverageMeter, RunningAverageMeter):
                Existing training / val loss running average meters.
            elbo_meters (RunningAverageMeter, RunningAverageMeter):
                Existing training / val elbo running average meters.
            loss_hists (list, list): Train/val loss history arrays.
            elbo_hists (list, list): Train/val elbo history arrays.
        """

        self.model = model
        self.device = device

        self.train_elbo_meter, self.val_elbo_meter = None, None
        self.train_elbo_hist, self.val_elbo_hist = None, None
        self.train_loss_meter, self.val_loss_meter = None, None
        self.train_loss_hist, self.val_loss_hist = None, None

        self.init_history(loss_hists, elbo_hists)
        self.init_meter(loss_meters, elbo_meters)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.plot_func = plot_func
        self.execution_arg_history = []

    def init_history(self, loss_hist, elbo_hist):
        """Initialize lists tracking loss and elbo history.

        Args:
            loss_hist (list, list): Previous loss history. Optional.
            elbo_hist (list, list): Previous elbo history. Optional.
        """
        if loss_hist is None:
            self.train_loss_hist = []
            self.val_loss_hist = []
        else:
            self.train_loss_hist = loss_hist[0]
            self.val_loss_hist = loss_hist[1]

        if elbo_hist is None:
            self.train_elbo_hist = []
            self.val_elbo_hist = []
        else:
            self.train_elbo_hist = elbo_hist[0]
            self.val_elbo_hist = elbo_hist[1]

    def init_meter(self, loss_meters, elbo_meters):
        """Initialize running loss and elbo meters.

        Args:
            loss_meters (RunningAverageMeter, RunningAverageMeter):
                Previous loss meters.
            elbo_meters (RunningAverageMeter, RunningAverageMeter):
                Previous elbo meters.
        """
        if loss_meters is None:
            self.train_loss_meter = RunningAverageMeter()
            self.val_loss_meter = RunningAverageMeter(0.5)
        else:
            self.train_loss_meter = loss_meters[0]
            self.val_loss_meter = loss_meters[1]

        if elbo_meters is None:
            self.train_elbo_meter = RunningAverageMeter()
            self.val_elbo_meter = RunningAverageMeter(0.5)
        else:
            self.train_elbo_meter = elbo_meters[0]
            self.val_elbo_meter = elbo_meters[1]

    def print_loss(self, epoch):
        """Print train and validation loss and elbo."""
        out_str = 'Epoch: {}, Train ELBO: {:.3f}, Val ELBO: {:.3f}, ' \
                  'Train MSE: {:.3f}, Val MSE: {:.3f}'

        print(out_str.format(epoch, -self.train_elbo_meter.avg,
                             -self.val_elbo_meter.avg,
                             self.train_loss_meter.avg,
                             self.val_loss_meter.avg), flush=True)

    def plot_loss(self):
        """Plot training and validation loss and elbo history."""
        train_elbo_range = range(len(self.train_elbo_hist))
        val_elbo_range = range(len(self.val_elbo_hist))
        train_loss_range = range(len(self.train_loss_hist))
        val_loss_range = range(len(self.val_loss_hist))

        fig, ax = plt.subplots(2, 2)
        ax[0][0].plot(train_elbo_range, self.train_elbo_hist)
        ax[0][0].title.set_text("Train ELBO")
        ax[0][1].plot(val_elbo_range, self.val_elbo_hist)
        ax[0][1].title.set_text("Val ELBO")
        ax[1][0].plot(train_loss_range, self.train_loss_hist)
        ax[1][0].title.set_text("Train MSE")
        ax[1][1].plot(val_loss_range, self.val_loss_hist)
        ax[1][1].title.set_text("Val MSE")
        plt.tight_layout()
        plt.show()

    def train(self, optimizer, args, scheduler=None, verbose=True,
              plt_traj=False, plt_loss=False):
        raise NotImplementedError

    def update_val_loss(self, args):
        raise NotImplementedError

    def plot_val_traj(self, args):
        raise NotImplementedError


class TrainLoopAR(TrainLoopBase):
    def __init__(self, model, train_loader, val_loader, device, plot_func=None,
                 loss_meters=None, elbo_meters=None, loss_hists=None,
                 elbo_hists=None):
        super().__init__(model, train_loader, val_loader, device, plot_func,
                         loss_meters, elbo_meters, loss_hists, elbo_hists)

    def train(self, optimizer, args, scheduler=None, verbose=True,
              plt_traj=False, plt_loss=False):
        for epoch in range(1, args['max_epochs'] + 1):
            optimizer.zero_grad()

            for b_num, (b_data, b_tp, b_len) in enumerate(self.train_loader):
                optimizer.zero_grad()

                d_tt = b_data.float().to(self.device)
                tp_tt = b_tp.float().to(self.device)

                pred_out = self.model.forward(d_tt, tp_tt, b_len, args)
                data_out = self.model.select_by_length(d_tt, b_len)

                loss = nn.MSELoss()(pred_out, data_out)

                self.train_loss_meter.update(loss.item())

                loss.backward()
                if 'clip_norm' in args and args['clip_norm']:
                    clip_grad_norm_(self.model.parameters(), args['clip_norm'])
                optimizer.step()

            if scheduler:
                scheduler.step()

            with torch.no_grad():
                self.update_val_loss(args)

                if self.plot_func and plt_traj:
                    self.plot_val_traj(args['plt_args'])
                    plt.show()
                self.train_loss_hist.append(self.train_loss_meter.avg)
                self.val_loss_hist.append(self.val_loss_meter.val)

            if verbose:
                if scheduler:
                    print("Current LR: {}".format(scheduler.get_last_lr()),
                          flush=True)
                if plt_loss:
                    self.plot_loss()
                self.print_loss(epoch)

    def update_val_loss(self, args):
        with torch.no_grad():
            losses = []
            for i, (b_data, b_tp, b_len) in enumerate(self.val_loader):
                d_tt = b_data.float().to(self.device)
                tp_tt = b_tp.float().to(self.device)

                pred_out = self.model.predict(d_tt, tp_tt, b_len, args)
                data_out = self.model.select_by_length(d_tt, b_len)

                loss = nn.MSELoss()(pred_out, data_out)
                losses.append(loss.item())
            self.val_loss_meter.update(np.mean(losses))

    def plot_val_traj(self, args):
        raise NotImplementedError


class TrainLoopAE(TrainLoopBase):
    def __init__(self, model, train_loader, val_loader, device, plot_func=None,
                 loss_meters=None, elbo_meters=None, loss_hists=None,
                 elbo_hists=None):
        super().__init__(model, train_loader, val_loader, device, plot_func,
                         loss_meters, elbo_meters, loss_hists, elbo_hists)

    def train(self, optimizer, args, scheduler=None, verbose=True,
              plt_traj=False, plt_loss=False):
        for epoch in range(1, args['max_epochs'] + 1):
            optimizer.zero_grad()

            for b_num, (b_data, b_tp, b_len) in enumerate(self.train_loader):
                optimizer.zero_grad()

                if len(args['aug_methods']) > 0:
                    b_data, b_tp, b_len = augment(b_data, b_tp, b_len,
                                                  args['aug_methods'],
                                                  args['aug_args'])

                d_tt = b_data.float().to(self.device)
                tp_tt = b_tp.float().to(self.device)

                tp_union = torch.unique(tp_tt)

                if args['aligned_data']:
                    mask, exp_d, _ = self.generate_mask_aligned(d_tt, b_len,
                                                                tp_union)
                else:
                    mask, exp_d, _ = self.generate_mask(d_tt, tp_tt, b_len,
                                                        tp_union)

                mask = torch.Tensor(mask).float().to(self.device)
                exp_d = exp_d.float().to(self.device)

                out = self.model.forward(exp_d, tp_union, mask)

                select_mask = (mask == 1).unsqueeze(2)
                p_flat = torch.masked_select(out[0], select_mask)
                d_flat = torch.masked_select(exp_d, select_mask)

                elbo = get_elbo(d_flat, p_flat, out[1], out[2], args['l_std'],
                                min(1, epoch / args['kl_burn_max']))

                with torch.no_grad():
                    loss = nn.MSELoss()(d_flat, p_flat)

                self.train_elbo_meter.update(elbo.item())
                self.train_loss_meter.update(loss.item())

                elbo.backward()
                if 'clip_norm' in args and args['clip_norm']:
                    clip_grad_norm_(self.model.parameters(), args['clip_norm'])
                optimizer.step()

            if scheduler:
                scheduler.step()

            with torch.no_grad():
                self.update_val_loss(args)

                if self.plot_func and plt_traj:
                    self.plot_val_traj(args['plt_args'])
                    plt.show()

                self.train_loss_hist.append(self.train_loss_meter.avg)
                self.train_elbo_hist.append(self.train_elbo_meter.avg)
                self.val_loss_hist.append(self.val_loss_meter.avg)
                self.val_elbo_hist.append(self.val_elbo_meter.avg)

            if verbose:
                if scheduler:
                    print("Current LR: {}".format(scheduler.get_last_lr()),
                          flush=True)
                if plt_loss:
                    self.plot_loss()
                self.print_loss(epoch)

    def update_val_loss(self, args):
        with torch.no_grad():
            losses = []
            elbos = []
            for i, (b_data, b_tp, b_len) in enumerate(self.val_loader):
                d_tt = b_data.float().to(self.device)
                tp_tt = b_tp.float().to(self.device)

                tp_union = torch.unique(tp_tt)

                if args['aligned_data']:
                    mask, exp_d, _ = self.generate_mask_aligned(d_tt, b_len,
                                                                tp_union)
                else:
                    mask, exp_d, _ = self.generate_mask(d_tt, tp_tt, b_len,
                                                        tp_union)

                mask = torch.Tensor(mask).float().to(self.device)
                exp_d = exp_d.float().to(self.device)

                out = self.model.forward(exp_d, tp_union, mask)

                select_mask = (mask == 1).unsqueeze(2)
                p_flat = torch.masked_select(out[0], select_mask)
                d_flat = torch.masked_select(exp_d, select_mask)

                elbo = get_elbo(d_flat, p_flat, out[1], out[2],
                                args['l_std'], 1)

                loss = nn.MSELoss()(d_flat, p_flat)

                losses.append(loss.item())
                elbos.append(elbo.item())

            self.val_elbo_meter.update(np.mean(elbos))
            self.val_loss_meter.update(np.mean(losses))

    def plot_val_traj(self, args):
        raise NotImplementedError

    @staticmethod
    def generate_mask(data, tps, length, tp_union):
        """Generates mask to process irregularly sampled data.

        Converts irregularly observed data by mapping each observation in a
        data trajectory its position in the union of all times of observation.
        This method is slow and should be optimized.

        Outputs a binary array mask of shape B x L, where B is batch size, and
        L is the length of tp_union. Mask denotes if data is observed at the
        particular time in the timepoint union.

        Outputs an expanded data tensor which inserts 0's in each data
        trajectory at times where it is not observed in the timepoint union.

        Finally, outputs the indices in the timepoint union where each data
        trajectory is observed, used for later reconstruction.

        Args:
            data (torch.Tensor): Input data.
            tps (torch.Tensor): Times of data observation.
            length (list of int): Number of samples per trajectory.
            tp_union (torch.Tensor): Union of all times of observation.

        Returns:
            np.ndarray, torch.Tensor, list of np.ndarray: mask, exp_data, r_arr.
        """
        tp_map = {tp_union[i].item(): i for i in range(len(tp_union))}

        mask = np.zeros((data.shape[0], tp_union.shape[0]))
        e_data = torch.zeros((data.shape[0], tp_union.shape[0], data.shape[2]))
        e_data = e_data.to(data.device)
        r_arr = []

        for i in range(len(mask)):
            inds = [tp_map[tps[i][j].item()] for j in range(length[i])]
            mask[i, inds] = 1
            e_data[i, inds] = data[i, :length[i]]
            r_arr.append(np.where(mask[i] == 1)[0])

        return mask, e_data, r_arr

    @staticmethod
    def generate_mask_aligned(data, length, tp_union):
        """Generates mask assuming aligned data.

        Generates mask for data which is regularly sampled, but contains
        ragged ends (sequences are of different length). Should result in
        identical output to generic mask generator, but much faster.

        Args:
            data (torch.Tensor): Input data.
            length (list of int): Number of samples per trajectory.
            tp_union (torch.Tensor): Union of all times of observation.

        Returns:
            np.ndarray, torch.Tensor, list of np.ndarray: mask, exp_data, r_arr.
        """
        mask = np.zeros((data.shape[0], tp_union.shape[0]))
        e_data = torch.zeros((data.shape[0], tp_union.shape[0], data.shape[2]))
        e_data = e_data.to(data.device)
        r_arr = []

        for i, l in enumerate(length):
            mask[i, :l] = 1
            e_data[i, :l] = data[i, :l]
            r_arr.append(np.where(mask[i] == 1)[0])

        return mask, e_data, r_arr

    @staticmethod
    def select_from_flat(ind, flat_data, lengths):
        """Selects trajectory from flattened array by index.

        Args:
            ind (int): Index of desired trajectory.
            flat_data (torch.Tensor): Flattened 1-D Tensor of trajectories.
            lengths (list of int): Number of samples per trajectory.

        Returns:
            torch.Tensor: Trajectory at index.
        """
        base = sum(lengths[:ind])
        return flat_data[base:base+lengths[ind]]
