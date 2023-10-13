import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

import diffuser.utils as utils

# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[..., None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, mish=True, n_groups=8):
        super().__init__()

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            act_fn,
        )

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."

    def __init__(self, n_channels, qk_n_channels):
        super().__init__()
        self.query_layer = nn.Conv1d(n_channels, qk_n_channels, kernel_size=1)
        self.key_layer = nn.Conv1d(n_channels, qk_n_channels, kernel_size=1)
        self.value_layer = nn.Conv1d(n_channels, n_channels, kernel_size=1)

    def forward(self, x):
        x_flat = x.reshape(x.shape[0] * x.shape[1], x.shape[2], -1)
        # Notation from the paper.
        query, key, value = (
            self.query_layer(x_flat),
            self.key_layer(x_flat),
            self.value_layer(x_flat),
        )
        query = query.reshape(x.shape[0], x.shape[1], -1)
        key = key.reshape(x.shape[0], x.shape[1], -1)
        value = value.reshape(x.shape[0], x.shape[1], -1)

        beta = F.softmax(
            torch.bmm(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1]), dim=-1
        )
        output = torch.bmm(beta, value).reshape(x.shape)
        return output


class MlpSelfAttention(nn.Module):
    def __init__(self, dim_in, dim_hidden=128):
        super().__init__()
        self.query_layer = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.key_layer = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_in),
        )

    def forward(self, x):
        x_flat = x.reshape(x.shape[0] * x.shape[1], -1)
        query, key, value = (
            self.query_layer(x_flat),
            self.key_layer(x_flat),
            self.value_layer(x_flat),
        )
        query = query.reshape(x.shape[0], x.shape[1], -1)
        key = key.reshape(x.shape[0], x.shape[1], -1)
        value = value.reshape(x.shape[0], x.shape[1], -1)

        beta = F.softmax(
            torch.bmm(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1]), dim=-1
        )
        output = torch.bmm(beta, value).reshape(x.shape)
        return output


# -----------------------------------------------------------------------------#
# ---------------------------------- sampling ---------------------------------#
# -----------------------------------------------------------------------------#


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def apply_conditioning(x, conditions, action_dim):
    # Flag variable to check if normal conditions are applied before setting
    # player_idxs or player_hoop_sides.
    apply_basic_cond = False
    for t, val in conditions.items():
        if isinstance(t, str):
            if t == "player_idxs":
                assert apply_basic_cond
                if x.shape[-1] < 4:  # pure position information w.o. player info
                    x = torch.cat([val, x], dim=-1)
                else:
                    x[:, :, :, 0] = val
            elif t == "player_hoop_sides":
                assert apply_basic_cond
                if x.shape[-1] < 4:  # pure position information w.o. player info
                    x = torch.cat([x, val], dim=-1)
                else:
                    x[:, :, :, -1] = val
            else:
                continue
        elif isinstance(t, int):
            if "agent_idx" in conditions:
                index = (
                    conditions["agent_idx"]
                    .long()
                    .unsqueeze(-1)
                    .repeat(1, 1, x.shape[-1] - action_dim)
                )
                x[:, t, :, action_dim:].scatter_(1, index, val.clone().unsqueeze(1))
            else:
                x[:, t, :, action_dim:] = val.clone()
            apply_basic_cond = True
        elif isinstance(t, tuple) or isinstance(t, list):
            assert len(t) == 2, t
            if "agent_idx" in conditions:
                index = (
                    conditions["agent_idx"]
                    .long()
                    .unsqueeze(-1)
                    .repeat(1, t[1] - t[0], 1, x.shape[-1] - action_dim)
                )
                x[:, t[0] : t[1], :, action_dim:].scatter_(
                    2, index, val.clone().unsqueeze(2)
                )
            else:
                x[:, t[0] : t[1], :, action_dim:] = val.clone()
            apply_basic_cond = True
        else:
            raise TypeError(type(t))
    return x


# -----------------------------------------------------------------------------#
# ---------------------------------- losses -----------------------------------#
# -----------------------------------------------------------------------------#


class WeightedLoss(nn.Module):
    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer("weights", weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        """
        pred, targ : tensor
            [ batch_size x horizon x transition_dim ]
        """
        loss = self._loss(pred, targ)
        # weighted_loss = (loss * self.weights).mean()
        if self.action_dim > 0:
            a0_loss = (
                loss[:, 0, : self.action_dim] / self.weights[0, : self.action_dim]
            ).mean()
            info = {"a0_loss": a0_loss}
        else:
            info = {}
        return loss * self.weights, info
        # return weighted_loss, {"a0_loss": a0_loss}


class WeightedStateLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, pred, targ):
        """
        pred, targ : tensor
            [ batch_size x horizon x transition_dim ]
        """
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        return loss * self.weights, {"a0_loss": weighted_loss}
        # return weighted_loss, {"a0_loss": weighted_loss}


class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(), utils.to_np(targ).squeeze()
            )[0, 1]
        else:
            corr = np.NaN

        info = {
            "mean_pred": pred.mean(),
            "mean_targ": targ.mean(),
            "min_pred": pred.min(),
            "min_targ": targ.min(),
            "max_pred": pred.max(),
            "max_targ": targ.max(),
            "corr": utils.to_torch(corr, device=pred.device),
        }

        return loss, info


class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


class WeightedStateL2(WeightedStateLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


class ValueL1(ValueLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class ValueL2(ValueLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


Losses = {
    "l1": WeightedL1,
    "l2": WeightedL2,
    "state_l2": WeightedStateL2,
    "value_l1": ValueL1,
    "value_l2": ValueL2,
}
