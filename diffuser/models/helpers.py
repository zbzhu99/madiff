import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
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
    def __init__(
        self,
        n_channels: int,
        qk_n_channels: int,
        v_n_channels: int,
        nheads: int = 4,
        residual: bool = False,
    ):
        super().__init__()
        self.nheads = nheads
        self.query_layer = nn.Conv1d(n_channels, qk_n_channels * nheads, kernel_size=1)
        self.key_layer = nn.Conv1d(n_channels, qk_n_channels * nheads, kernel_size=1)
        self.value_layer = nn.Conv1d(n_channels, v_n_channels * nheads, kernel_size=1)
        self.attend = nn.Softmax(dim=-1)
        self.residual = residual
        if residual:
            self.gamma = nn.Parameter(torch.zeros([1]))

    def forward(self, x, states: torch.Tensor = None):
        x_flat = rearrange(x, "b a f t -> (b a) f t")
        query, key, value = (
            self.query_layer(x_flat),
            self.key_layer(x_flat),
            self.value_layer(x_flat),
        )

        query = rearrange(
            query, "(b a) (h d) t -> h b a (d t)", h=self.nheads, a=x.shape[1]
        )
        key = rearrange(
            key, "(b a) (h d) t -> h b a (d t)", h=self.nheads, a=x.shape[1]
        )
        value = rearrange(
            value, "(b a) (h d) t -> h b a (d t)", h=self.nheads, a=x.shape[1]
        )

        dots = einsum(query, key, "h b a1 f, h b a2 f -> h b a1 a2") / math.sqrt(
            query.shape[-1]
        )
        attn = self.attend(dots)
        out = einsum(attn, value, "h b a1 a2, h b a2 f -> h b a1 f")

        out = rearrange(out, "h b a f -> b a (h f)")
        out = out.reshape(x.shape)
        if self.residual:
            out = x + self.gamma * out
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout: float = 0, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


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


def apply_conditioning(x, conditions):
    cond_masks = conditions["masks"].to(bool)
    x[cond_masks] = conditions["x"][cond_masks].clone()

    if "player_idxs" in conditions.keys():
        if x.shape[-1] < 4:  # pure position information w.o. player info
            x = torch.cat([conditions["player_idxs"], x], dim=-1)
            x = torch.cat([x, conditions["player_hoop_sides"]], dim=-1)
        else:
            x[:, :, :, 0] = conditions["player_idxs"]
            x[:, :, :, -1] = conditions["player_hoop_sides"]

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
