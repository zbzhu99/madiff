import torch.nn as nn
from einops.layers.torch import Rearrange

from .helpers import Conv1dBlock


class ResidualPlayerTemporalBlock(nn.Module):
    def __init__(
        self,
        inp_channels,
        out_channels,
        time_embed_dim,
        player_embed_dim,
        kernel_size=5,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size),
            ]
        )

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_embed_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.player_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(
                player_embed_dim, out_channels
            ),  # additional dimension for player_hoop_sides
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, p_embed, t):
        """
        x : [ batch_size x inp_channels x horizon ]
        p_idx: [batch_size x player_embed_dim ]
        t : [ batch_size x time_embed_dim ]
        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x) + self.time_mlp(t) + self.player_mlp(p_embed)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)
