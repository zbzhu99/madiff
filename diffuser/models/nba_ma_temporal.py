from copy import copy
from typing import Tuple

import einops
import torch
from torch import nn
from torch.distributions import Bernoulli

from .helpers import (
    Conv1dBlock,
    Downsample1d,
    SinusoidalPosEmb,
    Upsample1d,
    SelfAttention,
)
from .nba_temporal import ResidualPlayerTemporalBlock


class PlayerConvAttentionDeconv(nn.Module):
    agent_share_parameters = False

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        cond_dim: int,
        dim: int = 128,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        n_agents: int = 2,
        returns_condition: bool = False,
        condition_dropout: float = 0.1,
        calc_energy: bool = False,
        kernel_size: int = 5,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.condition_dropout = condition_dropout

        dims = [transition_dim + 1, *map(lambda m: dim * m, dim_mults)]
        player_dim = dim
        time_dim = dim
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        if calc_energy:
            mish = False
            act_fn = nn.SiLU()
        else:
            mish = True
            act_fn = nn.Mish()

        self.time_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    SinusoidalPosEmb(dim),
                    nn.Linear(dim, dim * 4),
                    act_fn,
                    nn.Linear(dim * 4, dim),
                )
                for _ in range(n_agents)
            ]
        )
        # 500 is a number that larger than maximum player id
        self.player_embedding = nn.Embedding(500, player_dim)

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy

        if self.returns_condition:
            self.returns_mlp = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(1, dim),
                        act_fn,
                        nn.Linear(dim, dim * 4),
                        act_fn,
                        nn.Linear(dim * 4, dim),
                    )
                    for _ in range(n_agents)
                ]
            )

            self.mask_dist = Bernoulli(probs=1 - self.condition_dropout)

        self.downs = nn.ModuleList([nn.ModuleList([]) for _ in range(n_agents)])
        self.ups = nn.ModuleList([nn.ModuleList([]) for _ in range(n_agents)])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            for i in range(n_agents):
                self.downs[i].append(
                    nn.ModuleList(
                        [
                            ResidualPlayerTemporalBlock(
                                dim_in,
                                dim_out,
                                kernel_size=kernel_size,
                                time_embed_dim=time_dim,
                                player_embed_dim=player_dim,
                            ),
                            ResidualPlayerTemporalBlock(
                                dim_out,
                                dim_out,
                                kernel_size=kernel_size,
                                time_embed_dim=time_dim,
                                player_embed_dim=player_dim,
                            ),
                            Downsample1d(dim_out) if not is_last else nn.Identity(),
                        ]
                    )
                )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = nn.ModuleList(
            [
                ResidualPlayerTemporalBlock(
                    mid_dim,
                    mid_dim,
                    time_embed_dim=time_dim,
                    player_embed_dim=player_dim,
                    kernel_size=kernel_size,
                )
                for _ in range(n_agents)
            ]
        )
        self.mid_block2 = nn.ModuleList(
            [
                ResidualPlayerTemporalBlock(
                    mid_dim,
                    mid_dim,
                    time_embed_dim=time_dim,
                    player_embed_dim=player_dim,
                    kernel_size=kernel_size,
                )
                for _ in range(n_agents)
            ]
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            for i in range(n_agents):
                self.ups[i].append(
                    nn.ModuleList(
                        [
                            ResidualPlayerTemporalBlock(
                                dim_out * 2,
                                dim_in,
                                time_embed_dim=time_dim,
                                player_embed_dim=player_dim,
                                kernel_size=kernel_size,
                            ),
                            ResidualPlayerTemporalBlock(
                                dim_in,
                                dim_in,
                                time_embed_dim=time_dim,
                                player_embed_dim=player_dim,
                                kernel_size=kernel_size,
                            ),
                            Upsample1d(dim_in) if not is_last else nn.Identity(),
                        ]
                    )
                )

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.ModuleList(
            [
                nn.Sequential(
                    Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
                    nn.Conv1d(dim, transition_dim, 1),
                )
                for _ in range(n_agents)
            ]
        )

        self.self_attn = [SelfAttention(in_out[-1][1], in_out[-1][1] // 16)]
        for dims in reversed(in_out):
            self.self_attn.append(SelfAttention(dims[1], dims[1] // 16))
        self.self_attn = nn.ModuleList(self.self_attn)

    def forward(
        self,
        x,
        time,
        returns=None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        **kwargs,
    ):
        """
        x : [ batch x horizon x agent x transition ]
        returns : [batch x horizon x agent]
        """

        assert (
            x.shape[2] == self.n_agents
        ), f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"

        if self.calc_energy:
            x_inp = copy(x)

        x = einops.rearrange(x, "b t a f -> b a f t")

        player_idxs = x[:, :, 0, 0].to(dtype=torch.int)
        player_idxs = [player_idxs[:, a_idx] for a_idx in range(player_idxs.shape[1])]
        player_id_embd = [
            self.player_embedding(player_idxs[i]) for i in range(self.n_agents)
        ]

        t = [self.time_mlp[i](time) for i in range(self.n_agents)]
        x = x[:, :, 1:, :]
        x = [x[:, a_idx] for a_idx in range(x.shape[1])]  # a, b f t

        if self.returns_condition:
            assert returns is not None
            returns_embed = [
                self.returns_mlp[i](returns[:, :, i]) for i in range(self.n_agents)
            ]
            if use_dropout:
                # here use the same mask for all agents
                mask = self.mask_dist.sample(
                    sample_shape=(returns_embed[0].size(0), 1)
                ).to(returns_embed[0].device)
                returns_embed = [
                    returns_embed[i] * mask for i in range(len(returns_embed))
                ]
            if force_dropout:
                returns_embed = [
                    returns_embed[i] * 0 for i in range(len(returns_embed))
                ]

            t = [torch.cat([t[i], returns_embed[i]], dim=-1) for i in range(len(t))]

        h = [[] for _ in range(self.n_agents)]

        for layer_idx in range(len(self.downs[0])):
            for i in range(self.n_agents):
                resnet, resnet2, downsample = self.downs[i][layer_idx]
                x[i] = resnet(x[i], player_id_embd[i], t[i])
                x[i] = resnet2(x[i], player_id_embd[i], t[i])
                h[i].append(x[i])
                x[i] = downsample(x[i])

        for i in range(self.n_agents):
            x[i] = self.mid_block1[i](x[i], player_id_embd[i], t[i])
            x[i] = self.mid_block2[i](x[i], player_id_embd[i], t[i])

        x = self.self_attn[0](torch.stack(x, dim=1))  # b a f t
        x = [x[:, a_idx] for a_idx in range(x.shape[1])]  # a, b f t

        for layer_idx in range(len(self.ups[0])):
            hiddens = torch.stack([hid.pop() for hid in h], dim=1)  # b a f t
            hiddens = self.self_attn[layer_idx + 1](hiddens)
            for i in range(self.n_agents):
                resnet, resnet2, upsample = self.ups[i][layer_idx]
                x[i] = torch.cat((x[i], hiddens[:, i]), dim=1)
                x[i] = resnet(x[i], player_id_embd[i], t[i])
                x[i] = resnet2(x[i], player_id_embd[i], t[i])
                x[i] = upsample(x[i])

        for i in range(self.n_agents):
            x[i] = self.final_conv[i](x[i])

        x = torch.stack(x, dim=1)
        x = einops.rearrange(x, "b a f t -> b t a f")

        if self.calc_energy:
            # Energy function
            energy = ((x - x_inp) ** 2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x_inp, create_graph=True)
            return grad[0]
        else:
            return x


class PlayerSharedConvAttentionDeconv(nn.Module):
    agent_share_parameters = True

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        cond_dim: int,
        dim: int = 128,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        nhead: int = 4,
        n_agents: int = 2,
        use_attention: bool = True,
        returns_condition: bool = False,
        condition_dropout: float = 0.1,
        calc_energy: bool = False,
        kernel_size: int = 5,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.use_attention = use_attention

        self.condition_dropout = condition_dropout

        dims = [transition_dim + 1, *map(lambda m: dim * m, dim_mults)]
        player_dim = dim
        time_dim = dim
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        if calc_energy:
            mish = False
            act_fn = nn.SiLU()
        else:
            mish = True
            act_fn = nn.Mish()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )
        # 500 is the maximum player id
        self.player_embedding = nn.Embedding(500, player_dim)

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy

        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                nn.Linear(1, dim),
                act_fn,
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),
            )
            self.mask_dist = Bernoulli(probs=1 - self.condition_dropout)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualPlayerTemporalBlock(
                            dim_in,
                            dim_out,
                            kernel_size=kernel_size,
                            time_embed_dim=time_dim,
                            player_embed_dim=player_dim,
                        ),
                        ResidualPlayerTemporalBlock(
                            dim_out,
                            dim_out,
                            kernel_size=kernel_size,
                            time_embed_dim=time_dim,
                            player_embed_dim=player_dim,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualPlayerTemporalBlock(
            mid_dim,
            mid_dim,
            time_embed_dim=time_dim,
            player_embed_dim=player_dim,
            kernel_size=kernel_size,
        )
        self.mid_block2 = ResidualPlayerTemporalBlock(
            mid_dim,
            mid_dim,
            time_embed_dim=time_dim,
            player_embed_dim=player_dim,
            kernel_size=kernel_size,
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualPlayerTemporalBlock(
                            dim_out * 2,
                            dim_in,
                            time_embed_dim=time_dim,
                            player_embed_dim=player_dim,
                            kernel_size=kernel_size,
                        ),
                        ResidualPlayerTemporalBlock(
                            dim_in,
                            dim_in,
                            time_embed_dim=time_dim,
                            player_embed_dim=player_dim,
                            kernel_size=kernel_size,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, transition_dim, 1),
        )

        self.self_attn = [SelfAttention(in_out[-1][1], in_out[-1][1] // 16)]
        for dims in reversed(in_out):
            self.self_attn.append(SelfAttention(dims[1], dims[1] // 16))
        self.self_attn = nn.ModuleList(self.self_attn)

    def forward(
        self,
        x,
        time,
        returns=None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        **kwargs,
    ):
        """
        x : [ batch x horizon x agent x transition ]
        returns : [batch x horizon x agent]
        """

        assert (
            x.shape[2] == self.n_agents
        ), f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"

        if self.calc_energy:
            x_inp = copy(x)
        x = einops.rearrange(x, "b t a f -> b a f t")
        player_idxs = x[:, :, 0, 0].to(dtype=torch.int)  # bz, n_agent
        x = x[:, :, 1:, :]
        bs = x.shape[0]

        t = self.time_mlp(torch.stack([time for _ in range(x.shape[1])], dim=1))
        player_id_embd = self.player_embedding(player_idxs)

        if self.returns_condition:
            assert returns is not None
            returns = einops.rearrange(returns, "b t a -> b a t")
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                # here use the same mask for all agents
                mask = self.mask_dist.sample(
                    sample_shape=(returns_embed.size(0), returns_embed.size(1), 1)
                ).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed

            t = torch.cat([t, returns_embed], dim=-1)

        h = []
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        player_embd = player_id_embd.reshape(
            player_id_embd.shape[0] * player_id_embd.shape[1], player_id_embd.shape[2]
        )
        t = t.reshape(t.shape[0] * t.shape[1], t.shape[2])
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, player_embd, t)
            x = resnet2(x, player_embd, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, player_embd, t)
        x = self.mid_block2(x, player_embd, t)

        x = x.reshape(bs, x.shape[0] // bs, x.shape[1], x.shape[2])
        x = self.self_attn[0](x)  # b a f t

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        for layer_idx in range(len(self.ups)):
            hiddens = h.pop()
            hiddens = hiddens.reshape(
                bs, hiddens.shape[0] // bs, hiddens.shape[1], hiddens.shape[2]
            )
            hiddens = self.self_attn[layer_idx + 1](hiddens)
            hiddens = hiddens.reshape(
                hiddens.shape[0] * hiddens.shape[1], hiddens.shape[2], hiddens.shape[3]
            )
            resnet, resnet2, upsample = self.ups[layer_idx]
            x = torch.cat((x, hiddens), dim=1)
            x = resnet(x, player_embd, t)
            x = resnet2(x, player_embd, t)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.reshape(bs, x.shape[0] // bs, x.shape[1], x.shape[2])

        x = einops.rearrange(x, "b a f t -> b t a f")

        if self.calc_energy:
            # Energy function
            energy = ((x - x_inp) ** 2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x_inp, create_graph=True)
            return grad[0]
        else:
            return x


class PlayerSharedConvAttentionTemporalValue(nn.Module):
    agent_share_parameters = True

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        n_agents,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim + 1, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        player_dim = dim
        self.n_agents = n_agents
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        # 500 is the maximum player id
        self.player_embedding = nn.Embedding(500, player_dim)

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        print("ConvAttentionTemporalValue: ", in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(
                nn.ModuleList(
                    [
                        ResidualPlayerTemporalBlock(
                            dim_in,
                            dim_out,
                            kernel_size=5,
                            time_embed_dim=time_dim,
                            player_embed_dim=player_dim,
                        ),
                        ResidualPlayerTemporalBlock(
                            dim_out,
                            dim_out,
                            kernel_size=5,
                            time_embed_dim=time_dim,
                            player_embed_dim=player_dim,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 4
        mid_dim_3 = mid_dim // 16

        self.mid_block1 = ResidualPlayerTemporalBlock(
            mid_dim,
            mid_dim_2,
            kernel_size=5,
            time_embed_dim=time_dim,
            player_embed_dim=player_dim,
        )
        self.mid_block2 = ResidualPlayerTemporalBlock(
            mid_dim_2,
            mid_dim_3,
            kernel_size=5,
            time_embed_dim=time_dim,
            player_embed_dim=player_dim,
        )
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + player_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )
        self.self_attn = nn.ModuleList(
            [SelfAttention(dim[1], dim[1] // 16) for dim in in_out]
        )

    def forward(self, x, time, *args):
        """
        x : [ batch x horizon x n_agents x transition ]
        """

        assert (
            x.shape[2] == self.n_agents
        ), f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"

        x = einops.rearrange(x, "b t a f -> b a f t")
        player_idxs = x[:, :, 0, 0].to(dtype=torch.int)  # bz, n_agent
        x = x[:, :, 1:, :]
        assert x.shape[-2] == 3  # observation is (x, y, hoop_side)

        bs = x.shape[0]

        t = self.time_mlp(torch.stack([time for _ in range(x.shape[1])], dim=1))
        player_id_embd = self.player_embedding(player_idxs)

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        player_embd = player_id_embd.reshape(
            player_id_embd.shape[0] * player_id_embd.shape[1], player_id_embd.shape[2]
        )
        t = t.reshape(t.shape[0] * t.shape[1], t.shape[2])

        for layer_idx, (resnet, resnet2, downsample) in enumerate(self.blocks):
            x = resnet(x, player_embd, t)
            x = resnet2(x, player_embd, t)
            x = downsample(x)
            x = x.reshape(bs, x.shape[0] // bs, x.shape[1], x.shape[2])
            x = self.self_attn[layer_idx](x)
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])

        x = self.mid_block1(x, player_embd, t)
        x = self.mid_block2(x, player_embd, t)

        x = x.view(len(x), -1)
        x = self.final_block(
            torch.cat([x, player_embd, t], dim=-1)
        )  # x.shape[0] * x.shape[1], 1

        x = x.reshape(bs, -1)  # x.shape[0], x.shape[1], 1
        # take mean over agents
        out = x.mean(axis=1, keepdim=True)  # x.shape[0], 1

        return out
