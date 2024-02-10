from typing import Tuple

import einops
import torch
import torch.nn as nn

from .temporal import TemporalUnet, TemporalValue


class ConcatenatedTemporalUnet(nn.Module):
    agent_share_parameters = False

    def __init__(
        self,
        n_agents: int,
        horizon: int,
        transition_dim: int,
        dim: int = 128,
        history_horizon: int = 0,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        returns_condition: bool = False,
        env_ts_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        residual_attn: bool = False,  # not used here
        use_layer_norm: bool = False,
        max_path_length: int = 100,
        use_temporal_attention: bool = False,  # not used here
    ):
        super().__init__()

        self.n_agents = n_agents
        self.history_horizon = history_horizon
        self.use_temporal_attention = use_temporal_attention

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition

        self.net = TemporalUnet(
            horizon=horizon,
            history_horizon=history_horizon,
            transition_dim=transition_dim * n_agents,
            dim=dim,
            dim_mults=dim_mults,
            returns_condition=returns_condition,
            env_ts_condition=env_ts_condition,
            condition_dropout=condition_dropout,
            kernel_size=kernel_size,
            max_path_length=max_path_length,
        )

    def forward(
        self,
        x,
        time,
        returns=None,
        env_timestep=None,
        attention_masks=None,
        use_dropout: bool = True,
        force_dropout: bool = False,
    ):
        """
        x : [ batch x horizon x agent x transition ]
        returns : [batch x 1 x agent]
        """

        assert x.shape[2] == self.n_agents, f"{x.shape}, {self.n_agents}"

        concat_x = einops.rearrange(x, "b h a f -> b h (a f)")
        concat_x = self.net(
            concat_x,
            time=time,
            returns=returns.mean(dim=2) if returns is not None else None,
            env_timestep=env_timestep,
            use_dropout=use_dropout,
            force_dropout=force_dropout,
        )
        x = einops.rearrange(concat_x, "b h (a f) -> b h a f", a=self.n_agents)

        return x


class IndependentTemporalUnet(nn.Module):
    agent_share_parameters = False

    def __init__(
        self,
        n_agents: int,
        horizon: int,
        transition_dim: int,
        dim: int = 128,
        history_horizon: int = 0,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        returns_condition: bool = False,
        env_ts_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        residual_attn: bool = False,  # not used here
        max_path_length: int = 100,
        use_temporal_attention: bool = False,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.history_horizon = history_horizon
        self.use_temporal_attention = use_temporal_attention

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition

        self.nets = nn.ModuleList(
            [
                TemporalUnet(
                    horizon=horizon,
                    history_horizon=history_horizon,
                    transition_dim=transition_dim,
                    dim=dim,
                    dim_mults=dim_mults,
                    returns_condition=returns_condition,
                    env_ts_condition=env_ts_condition,
                    condition_dropout=condition_dropout,
                    kernel_size=kernel_size,
                    max_path_length=max_path_length,
                )
                for _ in range(n_agents)
            ]
        )

    def forward(
        self,
        x,
        time,
        returns=None,
        env_timestep=None,
        attention_masks=None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        **kwargs,
    ):
        """
        x : [ batch x horizon x agent x transition ]
        returns : [batch x agent x horizon]
        """

        assert x.shape[2] == self.n_agents, f"{x.shape}, {self.n_agents}"

        x_list = []
        for a_idx in range(self.n_agents):
            x_list.append(
                self.nets[a_idx](
                    x[:, :, a_idx, :],
                    time=time,
                    returns=returns[:, :, a_idx] if returns is not None else None,
                    env_timestep=env_timestep,
                    use_dropout=use_dropout,
                    force_dropout=force_dropout,
                )
            )
        x_list = torch.stack(x_list, dim=2)
        return x_list


class SharedIndependentTemporalUnet(nn.Module):
    agent_share_parameters = True

    def __init__(
        self,
        n_agents: int,
        horizon: int,
        history_horizon: int,  # not used
        transition_dim: int,
        dim: int = 128,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        returns_condition: bool = False,
        env_ts_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        residual_attn: bool = False,  # not used here
        max_path_length: int = 100,
    ):
        super().__init__()

        self.n_agents = n_agents

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition
        self.history_horizon = history_horizon

        self.net = TemporalUnet(
            horizon=horizon,
            history_horizon=history_horizon,
            transition_dim=transition_dim,
            dim=dim,
            dim_mults=dim_mults,
            returns_condition=returns_condition,
            env_ts_condition=env_ts_condition,
            condition_dropout=condition_dropout,
            kernel_size=kernel_size,
            max_path_length=max_path_length,
        )

    def forward(
        self,
        x,
        time,
        returns=None,
        env_timestep=None,
        attention_masks=None,
        use_dropout=True,
        force_dropout=False,
        **kwargs,
    ):
        """
        x : [ batch x horizon x agent x transition ]
        returns : [batch x agent x horizon]
        """

        assert x.shape[2] == self.n_agents, f"{x.shape}, {self.n_agents}"

        x = einops.rearrange(x, "b t a f -> b a t f")
        bs = x.shape[0]

        x = self.net(
            x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]),
            time=torch.cat([time for _ in range(x.shape[1])], dim=0),
            returns=torch.cat(
                [returns[:, :, a_idx] for a_idx in range(self.n_agents)], dim=0
            )
            if returns is not None
            else None,
            env_timestep=torch.cat([env_timestep for _ in range(x.shape[1])], dim=0)
            if env_timestep is not None
            else None,
            use_dropout=use_dropout,
            force_dropout=force_dropout,
        )
        x = x.reshape(bs, x.shape[0] // bs, x.shape[1], x.shape[2])
        x = einops.rearrange(x, "b a t f -> b t a f")
        return x


class SharedIndependentTemporalValue(nn.Module):
    agent_share_parameters = True

    def __init__(
        self,
        horizon,
        transition_dim,
        n_agents,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.net = TemporalValue(
            horizon=horizon,
            transition_dim=transition_dim,
            dim=dim,
            dim_mults=dim_mults,
            out_dim=out_dim,
        )

    def forward(self, x, time, *args):
        """
        x : [ batch x horizon x n_agents x transition ]
        """

        assert (
            x.shape[2] == self.n_agents
        ), f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"

        x = einops.rearrange(x, "b t a f -> b a t f")
        bs = x.shape[0]

        out = self.net(
            x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]),
            time=torch.cat([time for _ in range(x.shape[1])], dim=0),
        )
        out = out.reshape(bs, out.shape[0] // bs, out.shape[1])

        return out
