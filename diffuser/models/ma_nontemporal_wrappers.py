from typing import Tuple

import torch
import torch.nn as nn

from .nontemporal import BCMLPnet


class IndependentBCMLPnet(nn.Module):
    agent_share_parameters = False

    def __init__(
        self,
        n_agents: int,
        observation_dim: int,
        action_dim: int,
        dim: int = 128,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        init_w=1e-3,
        conditioned_std: bool = False,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.nets = nn.ModuleList(
            [
                BCMLPnet(
                    observation_dim=observation_dim,
                    action_dim=action_dim,
                    dim=dim,
                    dim_mults=dim_mults,
                    init_w=init_w,
                    conditioned_std=conditioned_std,
                )
                for _ in range(n_agents)
            ]
        )

    def forward(self, obs, deterministic=False):
        """
        obs : [ batch x 1 x agent x obs_dim ]
        """

        assert obs.shape[-2] == self.n_agents, f"{obs.shape}, {self.n_agents}"
        output_list = []
        for a_idx in range(self.n_agents):
            output_list.append(
                self.nets[a_idx](
                    obs[..., a_idx, :],
                    deterministic=deterministic,
                )
            )
        output_list = list(zip(*output_list))

        for idx in range(len(output_list)):
            if output_list[idx][0] is not None:
                output_list[idx] = torch.stack(output_list[idx], dim=-2)

        return output_list

    def get_log_prob(self, obs, acts):
        assert obs.shape[-2] == self.n_agents, f"{obs.shape}, {self.n_agents}"
        log_prob_list = []
        for a_idx in range(self.n_agents):
            log_prob_list.append(
                self.nets[a_idx].get_log_prob(
                    obs[..., a_idx, :],
                    acts[..., a_idx, :],
                )
            )
        log_prob_list = torch.stack(log_prob_list, dim=2)
        return log_prob_list


class SharedBCMLPnet(nn.Module):
    agent_share_parameters = True

    def __init__(
        self,
        n_agents: int,
        observation_dim: int,
        action_dim: int,
        dim: int = 128,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        init_w=1e-3,
        conditioned_std: bool = False,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.net = BCMLPnet(
            observation_dim=observation_dim,
            action_dim=action_dim,
            dim=dim,
            dim_mults=dim_mults,
            init_w=init_w,
            conditioned_std=conditioned_std,
        )

    def forward(self, obs, deterministic=False):
        """
        obs : [ batch x 1 x agent x obs_dim ]
        """

        assert obs.shape[-2] == self.n_agents, f"{obs.shape}, {self.n_agents}"
        output = self.net(obs)
        return output

    def get_log_prob(self, obs, acts):
        assert obs.shape[-2] == self.n_agents, f"{obs.shape}, {self.n_agents}"
        log_prob = self.net.get_log_prob(obs, acts)
        return log_prob
